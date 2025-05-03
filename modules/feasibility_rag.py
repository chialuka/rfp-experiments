from __future__ import annotations

# Standard library imports
import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Optional, TypedDict

# Third-party imports
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# Constants
CHAT_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Prompt templates
PROMPT_TEMPLATE = (
    "You are a feasibility analyst deciding if the organisation can meet a requirement.\n"
    "\n"
    "<Requirement>\n{requirement}\n</Requirement>\n"
    "\n"
    "<Context>\n{context}\n</Context>\n"
    "\n"
    "Answer strictly as JSON with keys: feasible (Yes|No|Uncertain), reason, citations (array)."
)

# Data models
class Requirement(BaseModel):
    page: str = Field(..., alias="Page")
    section: str = Field(..., alias="Section")
    requirement_text: str = Field(..., alias="Requirement Text")
    obligation_verb: str = Field(..., alias="Obligation Verb")
    obligation_level: str = Field(..., alias="Obligation Level")
    cross_references: Optional[str] = Field(None, alias="Cross-References")
    human_review_flag: Optional[str] = Field(None, alias="Human Review Flag")

    # Convenience helpers
    def to_query(self) -> str:
        return (
            f"Requirement: {self.requirement_text}\n"
            f"Obligation level: {self.obligation_level}. "
            f"Have we previously met an equivalent requirement? Provide proofs if yes."
        )


class ComplianceMatrix(BaseModel):
    rfp_id: str = "UNSET"
    requirements: List[Requirement]


class RFPState(TypedDict):
    content: str  # Raw RFP text
    requirements: List[Requirement]  # Extracted requirements
    requirement: Requirement  # Current requirement being processed
    query: str
    context: List[Document]
    verdict: dict
    results: List[dict]  # Accumulated results


# Global instances
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
chat_llm = init_chat_model(CHAT_MODEL_NAME, model_provider="openai")
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# Utility functions
async def load_documents_from_urls(urls: List[str]) -> List[Document]:
    """Load and split documents from URLs."""
    docs: List[Document] = []
    for url in urls:
        print(f"Loading {url} …")
        loader = PyPDFLoader(url)
        pages = await loader.alazy_load()
        for page in pages:
            page.metadata.setdefault("rfp_source", url)
        docs.extend(TEXT_SPLITTER.split_documents(pages))
    return docs


async def create_vector_store(file_urls: List[str]) -> VectorStore:
    """Create or update a Chroma index with documents."""
    docs = await load_documents_from_urls(file_urls)
    chroma_dir = Path(".vectorstore/chroma")
    
    if chroma_dir.exists():
        vs = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
        vs.add_documents(docs)
    else:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        vs = Chroma.from_documents(docs, embeddings, persist_directory=str(chroma_dir))

    vs.persist()
    return vs


def load_vector_store() -> VectorStore:
    """Load the existing Chroma vector store."""
    chroma_dir = Path(".vectorstore/chroma")
    if not chroma_dir.exists():
        raise ValueError("Vector store not found. Please build it first.")
    return Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)


# Graph node functions - in order of workflow execution
def extract_requirements(state: RFPState):
    """Extract requirements from the content using LLM."""
    prompt = """
    Extract vendor requirements from the following content. For each requirement, identify:
    1. Page number (use "N/A" if not specified)
    2. Section name
    3. The exact requirement text
    4. Obligation verb (shall, must, will, should, may, etc.)
    5. Obligation level (Mandatory, Conditional, Recommended, Optional)
    6. Any cross-references
    7. Whether it needs human review

    Format each requirement as a JSON object with these keys:
    {
        "page": "page number or N/A",
        "section": "section name",
        "requirement_text": "the full text of the requirement",
        "obligation_verb": "the key verb",
        "obligation_level": "Mandatory/Conditional/Recommended/Optional",
        "cross_references": "any references or None",
        "human_review_flag": "Yes - reason" or "No"
    }

    Return a JSON array of all requirements.
    
    CONTENT:
    {content}
    """
    
    llm_prompt = prompt.format(content=state["content"])
    response = chat_llm.invoke(llm_prompt)
    
    try:
        # Extract the JSON array from the response
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            requirements_data = json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            requirements_data = json.loads(response.content)
        
        requirements = []
        for req_data in requirements_data:
            requirement = Requirement(
                Page=req_data.get("page", "N/A"),
                Section=req_data.get("section", "Unknown"),
                Requirement_Text=req_data.get("requirement_text", ""),
                Obligation_Verb=req_data.get("obligation_verb", ""),
                Obligation_Level=req_data.get("obligation_level", ""),
                Cross_References=req_data.get("cross_references", "None"),
                Human_Review_Flag=req_data.get("human_review_flag", "No")
            )
            requirements.append(requirement)
        
        return {"requirements": requirements}
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"LLM response: {response.content}")
        return {"requirements": []}


def process_requirement(state: RFPState):
    """Process the next requirement or end if all are processed."""
    if not state["requirements"]:
        # No requirements to process
        return {"results": []}
    
    current_req = state["requirements"][0]
    remaining = state["requirements"][1:] if len(state["requirements"]) > 1 else []
    
    return {
        "requirement": current_req,
        "requirements": remaining
    }


def formulate_query(state: RFPState):
    """Create a query from the current requirement."""
    req: Requirement = state["requirement"]
    return {"query": req.to_query()}


def make_retrieve_node(vector_store: VectorStore):
    """Factory function to create a retrieve node with access to vector store."""
    def retrieve(state: RFPState):
        """Retrieve relevant documents for the current query."""
        vs_retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 8,
                "filter": {
                    "obligation_level": state["requirement"].obligation_level.lower()
                },
            }
        )
        ctx_docs = vs_retriever.invoke(state["query"])
        return {"context": ctx_docs}

    return retrieve


def assess(state: RFPState):
    """Assess feasibility of meeting the requirement based on retrieved context."""
    req: Requirement = state["requirement"]
    ctx_text = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = PROMPT_TEMPLATE.format(requirement=req.requirement_text, context=ctx_text)
    response = chat_llm.invoke(prompt)
    verdict_json = json.loads(response.content)
    return {"verdict": verdict_json}


def collect_result(state: RFPState):
    """Collect the result for the current requirement."""
    current_results = state.get("results", [])
    current_verdict = state["verdict"]
    
    result = {
        "req_no": len(current_results) + 1,
        "section": state["requirement"].section,
        "requirement": state["requirement"].requirement_text,
        "feasible": current_verdict["feasible"],
        "reason": current_verdict["reason"],
        "citations": "; ".join(current_verdict.get("citations", [])),
    }
    
    return {"results": current_results + [result]}


def should_continue(state: RFPState):
    """Decide whether to process another requirement or finish."""
    if state["requirements"]:
        return "process_next"
    else:
        return "end"


# Graph builder and main function
def build_graph(vector_store: VectorStore) -> StateGraph:
    """Build the complete graph for RFP analysis."""
    graph = StateGraph(RFPState)
    
    # Extract requirements
    graph.add_node("extract", extract_requirements)
    
    # Process requirements in a loop
    graph.add_node("process", process_requirement)
    graph.add_node("query", formulate_query)
    graph.add_node("retrieve", make_retrieve_node(vector_store))
    graph.add_node("assess", assess)
    graph.add_node("collect", collect_result)
    
    # Add edges
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "process")
    graph.add_edge("process", "query")
    graph.add_edge("query", "retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_edge("assess", "collect")
    
    # Create conditional branching
    graph.add_conditional_edges(
        "collect",
        should_continue,
        {
            "process_next": "process",
            "end": END
        }
    )
    
    return graph.compile()


async def rfp_feasibility_analysis(content: str) -> List[dict]:
    """Analyze RFP requirements against past RFPs.
    
    Args:
        content: Text content containing requirements (any format)
        
    Returns:
        List of verdict dictionaries for each requirement
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Set the OPENAI_API_KEY environment variable")

    # 1. Load vector store
    vector_store = load_vector_store()

    # 2. Build and run RAG pipeline with integrated extraction
    rag_graph = build_graph(vector_store)
    
    # 3. Run the workflow with content as input
    result = await rag_graph.invoke({"content": content})
    
    return result["results"]

