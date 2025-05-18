from __future__ import annotations

# Standard library imports
import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, TypedDict, Dict, Any
from uuid import uuid4
import asyncio

# Third-party imports
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import Settings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Local imports
from prompts.feasibility import EXTRACT_REQUIREMENTS_PROMPT, ASSESS_FEASIBILITY_PROMPT
from db import supabase

# Initialize the ChromaDB client
chroma_client = chromadb.PersistentClient(
    path=".vectorstore/chroma", settings=Settings(allow_reset=True)
)

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Constants
CHAT_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


# Data models
class Requirement(BaseModel):
    page: str = Field(..., alias="Page")
    section: str = Field(..., alias="Section")
    requirement_text: Optional[str] = Field(None, alias="Requirement Text")
    obligation_verb: Optional[str] = Field(None, alias="Obligation Verb")
    obligation_level: Optional[str] = Field(None, alias="Obligation Level")
    cross_references: Optional[str] = Field(None, alias="Cross-References")
    human_review_flag: Optional[str] = Field(None, alias="Human Review Flag")

    # Convenience helpers
    def to_query(self) -> str:
        return (
            f"Requirement: {self.requirement_text}\n"
            f"Obligation level: {self.obligation_level}. "
            f"Have we previously met an equivalent requirement? Provide proofs if yes."
        )


class RFPState(TypedDict):
    content: str  # Raw RFP text
    requirements: List[Requirement]  # All extracted requirements
    current_req_index: int  # Index of the current requirement being processed
    results: List[dict]  # Accumulated results
    vector_store: Any  # Reference to vector store


# Global instances
# For ChromaDB native operations
chroma_embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL_NAME
)

# For LangChain operations
langchain_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

chat_llm = ChatOpenAI(
    model=CHAT_MODEL_NAME,
    temperature=0,  # Lower temperature for more deterministic outputs
    model_kwargs={"response_format": {"type": "json_object"}},  # Force JSON responses
)
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
        pages = await loader.aload()
        for page in pages:
            page.metadata.setdefault("rfp_source", url)
        docs.extend(TEXT_SPLITTER.split_documents(pages))
    return docs


def reset_vector_store(scope: str = "namespace") -> None:
    """Reset Pinecone data.

    scope = "namespace" → clear one namespace
    scope = "index"     → delete and recreate entire index
    """
    if scope == "namespace":
        index = pc.Index("z-bids")
        index.delete(delete_all=True, namespace="rfp")
    elif scope == "index":
        pc.delete_index("z-bids")
        pc.create_index(
            name="z-bids",
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws", region=os.getenv("PINECONE_ENV", "us-east-1")
            ),
        )
    else:
        raise ValueError("scope must be 'namespace' or 'index'")
    return {"status": "success"}


async def create_vector_store(file_urls: List[str]) -> Dict[str, Any]:
    """Create or update a Chroma index with documents."""
    documents = await load_documents_from_urls(file_urls)

    if "z-bids" not in pc.list_indexes().names():
        pc.create_index(
            name="z-bids",
            dimension=3072, # 3072 is the dimension of the embedding model
            metric="cosine",
            spec=ServerlessSpec(  # free tier
                cloud="aws", region=os.getenv("PINECONE_ENV", "us-east-1")
            ),
        )
    host = pc.describe_index("z-bids").host
    print(f"host:{host}")
    index = pc.Index(host=host)

    vector_store = PineconeVectorStore(
        index,  # the low-level Index object
        embedding=langchain_embeddings,
        text_key="content",  # which metadata field stores the raw text
        namespace="rfp",  # optional logical partition
    )

    # Add documents using LangChain interface
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    # Return serializable result instead of the vector store object
    return {
        "status": "success",
        "documents_processed": len(documents),
        "document_sources": file_urls,
    }


def load_vector_store() -> VectorStore:
    """Load the existing Chroma vector store."""
    if "z-bids" not in pc.list_indexes().names():
        raise RuntimeError(
            f"Pinecone index 'z-bids' not found - run create_vector_store first."
        )
    host = pc.describe_index("z-bids").host
    print(f"host:{host}")
    index = pc.Index(host=host)

    # Create a LangChain Pinecone instance that wraps the Pinecone collection
    return PineconeVectorStore(
        index,  # the low-level Index object
        embedding=langchain_embeddings,
        text_key="content",  # which metadata field stores the raw text
        namespace="rfp",  # optional logical partition
    )


# Graph node functions
async def extract_requirements(state: RFPState):
    """Extract requirements from the content using LLM."""
    try:
        llm_prompt = EXTRACT_REQUIREMENTS_PROMPT.format(content=state["content"])
        response = await chat_llm.ainvoke(llm_prompt)

        # Extract the JSON array from the response
        json_match = re.search(r"\[\s*\{.*\}\s*\]", response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            requirements_data = json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            response_json = json.loads(response.content)
            # Check if the response has a "requirements" key
            if "requirements" in response_json:
                requirements_data = response_json["requirements"]
            else:
                requirements_data = response_json

        print(f"Extracted {len(requirements_data)} requirements")

        requirements = []
        for req_data in requirements_data:
            try:
                requirement = Requirement(
                    Page=req_data.get("page", "N/A"),
                    Section=req_data.get("section", "Unknown"),
                    **{
                        "Requirement Text": req_data.get("requirement_text", ""),
                        "Obligation Verb": req_data.get("obligation_verb", ""),
                        "Obligation Level": req_data.get("obligation_level", ""),
                        "Cross-References": req_data.get("cross_references", "None"),
                        "Human Review Flag": req_data.get("human_review_flag", "No"),
                    },
                )
                requirements.append(requirement)
            except Exception as req_error:
                print(f"Error parsing requirement: {req_error}")
                continue

        print(f"Found {len(requirements)} valid requirements")

        return {
            "requirements": requirements,
            "current_req_index": 0 if requirements else -1,
            "results": [],
        }
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {"requirements": [], "current_req_index": -1, "results": []}


async def process_requirement(state: RFPState):
    """Process a single requirement."""
    try:
        # Get the current requirement
        requirements = state["requirements"]
        current_index = state.get("current_req_index", -1)

        # Check if we have valid index
        if current_index < 0 or current_index >= len(requirements):
            print("No more requirements to process")
            return {"current_req_index": -1}

        req = requirements[current_index]
        print(
            f"Processing requirement {current_index+1}/{len(requirements)}: {req.requirement_text[:50]}..."
        )

        # Get the vector store from state
        vector_store = state.get("vector_store")
        if not vector_store:
            print("Vector store not found in state")
            return {"current_req_index": -1}

        # Create query
        query = req.to_query()

        # Retrieve relevant documents
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 8})
            documents = retriever.invoke(query)
            print(f"Retrieved {len(documents)} documents")
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            documents = []

        # Assess feasibility
        try:
            verdict = await assess_requirement(req, documents)
            print(f"Assessment: {verdict.get('feasible', 'Unknown')}")
        except Exception as e:
            print(f"Error during assessment: {str(e)}")
            verdict = {
                "feasible": "Uncertain",
                "reason": f"Error during assessment: {str(e)}",
                "citations": [],
            }

        # Collect result
        result = {
            "req_no": current_index + 1,
            "section": req.section,
            "requirement": req.requirement_text,
            "feasible": verdict.get("feasible", "Uncertain"),
            "reason": verdict.get("reason", "No assessment available"),
            "citations": "; ".join(verdict.get("citations", [])),
        }

        # Add to results and move to next requirement
        return {
            "results": state.get("results", []) + [result],
            "current_req_index": (
                current_index + 1 if current_index + 1 < len(requirements) else -1
            ),
        }
    except Exception as e:
        print(f"Error processing requirement: {e}")
        return {
            "current_req_index": state.get("current_req_index", -1) + 1,
            "results": state.get("results", []),
        }


def should_continue(state: RFPState):
    """Decide whether to process another requirement or finish."""
    current_index = state.get("current_req_index", -1)

    if current_index >= 0:
        print(f"Continue processing next requirement ({current_index + 1})")
        return "continue"
    else:
        print("No more requirements to process, ending")
        return "end"


# Function to assess a single requirement
async def assess_requirement(
    requirement: Requirement, documents: List[Document]
) -> Dict:
    """Assess feasibility of a requirement based on retrieved documents."""
    try:
        ctx_text = (
            "\n\n".join(doc.page_content for doc in documents) if documents else ""
        )

        # If no context was retrieved, note that in the verdict
        if not ctx_text:
            return {
                "feasible": "Uncertain",
                "reason": "No relevant context was found to assess this requirement.",
                "citations": [],
            }

        prompt = ASSESS_FEASIBILITY_PROMPT.format(
            requirement=requirement.requirement_text, context=ctx_text
        )
        response = await chat_llm.ainvoke(prompt)
        verdict_json = json.loads(response.content)
        return verdict_json
    except Exception as e:
        print(f"Error during assessment: {e}")
        return {
            "feasible": "Uncertain",
            "reason": f"Error during assessment: {str(e)}",
            "citations": [],
        }


# Graph builder and main function
def build_graph() -> StateGraph:
    """Build the processing graph for RFP analysis."""
    graph = StateGraph(RFPState)

    # Add nodes
    graph.add_node("extract", extract_requirements)
    graph.add_node("process", process_requirement)

    # Add edges
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "process")

    # Add conditional edge for continued processing
    graph.add_conditional_edges(
        "process",
        should_continue,
        {
            "continue": "process",  # Loop back to process next requirement
            "end": END,
        },
    )

    return graph.compile()


async def rfp_feasibility_analysis(content: str, document_id: int) -> List[dict]:
    """Analyze RFP requirements against past RFPs using streamed LangGraph approach.

    Args:
        content: Text content containing requirements (any format)

    Returns:
        List of verdict dictionaries for each requirement
    """
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("Set the OPENAI_API_KEY environment variable")

        print("Starting RFP feasibility analysis...")

        # Load vector store
        vector_store = load_vector_store()
        print("Vector store loaded successfully")

        # Build the graph
        workflow = build_graph()
        print("Graph built successfully")

        # Initialize state with vector store
        initial_state = {
            "content": content,
            "vector_store": vector_store,  # Pass vector store in initial state
        }

        # Execute the graph using astream to avoid recursion limits
        print("Streaming graph execution")
        results: list[dict] = []

        async for st in workflow.astream(initial_state, stream_mode="values"):
            if len(st.get("results", [])) > len(results):
                results = st["results"]

        print(f"Analysis complete. Processed {len(results)} requirements")

        # Add this at the end of the function
        supabase.table("documents").update(
            {
                "feasibilityCheck": results,
            }
        ).eq("id", document_id).execute()

        return {"status": "success", "results": results}

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"Error in RFP analysis ({error_type}): {error_msg}")
        return [{"error": f"Analysis failed: {error_msg}"}]


def sync_rfp_feasibility_analysis(content, document_id):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(rfp_feasibility_analysis(content, document_id))
    loop.close()
    return result
