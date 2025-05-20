import re
import json
from typing import Dict, List
import asyncio
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from models.state import FeasibilityState, Requirement
from services.llm_service import get_openai_llm
from services.vector_service import load_vector_store
from prompts.feasibility import EXTRACT_REQUIREMENTS_PROMPT, ASSESS_FEASIBILITY_PROMPT

# Initialize LLM
chat_llm = get_openai_llm(json_mode=True)


async def extract_requirements(state: FeasibilityState):
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


async def process_requirement(state: FeasibilityState):
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


def should_continue(state: FeasibilityState):
    """Decide whether to process another requirement or finish."""
    current_index = state.get("current_req_index", -1)

    if current_index >= 0:
        print(f"Continue processing next requirement ({current_index + 1})")
        return "continue"
    else:
        print("No more requirements to process, ending")
        return "end"


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


def build_feasibility_graph() -> StateGraph:
    """Build the processing graph for RFP analysis."""
    graph = StateGraph(FeasibilityState)

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

    return graph 
