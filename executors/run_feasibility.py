from typing import Dict, List
import asyncio
import os
from langgraph.checkpoint.memory import MemorySaver
from graphs.feasibility import build_feasibility_graph
from services.vector_service import load_vector_store
from IPython.display import Image
from db import supabase

# Initialize memory for LangGraph
memory = MemorySaver()


async def run_feasibility_workflow(content: str) -> Dict:
    """
    Run the RFP feasibility analysis workflow.
    
    Args:
        content: The RFP content or compliance matrix to analyze
        
    Returns:
        Dict containing the workflow results
    """
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("Set the OPENAI_API_KEY environment variable")

        print("Starting RFP feasibility analysis...")

        # Load vector store
        vector_store = load_vector_store()
        print("Vector store loaded successfully")

        # Build the graph
        graph = build_feasibility_graph()
        workflow = graph.compile()
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
        return {"status": "success", "results": results}

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"Error in RFP analysis ({error_type}): {error_msg}")
        return {"status": "error", "results": [{"error": f"Analysis failed: {error_msg}"}]}


async def rfp_feasibility_analysis(content: str, document_id: int) -> Dict:
    """
    Run feasibility analysis on RFP content and return results.
    
    Args:
        content: The RFP content or compliance matrix
        document_id: The document ID in the database
        
    Returns:
        Dict containing the analysis results
    """
    # Run the analysis
    result = await run_feasibility_workflow(content)
    
    # Update the database
    if document_id:
        supabase.table("documents").update(
            {"feasibilityCheck": result.get("results", [])}
        ).eq("id", document_id).execute()
    
    return result


def sync_rfp_feasibility_analysis(content: str, document_id: int) -> Dict:
    """
    Synchronous wrapper for rfp_feasibility_analysis.
    This is used by the worker thread.
    
    Args:
        content: The RFP content or compliance matrix
        document_id: The document ID in the database
        
    Returns:
        Dict containing the analysis results
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(rfp_feasibility_analysis(content, document_id))
    loop.close()
    return result


def visualize_feasibility_graph(output_path: str = "feasibility_workflow.png") -> bool:
    """Generate and save a visualization of the feasibility analysis workflow."""
    try:
        # Create and compile the graph
        graph = build_feasibility_graph()
        app = graph.compile()

        # Generate and save the visualization
        image = Image(app.get_graph().draw_mermaid_png())
        with open(output_path, "wb") as f:
            f.write(image.data)
        print(f"📊 Workflow diagram saved as {output_path}")
        return True
    except Exception as e:
        print(f"⚠️ Could not save workflow diagram: {e}")
        return False 
