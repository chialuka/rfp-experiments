from typing import Dict
from langgraph.checkpoint.memory import MemorySaver
from graphs.compliance import create_compliance_graph
from IPython.display import Image

# Initialize memory for LangGraph
memory = MemorySaver()


def run_compliance_workflow(initial_state: Dict) -> Dict:
    """
    Run the RFP compliance matrix workflow with the given initial state.
    
    Args:
        initial_state: The initial state for the workflow
        
    Returns:
        Dict containing the workflow results
    """
    # Create and compile the graph
    graph = create_compliance_graph()
    app = graph.compile(checkpointer=memory)

    # Run the workflow with thread_id
    result = app.invoke(initial_state, {"configurable": {"thread_id": "compliance-1"}})

    return result


def visualize_compliance_graph(output_path: str = "compliance_workflow.png") -> bool:
    """Generate and save a visualization of the compliance matrix workflow."""
    try:
        # Create and compile the graph
        graph = create_compliance_graph()
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
