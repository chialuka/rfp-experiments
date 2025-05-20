from typing import Dict
from langgraph.graph import StateGraph, START, END
from models.state import RFPState
from prompts.compliance_matrix import RFP_ANALYSIS_STAGES
from services.llm_service import get_anthropic_llm

# Initialize the LLM
llm = get_anthropic_llm()


def execute_stage(stage_index: int):
    """Create a function to execute a specific stage of RFP analysis"""

    def _execute_stage(state: RFPState) -> RFPState:
        stage = RFP_ANALYSIS_STAGES[stage_index]
        stage_name = stage["name"]
        print(f"\n📋 Stage {stage_index + 1}: {stage_name}")

        # Check if pdf_data is valid for the first stage
        if stage_index == 0 and state["pdf_data"] is None:
            error_msg = "No PDF data available"
            print(f"❌ Error: {error_msg}")
            return state

        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": stage["system_prompt"]},
        ]

        # For stage 0, use pdf_data. For other stages, use previous_output
        if stage_index == 0:
            messages.append(
                {
                    "role": "user",
                    "content": stage["message_template"].format(
                        pdf_data=state["pdf_data"]
                    ),
                }
            )
        else:
            if "previous_output" not in state:
                print(f"❌ Error: No output from previous stage available")
                return state
                
            messages.append(
                {
                    "role": "user",
                    "content": stage["message_template"].format(
                        previous_output=state["previous_output"]
                    ),
                }
            )

        # Get response from LLM
        response = llm.invoke(messages)
        print(f"✅ Stage {stage_index + 1} complete")

        # Initialize stage_outputs if not present
        stage_outputs = state.get("stage_outputs", {})
        stage_outputs[stage_name] = response.content

        # Update state
        new_state = {
            **state,
            "current_stage": stage_index + 1,
            "previous_output": response.content,
            "stage_outputs": stage_outputs,
        }

        # Only set final_table in the last stage
        if stage_index == len(RFP_ANALYSIS_STAGES) - 1:
            new_state["final_table"] = response.content

        return new_state

    return _execute_stage


def initialize(state: Dict) -> RFPState:
    """Initialize the state with provided input and set up for PDF processing"""
    print(f"\n🚀 Initializing RFP analysis workflow...")

    # Get the pdf_content from input or use a default
    pdf_content = state.get("pdf_data", "")

    return {
        "pdf_filename": None,
        "pdf_data": pdf_content,
        "current_stage": 0,
        "previous_output": None,
        "final_table": None,
        "stage_outputs": {},
    }


def create_compliance_graph() -> StateGraph:
    """Create and configure the RFP analysis workflow graph."""
    # Create the graph
    graph = StateGraph(RFPState)

    # Add nodes for initialization
    graph.add_node("initialize", initialize)

    # Add nodes for each analysis stage
    for i in range(len(RFP_ANALYSIS_STAGES)):
        graph.add_node(f"stage_{i}", execute_stage(i))

    # Set entry point
    graph.set_entry_point("initialize")

    # Add edges for initialization
    graph.add_edge("initialize", "stage_0")

    # Add edges between stages
    for i in range(len(RFP_ANALYSIS_STAGES) - 1):
        graph.add_edge(f"stage_{i}", f"stage_{i + 1}")

    # Add edge from last stage to END
    graph.add_edge(f"stage_{len(RFP_ANALYSIS_STAGES) - 1}", END)

    return graph 
