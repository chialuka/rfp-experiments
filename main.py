from typing import Dict, List, Optional
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from IPython.display import Image
import os
from pathlib import Path
from dotenv import load_dotenv
from reducto import Reducto
from prompts import RFP_ANALYSIS_STAGES

# Load environment variables
load_dotenv()

# Initialize memory for LangGraph
memory = MemorySaver()

# Initialize the LLM
llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")


class RFPState(Dict):
    """State for RFP analysis workflow"""

    pdf_filename: str
    pdf_data: Optional[Dict]
    current_stage: int
    previous_output: Optional[str]
    final_table: Optional[str]
    stage_outputs: Dict[str, str]  # Track outputs from each stage


@tool
def parse_pdf(filename: str) -> Dict:
    """Parse a PDF document using Reducto and extract the full text."""
    try:
        reducto_api_key = os.environ.get("REDUCTO_API_KEY")
        if not reducto_api_key:
            return {"error": "REDUCTO_API_KEY environment variable not set"}

        pdf_path = Path(filename)
        if not pdf_path.exists():
            return {"error": f"File not found: {filename}"}

        client = Reducto(api_key=reducto_api_key)
        upload = client.upload(file=pdf_path)
        parse_result = client.parse.run(document_url=upload)
        document_text = "\n".join([r.content for r in parse_result.result.chunks])

        return {
            "document_text": document_text,
            "document_filename": filename,
            "chunk_count": len(parse_result.result.chunks),
        }
    except Exception as e:
        return {"error": f"Error parsing PDF: {str(e)}"}


def initialize(state: Dict) -> RFPState:
    """Initialize the state with provided input and set up for PDF processing"""
    print(f"\n🚀 Initializing RFP analysis workflow...")

    # Get the pdf_filename from input or use a default
    pdf_filename = state.get("pdf_filename", "merged_documents")

    return {
        "pdf_filename": pdf_filename,
        "pdf_data": None,
        "current_stage": 0,
        "previous_output": None,
        "final_table": None,
        "stage_outputs": {},
    }


def process_pdf(state: RFPState) -> RFPState:
    """Process the PDF document and update state with the data"""
    print(f"\n🔍 Analyzing PDF...")
    print("📄 Extracting text from PDF...")

    # If pdf_data is already provided and valid (from a single PDF file), use it
    if state["pdf_data"] and "error" not in state["pdf_data"]:
        print(
            f"✅ Using provided PDF data with {len(state['pdf_data']['document_text'].split())} words"
        )
        return {**state, "current_stage": 0, "previous_output": None}

    # Otherwise parse the PDF by filename
    pdf_data = parse_pdf(state["pdf_filename"])

    if "error" in pdf_data:
        print(f"❌ Error analyzing PDF: {pdf_data['error']}")
        return {**state, "pdf_data": pdf_data}

    print(f"✅ Successfully extracted {len(pdf_data['document_text'].split())} words from PDF")
    print("🤖 Starting RFP analysis chain...")

    return {**state, "pdf_data": pdf_data, "current_stage": 0, "previous_output": None}


def execute_stage(stage_index: int):
    """Create a function to execute a specific stage of RFP analysis"""

    def _execute_stage(state: RFPState) -> RFPState:
        stage = RFP_ANALYSIS_STAGES[stage_index]
        stage_name = stage["name"]
        print(f"\n📋 Stage {stage_index + 1}: {stage_name}")

        # Check if pdf_data is valid for the first stage
        if stage_index == 0 and (state["pdf_data"] is None or "error" in state["pdf_data"]):
            error_msg = (
                "No valid PDF data available"
                if state["pdf_data"] is None
                else state["pdf_data"]["error"]
            )
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
                        pdf_data=state["pdf_data"]["document_text"]
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


def visualize(output_path: str = "rfp_workflow.png") -> bool:
    """Generate and save a visualization of the RFP analysis workflow."""
    try:
        # Create the graph
        graph = StateGraph(RFPState)

        # Add nodes for initialization and PDF processing
        graph.add_node("initialize", initialize)
        graph.add_node("process_pdf", process_pdf)

        # Add nodes for each analysis stage
        for i in range(len(RFP_ANALYSIS_STAGES)):
            graph.add_node(f"stage_{i}", execute_stage(i))

        # Set entry point
        graph.set_entry_point("initialize")

        # Add edges for initialization and PDF processing
        graph.add_edge("initialize", "process_pdf")
        graph.add_edge("process_pdf", "stage_0")

        # Add edges between stages
        for i in range(len(RFP_ANALYSIS_STAGES) - 1):
            graph.add_edge(f"stage_{i}", f"stage_{i + 1}")

        # Add edge from last stage to END
        graph.add_edge(f"stage_{len(RFP_ANALYSIS_STAGES) - 1}", END)

        # Compile the graph
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


def get_pdfs_from_storage() -> Optional[Dict]:
    """Get a single PDF file from the file_storage directory."""
    storage_dir = "file_storage"
    if not os.path.exists(storage_dir):
        print(f"❌ Directory '{storage_dir}' not found. Creating it...")
        os.makedirs(storage_dir)
        return None

    pdf_files = [
        os.path.join(storage_dir, f) for f in os.listdir(storage_dir) if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"❌ No PDF files found in '{storage_dir}' directory")
        return None

    if len(pdf_files) > 1:
        print(f"⚠️ Multiple PDFs found. Using only: {os.path.basename(pdf_files[0])}")

    selected_pdf = pdf_files[0]
    print(f"\n📚 Processing PDF: {os.path.basename(selected_pdf)}")
    return parse_pdf(selected_pdf)


def main():
    """Run the RFP analysis workflow"""
    # Generate workflow visualization
    visualize()

    # Get PDF from storage directory
    pdf_data = get_pdfs_from_storage()
    if not pdf_data:
        print("\nPlease place a PDF file in the 'file_storage' directory and run the script again.")
        return

    if "error" in pdf_data:
        print(f"\n❌ Error: {pdf_data['error']}")
        return

    print(f"\n✅ Successfully processed PDF: {pdf_data['document_filename']}")
    print("🤖 Starting RFP analysis chain...")

    # Create initial state with all required fields
    initial_state = {
        "pdf_filename": pdf_data["document_filename"],
        "pdf_data": pdf_data,
        "current_stage": 0,
        "previous_output": None,
        "final_table": None,
        "stage_outputs": {},
    }

    # Create the graph
    graph = StateGraph(RFPState)

    # Add nodes for initialization and PDF processing
    graph.add_node("initialize", initialize)
    graph.add_node("process_pdf", process_pdf)

    # Add nodes for each analysis stage
    for i in range(len(RFP_ANALYSIS_STAGES)):
        graph.add_node(f"stage_{i}", execute_stage(i))

    # Set entry point
    graph.set_entry_point("initialize")

    # Add edges for initialization and PDF processing
    graph.add_edge("initialize", "process_pdf")
    graph.add_edge("process_pdf", "stage_0")

    # Add edges between stages
    for i in range(len(RFP_ANALYSIS_STAGES) - 1):
        graph.add_edge(f"stage_{i}", f"stage_{i + 1}")

    # Add edge from last stage to END
    graph.add_edge(f"stage_{len(RFP_ANALYSIS_STAGES) - 1}", END)

    # Compile the graph with proper checkpointer configuration
    app = graph.compile(checkpointer=memory)

    # Run the workflow with thread_id
    result = app.invoke(initial_state, {"configurable": {"thread_id": "1"}})

    # Save the final table
    if result["final_table"]:
        output_file = "RFP_requirements.txt"
        with open(output_file, "w") as f:
            f.write(result["final_table"])
        print(f"\n✅ Analysis complete! Results saved to {output_file}")
    else:
        print("\n❌ Analysis failed to complete successfully.")

    # Save debug log with all stage outputs
    debug_log_file = "debug_stage_outputs.txt"
    with open(debug_log_file, "w") as f:
        f.write("RFP Analysis Debug Log\n")
        f.write("=" * 80 + "\n\n")
        for stage_name, output in result["stage_outputs"].items():
            f.write(f"Stage: {stage_name}\n")
            f.write("-" * 80 + "\n")
            f.write(output)
            f.write("\n\n" + "=" * 80 + "\n\n")
    print(f"📝 Debug log saved to {debug_log_file}")


if __name__ == "__main__":
    main()
