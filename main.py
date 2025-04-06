from typing import Annotated
from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from IPython.display import Image


class ChatState(TypedDict):
    """Simple state that only tracks messages."""

    messages: Annotated[list, add_messages]


# Initialize the LLM
llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")

# Create the graph
graph = StateGraph(ChatState)


# Define the chatbot function
def chatbot(state: ChatState):
    return {"messages": [llm.invoke(state["messages"])]}


# Add the chatbot node
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")

# Compile the graph
app = graph.compile()


def chat(message: str):
    """Simple function to send a message and get a response."""
    state = ChatState(messages=[{"role": "user", "content": message}])
    result = app.invoke(state)
    return result["messages"][-1].content


def visualize(output_path: str = "chat_workflow.png"):
    """Generate and save a visualization of the workflow."""
    image = Image(app.get_graph().draw_mermaid_png())
    with open(output_path, "wb") as f:
        f.write(image.data)
    print(f"📊 Workflow diagram saved as {output_path}")


if __name__ == "__main__":
    # Generate visualization
    visualize()

    # Start chat loop
    while True:
        message = input("\nEnter your message (or 'quit' to exit): ")
        if message.lower() == "quit":
            break
        response = chat(message)
        print(f"\nResponse: {response}")
