from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
import getpass
import os
import asyncio


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


async def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    print(f"Loaded {len(pages)} pages from {file_path}")
    return pages


def split_text(pages: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(pages)


def create_vector_store(documents: List[Document]) -> VectorStore:
    # Index chunks
    _ = vector_store.add_documents(documents)
    return vector_store


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    prompt = hub.pull("rlm/rag-prompt")
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


def create_rag_graph() -> StateGraph:
    # Define prompt for question-answering
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph


async def retrieve_answer(file_path: str, question: str):
    pages = await load_pdf(file_path)
    splits = split_text(pages)
    create_vector_store(splits)
    graph = create_rag_graph()
    result = graph.invoke({"question": question})
    return result


async def main(file_path: str, question: str):
    # Load PDF
    result = await retrieve_answer(file_path, question)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
