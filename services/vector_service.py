import os
import asyncio
from typing import List, Dict, Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Constants
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# For LangChain operations
langchain_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


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


def reset_vector_store(scope: str = "namespace") -> Dict:
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
            dimension=3072,  # 3072 is the dimension of the embedding model
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
