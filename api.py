from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pathlib import Path
from typing import Dict
from contextlib import asynccontextmanager
import logging
import uvicorn

from main import run_workflow
from modules.feasibility_rag import (
    rfp_feasibility_analysis,
    create_vector_store,
    reset_vector_store,
)
from db import supabase

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RFP Analysis API")
    yield
    logger.info("Shutting down RFP Analysis API")


app = FastAPI(
    title="RFP Analysis API",
    description="API for analyzing RFP documents using LangGraph and Claude",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create file_storage directory if it doesn't exist
STORAGE_DIR = Path("file_storage")
STORAGE_DIR.mkdir(exist_ok=True)


class ComplianceRequest(BaseModel):
    pdf_file_content: str
    document_id: int


class FeasibilityRequest(BaseModel):
    content: str
    document_id: int


class VectorizeRequest(BaseModel):
    file_path: List[str]


@app.post("/rfp/analyze")
async def analyze_rfp(request: ComplianceRequest) -> Dict:
    """
    Analyze an RFP document. Generate a compliance matrix and feasibility checklist for the document.

    Args:
        request: The request containing the PDF file content

    Returns:
        Dict containing the analysis results
    """
    print(f"PDF content: {request.document_id}")

    try:
        # Create initial state
        initial_state = {
            "pdf_filename": None,
            "pdf_data": request.pdf_file_content,
            "current_stage": 0,
            "previous_output": None,
            "final_table": None,
            "stage_outputs": {},
        }

        # Run the workflow
        result = run_workflow(initial_state)
        compliance_matrix = result.get("final_table")

        feasibility_result = await rfp_feasibility_analysis(
            compliance_matrix, request.document_id
        )

        # Update the database with the compliance matrix
        supabase.table("documents").update(
            {
                "complianceMatrix": compliance_matrix,
                "feasibilityCheck": feasibility_result.get("results"),
            }
        ).eq("id", request.document_id).execute()

        return {
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error analyzing RFP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rfp/compliance-matrix")
async def get_compliance_matrix(request: ComplianceRequest) -> Dict:
    """
    Get the compliance matrix for an RFP document.
    """
    # Create initial state
    initial_state = {
        "pdf_filename": None,
        "pdf_data": request.pdf_file_content,
        "current_stage": 0,
        "previous_output": None,
        "final_table": None,
        "stage_outputs": {},
    }

    # Run the workflow
    result = run_workflow(initial_state)
    compliance_matrix = result.get("final_table")

    # Update the database with the compliance matrix
    supabase.table("documents").update(
        {
            "complianceMatrix": compliance_matrix,
        }
    ).eq("id", request.document_id).execute()

    return {
        "status": "success",
        "complianceMatrix": compliance_matrix,
    }


@app.post("/rfp/vectorize")
async def vectorize(request: VectorizeRequest) -> Dict:
    """
    Vectorize a list of RFP documents.
    """
    print(f"Processing file paths: {request.file_path}")
    result = await create_vector_store(request.file_path)
    return result


@app.post("/rfp/feasibility")
async def check_feasibility(request: FeasibilityRequest) -> Dict:
    """
    Check the feasibility of an RFP document.
    """
    result = await rfp_feasibility_analysis(request.content, request.document_id)
    supabase.table("documents").update(
        {
            "feasibilityCheck": result.get("results"),
        }
    ).eq("id", request.document_id).execute()
    return {"status": "success", "result": result.get("results")}


@app.get("/rfp/vector/reset")
async def reset_store() -> Dict:
    """
    Reset the vector store.
    """
    result = reset_vector_store()
    return result


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=2500, reload=True)
