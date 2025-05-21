from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime
import logging
from rq import Queue

from executors.run_compliance import run_compliance_and_save_to_db
from executors.run_feasibility import sync_rfp_feasibility_analysis
from executors.run_full_analysis import run_full_analysis_sync
from services.vector_service import create_vector_store, reset_vector_store
from db import supabase

# Initialize logging
logger = logging.getLogger(__name__)

# Define request models
class ComplianceRequest(BaseModel):
    pdf_file_content: str
    document_id: int


class FeasibilityRequest(BaseModel):
    content: str
    document_id: int


class VectorizeRequest(BaseModel):
    file_path: List[str]


def create_router(queue: Queue) -> APIRouter:
    """
    Create and configure the router with the provided queue.
    
    Args:
        queue: The RQ queue for background tasks
        
    Returns:
        Configured FastAPI router
    """
    router = APIRouter(prefix="/rfp", tags=["rfp"])

    @router.post("/analyze")
    async def analyze_rfp(request: ComplianceRequest) -> Dict:
        """
        Analyze an RFP document. Generate a compliance matrix and feasibility checklist for the document.

        Args:
            request: The request containing the PDF file content

        Returns:
            Dict containing the analysis results
        """
        print(f"Analyzing document: {request.document_id}")
        queue.enqueue(run_full_analysis_sync, request.pdf_file_content, request.document_id)
        return {"status": "processing"}


    @router.post("/compliance-matrix")
    async def get_compliance_matrix(request: ComplianceRequest) -> Dict:
        """
        Get the compliance matrix for an RFP document.
        """
        queue.enqueue(run_compliance_and_save_to_db, request.pdf_file_content, request.document_id)
        return {"status": "processing"}


    @router.post("/vectorize")
    async def vectorize(request: VectorizeRequest) -> Dict:
        """
        Vectorize a list of RFP documents.
        """
        print(f"Processing file paths: {request.file_path}")
        result = await create_vector_store(request.file_path)
        return result


    @router.post("/feasibility")
    async def check_feasibility(request: FeasibilityRequest) -> Dict:
        """
        Check the feasibility of an RFP document.
        """
        queue.enqueue(sync_rfp_feasibility_analysis, request.content, request.document_id)
        return {"status": "processing"}


    @router.delete("/vector/reset")
    async def reset_store() -> Dict:
        """
        Reset the vector store.
        """
        result = reset_vector_store()
        return result


    @router.get("/job-status/{job_id}")
    def get_job_status(job_id: str):
        job = queue.fetch_job(job_id)
        if not job:
            return {"status": "not_found"}

        status = job.get_status()
        response = {"status": status}

        if status == "finished":
            response["result"] = job.result
            response["complete"] = True

        return response


    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        }
        
    return router 
