from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime
import logging
from rq import Queue

from executors.run_compliance import run_compliance_workflow
from executors.run_feasibility import rfp_feasibility_analysis, sync_rfp_feasibility_analysis
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
            result = run_compliance_workflow(initial_state)
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


    @router.post("/compliance-matrix")
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
        result = run_compliance_workflow(initial_state)
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
