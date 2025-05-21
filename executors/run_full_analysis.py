from typing import Dict
import asyncio
from fastapi import HTTPException
import logging

from executors.run_compliance import run_compliance_workflow
from executors.run_feasibility import rfp_feasibility_analysis
from db import supabase

logger = logging.getLogger(__name__)


async def run_full_analysis(pdf_file_content: str, document_id: int) -> Dict:
    """
    Run full analysis on RFP content and return results.
    """
    try:
        # Create initial state
        initial_state = {
            "pdf_filename": None,
            "pdf_data": pdf_file_content,
            "current_stage": 0,
            "previous_output": None,
            "final_table": None,
            "stage_outputs": {},
        }

        # Run the workflow
        result = run_compliance_workflow(initial_state)
        compliance_matrix = result.get("final_table")

        feasibility_result = await rfp_feasibility_analysis(
            compliance_matrix, document_id
        )

        # Update the database with the compliance matrix
        supabase.table("documents").update(
            {
                "complianceMatrix": compliance_matrix,
                "feasibilityCheck": feasibility_result.get("results"),
            }
        ).eq("id", document_id).execute()

        return {
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error analyzing RFP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def run_full_analysis_sync(pdf_file_content: str, document_id: int) -> Dict:
    """
    Synchronous wrapper for run_full_analysis.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(run_full_analysis(pdf_file_content, document_id))
    loop.close()
    return result
