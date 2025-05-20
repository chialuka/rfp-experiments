from typing import Dict, Any
from executors.run_feasibility import sync_rfp_feasibility_analysis

# This file defines task functions that could be used by workers
# Currently, we directly import sync_rfp_feasibility_analysis in the API
# But this file provides a place to define additional worker tasks in the future

def run_feasibility_analysis(content: str, document_id: int) -> Dict[str, Any]:
    """
    Background task to run feasibility analysis.
    
    Args:
        content: The RFP content or compliance matrix
        document_id: The document ID in the database
        
    Returns:
        Dict containing the analysis results
    """
    return sync_rfp_feasibility_analysis(content, document_id) 
