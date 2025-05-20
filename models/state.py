from typing import Dict, List, Optional, TypedDict, Any
from pydantic import BaseModel, Field


class RFPState(Dict):
    """State for RFP analysis workflow"""

    pdf_filename: str
    pdf_data: str
    current_stage: int
    previous_output: Optional[str]
    final_table: Optional[str]
    stage_outputs: Dict[str, str]  # Track outputs from each stage


class Requirement(BaseModel):
    page: str = Field(..., alias="Page")
    section: str = Field(..., alias="Section")
    requirement_text: Optional[str] = Field(None, alias="Requirement Text")
    obligation_verb: Optional[str] = Field(None, alias="Obligation Verb")
    obligation_level: Optional[str] = Field(None, alias="Obligation Level")
    cross_references: Optional[str] = Field(None, alias="Cross-References")
    human_review_flag: Optional[str] = Field(None, alias="Human Review Flag")

    # Convenience helpers
    def to_query(self) -> str:
        return (
            f"Requirement: {self.requirement_text}\n"
            f"Obligation level: {self.obligation_level}. "
            f"Have we previously met an equivalent requirement? Provide proofs if yes."
        )


class FeasibilityState(TypedDict):
    content: str  # Raw RFP text
    requirements: List[Requirement]  # All extracted requirements
    current_req_index: int  # Index of the current requirement being processed
    results: List[dict]  # Accumulated results
    vector_store: Any  # Reference to vector store 
