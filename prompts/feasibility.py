"""
This file contains all the prompts used to check the feasibility of the organization to meet the requirements.
"""

EXTRACT_REQUIREMENTS_PROMPT = """
    Extract vendor requirements from the following content. For each requirement, identify:
    1. Page number (use "N/A" if not specified)
    2. Section name
    3. The exact requirement text
    4. Obligation verb (shall, must, will, should, may, etc.)
    5. Obligation level (Mandatory, Conditional, Recommended, Optional)
    6. Any cross-references
    7. Whether it needs human review

    Format each requirement as a JSON object with these keys:
    {{
        "page": "page number or N/A",
        "section": "section name",
        "requirement_text": "the full text of the requirement",
        "obligation_verb": "the key verb",
        "obligation_level": "Mandatory/Conditional/Recommended/Optional",
        "cross_references": "any references or None",
        "human_review_flag": "Yes - reason" or "No"
    }}

    Return a JSON array of all requirements.
    
    CONTENT:
    {content}
"""

ASSESS_FEASIBILITY_PROMPT = """
    You are a feasibility analyst deciding if the organization can meet a requirement.

    <Requirement>
    {requirement}
    </Requirement>

    <Context>
    {context}
    </Context>

    Answer strictly as JSON with keys: feasible (Yes|No|Uncertain), reason, citations (array).
"""
