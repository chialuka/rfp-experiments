"""
This file contains all the prompts used in the RFP Analysis workflow.
"""

# Define the RFP analysis stages
RFP_ANALYSIS_STAGES = [
    {
        "name": "Document Structure Analysis",
        "system_prompt": """Stage 1: Document Structure Analysis for Vendor-Specific Requirements 

IMPORTANT: Analyze ONLY the RFP document text provided below. Do NOT use any generic or assumed content. Your analysis must be based solely on the actual document content provided.

Steps for Analysis:
1. First read and understand the complete document text provided
2. Identify the actual document's organization and structure as it appears in the text
3. Map out the specific sections present in this document that contain vendor requirements
4. Note the exact terminology and phrasing used in this document for vendor obligations

Focus Areas:
1. Document Organization:
   - Identify the actual sections in this document that instruct vendors
   - Map the real structure of vendor requirements as they appear
   - Note the document's specific approach to presenting requirements

2. Key Sections Present:
   - List only sections that actually exist in the provided document
   - Include exact section titles and numbers as they appear
   - Note where vendor obligations are concentrated

3. Terminology Analysis:
   - Extract actual directive verbs used in this document
   - Note the specific phrases establishing vendor responsibilities
   - Document the exact requirement language present

Output Required:
1. A document map showing only sections that exist in the provided text
2. Actual vendor-reference terminology found in this document
3. Real count of vendor obligations by section based on the text

Remember: Base your analysis ONLY on the provided document text. Do not include generic content or assumptions about what might typically be in an RFP.""",
        "message_template": """Please analyze ONLY the following RFP document text to create a blueprint for vendor requirements extraction. Do not use any assumed or generic content:

BEGIN DOCUMENT TEXT:
{pdf_data}
END DOCUMENT TEXT""",
    },
    {
        "name": "Explicit Requirements Extraction",
        "system_prompt": """STAGE 2: Explicit Vendor Requirements Extraction

Using the document map from Stage 1, systematically extract all EXPLICITLY stated vendor requirements.

Focus EXCLUSIVELY on statements that direct the VENDOR to:

Take specific actions
Provide specific information
Meet specific qualifications
Deliver specific items
Comply with specific standards
Follow specific processes


For each section in the document map:
a. Identify ONLY sentences containing directive language DIRECTLY aimed at vendors
b. Extract the complete sentence/paragraph containing the requirement
c. Record the exact page number and section reference
d. Identify the specific obligation verb (shall, must, will, should, may, etc.)
e. Determine if the requirement contains cross-references to other sections
Classify each requirement's obligation level based on language:

Mandatory: Uses "shall," "must," "will," "required," "necessary," etc.
Conditional: Contains "if," "when," "unless," specifying circumstances
Recommended: Uses "should," "encouraged," "preferred," "recommended," etc.
Optional: Uses "may," "can," "optional," etc.


Apply these strict filters:

Does this statement EXPLICITLY direct the vendor to do something or possess something?
Is the obligation placed directly on the vendor/proposer/bidder/firm/contractor?
Exclude any statements about what the City will do, how the City will evaluate, or general information
Exclude process descriptions unless they specifically create vendor obligations
Create a table with all explicit vendor requirements organized by section order in the RFP.""",
        "message_template": "Extract explicit vendor requirements from: {previous_output}",
    },
    {
        "name": "Implicit Requirements Extraction",
        "system_prompt": """Stage 3: Implicit Vendor Requirements Extraction 
Identify ONLY IMPLICIT vendor requirements that lack explicit directive language but still create clear vendor obligations.
Specifically examine:
Evaluation criteria that directly imply necessary vendor proposal content
Technical specifications stated as facts but that vendors must implement
Descriptions of deliverables that clearly establish vendor responsibilities
Qualification criteria phrased as facts rather than directives
Submission format descriptions that vendors must follow
For each identified implicit requirement: a. Extract the complete text containing the implied requirement b. Record exact page and section reference c. Explicitly state why this creates a vendor obligation d. Mark the obligation verb as "Implied" e. Classify the obligation level based on context and consequences f. Note any cross-references
Apply these strict verification tests:
Would a vendor be penalized in evaluation or disqualified if they failed to address this item?
Does the statement clearly establish something the vendor must do, provide, or demonstrate?
Is the obligation placed on the vendor rather than the City?
Exclude purely informational statements that don't create vendor obligations Add only verified implicit vendor requirements to the table, clearly marked as "Implied".""",
        "message_template": "Extract implicit vendor requirements from: {previous_output}",
    },
    {
        "name": "Cross-References Analysis",
        "system_prompt": """Stage 4: Vendor Requirement Cross-References and Dependencies 
Review all extracted vendor requirements to identify and resolve cross-references and dependencies.
For each vendor requirement that references other sections, forms, or attachments: a. Identify only the specific cross-references that create additional vendor obligations b. Locate the referenced material in the RFP c. Determine if the referenced material contains additional vendor requirements d. Note if the vendor requirement depends on information from multiple sections
Create dependency chains where vendor requirements build on each other: a. Identify vendor requirements that depend on others being fulfilled first b. Note vendor requirements that expand upon or modify other requirements c. Flag potentially conflicting vendor requirements
Update the Cross-References column in the vendor requirements table with: a. Specific section/page references that create vendor obligations b. Form or attachment names that vendors must complete or address c. Dependencies on other vendor requirements (using unique identifiers) d. Conflicts or potential inconsistencies in vendor obligations This stage ensures no vendor-specific requirements are missed due to fragmentation across the RFP.""",
        "message_template": "Analyze cross-references and dependencies in: {previous_output}",
    },
    {
        "name": "Critical Requirements Review",
        "system_prompt": """Stage 5: Critical Vendor Requirements Requiring Expert Review 
Thoroughly review each extracted vendor requirement to identify those needing human expert attention.
Flag vendor requirements with "Yes" for human review if they: a. Contain ambiguous language that could lead vendors to different interpretations b. Present potential conflicts with other vendor requirements c. Reference external standards that vendors must meet without providing specific details d. Include complex technical specifications requiring expert knowledge for vendor implementation e. Contain multiple sub-requirements that vendors might overlook f. Have unclear implications for vendor proposal preparation or contract performance g. Include undefined terms or acronyms that could confuse vendors h. Spread vendor obligations across multiple disconnected sections i. Contain technical calculations vendors must perform or comply with j. Involve subjective judgments about vendor performance or deliverables
For each flagged vendor requirement: a. Provide a specific rationale for why vendor experts should review this requirement b. Suggest what vendor expertise might be required (legal, technical, financial, etc.) c. Note specific risks if vendors misinterpret this requirement
Apply this test: Could a reasonable vendor misinterpret what they must do to comply? If yes, flag it. Update the Human Review Flag column with Yes/No values and concise rationales focused on vendor compliance risks.""",
        "message_template": "Identify critical requirements needing expert review in: {previous_output}",
    },
    {
        "name": "Requirements Verification",
        "system_prompt": """Stage 6: Vendor Requirements Verification 
Conduct a systematic verification of the vendor requirements table to ensure complete coverage of all vendor obligations.
Map all vendor requirements to the document structure: a. Verify every page has been analyzed for explicit and implicit vendor obligations b. Confirm all sections that contain vendor instructions are represented c. Check that all forms and attachments vendors must complete are accounted for
Cross-check against key RFP components that create vendor obligations: a. Ensure all evaluation criteria correspond to extracted vendor requirements b. Verify all minimum qualifications vendors must meet are captured c. Confirm all submission instructions vendors must follow are represented d. Check that all timeline/milestone requirements vendors must adhere to are included
Identify potential gaps in vendor obligations by looking for: a. Sections with unusually few vendor requirements that might indicate missed obligations b. References to vendor responsibilities without corresponding detailed requirements c. Evaluation criteria without clear vendor requirement counterparts d. Required vendor-completed forms without extraction of form-related requirements
Review for vendor obligation completeness using these tests: a. Could a vendor prepare a fully compliant proposal using just these requirements? b. Are all potential vendor disqualification factors captured? c. Are all scoring elements that impact vendors represented by corresponding requirements? Fill any identified gaps by returning to the source document and extracting missed vendor-specific requirements.""",
        "message_template": "Verify completeness of requirements in: {previous_output}",
    },
    {
        "name": "Final Requirements Table",
        "system_prompt": """Stage 7: Final Vendor Requirements Table 
Produce the final comprehensive table of all vendor-specific requirements.
Consolidate all vendor requirements from stages 2-6 into a single table with these exact columns: a. Page: Page number where the vendor requirement appears b. Section: Exact section reference as in the RFP c. Requirement Text: Verbatim quote of the vendor obligation d. Obligation Verb: Specific term creating the vendor obligation (shall, must, implied, etc.) e. Obligation Level: Mandatory, Conditional, Recommended, or Optional for the vendor f. Cross-References: References to other sections, forms, or attachments the vendor must address g. Human Review Flag: Yes/No with specific vendor-focused rationale if Yes
Sort the table by: a. Primary: Section order (following the RFP's structure) b. Secondary: Page number
Perform a final quality check focused on vendor obligations: a. Ensure all vendor requirement text is quoted verbatim from the RFP b. Verify page and section references for each vendor requirement c. Confirm appropriate classification of vendor obligation levels d. Check that human review flags include specific vendor compliance rationales
Add a summary section showing: a. Total number of vendor requirements extracted b. Distribution of vendor requirements by obligation level c. Number of vendor requirements flagged for human expert review d. Sections with highest concentration of vendor requirements This final consolidated table represents a comprehensive extraction of all vendor-specific requirements from the RFP.""",
        "message_template": "Create final requirements table from: {previous_output}",
    },
]
