# RFP Analysis System

A Python-based system that automatically analyzes Request for Proposal (RFP) documents to extract and structure vendor requirements. The system uses AI to process RFP documents and generate a comprehensive table of requirements, making it easier for vendors to understand their obligations.

## Features

- Automated RFP document analysis
- Extraction of explicit and implicit vendor requirements
- Generation of structured requirements table
- Identification of critical requirements needing expert review
- Cross-reference analysis between requirements
- Debug logging of analysis stages

## Technologies Used

- **Python 3.x**: Core programming language
- **LangChain**: Framework for LLM application development
- **Claude-3-Sonnet**: Large Language Model for text analysis
- **LangGraph**: For building the analysis workflow
- **Reducto**: PDF parsing and text extraction
- **python-dotenv**: Environment variable management

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env to add your API keys:
# - REDUCTO_API_KEY
# - ANTHROPIC_API_KEY
```

## Usage

1. Place your RFP document (PDF) in the `file_storage` directory

2. Run the analysis:
```bash
python main.py
```

3. Check the output files:
- `requirements.txt`: Final structured requirements table
- `debug_stage_outputs.txt`: Detailed analysis logs
- `rfp_workflow.png`: Visual representation of the analysis workflow

## Output Format

The system generates a structured table containing:
```
Page | Section | Requirement Text | Obligation Verb | Obligation Level | Cross-References | Human Review Flag
-----|----------|-----------------|-----------------|------------------|------------------|------------------
1-2  | RFP Timeline | "Example requirement..." | Must | Mandatory | None | No
```

## License

[Add your license information here] 