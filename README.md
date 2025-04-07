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

## Prerequisites

- Python 3.8 or higher
- An Anthropic API key (for Claude)
- A Reducto API key (for PDF processing)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rfp-analysis.git
cd rfp-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```bash
touch .env
```

5. Add your API keys to the `.env` file:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
REDUCTO_API_KEY=your_reducto_api_key_here
```

## Project Structure

```
rfp-analysis/
├── file_storage/     # Place RFP PDFs here
├── main.py          # Main application code
├── prompts.py       # Analysis stage prompts
├── requirements.txt # Project dependencies
└── .env            # API keys (not in git)
```

## Usage

1. Place your RFP document (PDF) in the `file_storage` directory

2. Run the analysis:
```bash
python main.py
```

3. Check the output files:
- `RFP_requirements.txt`: Final structured requirements table
- `debug_stage_outputs.txt`: Detailed analysis logs
- `rfp_workflow.png`: Visual representation of the analysis workflow

## Common Issues

1. **API Key Errors**: Make sure both API keys are correctly set in your `.env` file
2. **PDF Processing Errors**: Ensure your PDF is text-based and not scanned
3. **Virtual Environment**: Always activate the virtual environment before running

## License

[Add your license information here] 