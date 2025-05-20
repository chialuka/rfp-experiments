# RFP Analysis System

A Python-based system that automatically analyzes Request for Proposal (RFP) documents to extract and structure vendor requirements. The system uses AI to process RFP documents and generate a comprehensive table of requirements, making it easier for vendors to understand their obligations and evaluate feasibility based on past proposals.

## Features

- **Automated RFP document analysis**
- **Compliance matrix generation**: Extraction of explicit and implicit vendor requirements
- **Feasibility analysis**: Semantic search of past RFPs to assess requirement feasibility
- **Background job processing**: Asynchronous processing via Redis queues
- **REST API**: FastAPI endpoints for integration with other systems
- **Vector search**: Semantic search using Pinecone for requirement comparison
- **Database integration**: Result storage in Supabase

## Architecture

The system combines several components:
- **FastAPI** web server for API endpoints
- **RQ (Redis Queue)** for background job processing
- **LangChain** for orchestrating LLM workflows
- **Pinecone** for vector storage and semantic search
- **Supabase** for structured data storage
- **Claude and OpenAI** models for text analysis

## API Endpoints

- `POST /rfp/analyze`: Analyze RFP document to generate compliance matrix and feasibility check
- `POST /rfp/compliance-matrix`: Generate only the compliance matrix
- `POST /rfp/feasibility`: Check feasibility of requirements against past RFPs
- `POST /rfp/vectorize`: Index RFP documents for future feasibility comparisons
- `DELETE /rfp/vector/reset`: Reset the vector store
- `GET /job-status/{job_id}`: Check status of background jobs
- `GET /health`: Health check endpoint

## Prerequisites

- Python 3.11 or higher (3.13 recommended for local development)
- Redis server (local for development, Redis add-on for Heroku)
- Anthropic API key (for Claude)
- OpenAI API key (for embeddings)
- Reducto API key (for PDF processing)
- Pinecone API key and index (for vector storage)
- Supabase project and credentials

## Local Setup

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
OPENAI_API_KEY=your_openai_api_key_here
REDUCTO_API_KEY=your_reducto_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENV=us-east-1
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
REDIS_URL=redis://localhost:6379/0
PYTHON_ENV=development
```

6. Start Redis (required for job queueing):
```bash
# Using Homebrew on macOS
brew services start redis

# Or run Redis directly
redis-server
```

7. Start the API server:
```bash
uvicorn api:app --reload
```

8. Start the worker process (in a separate terminal):
```bash
python worker.py
```

## Running on Heroku

1. Create a Heroku app:
```bash
heroku create your-app-name
```

2. Add Redis add-on:
```bash
heroku addons:create heroku-redis:hobby-dev
```

3. Configure environment variables:
```bash
heroku config:set ANTHROPIC_API_KEY=your_key
heroku config:set OPENAI_API_KEY=your_key
heroku config:set REDUCTO_API_KEY=your_key
heroku config:set PINECONE_API_KEY=your_key
heroku config:set PINECONE_ENV=us-east-1
heroku config:set SUPABASE_URL=your_url
heroku config:set SUPABASE_KEY=your_key
heroku config:set PYTHON_ENV=production
```

4. Deploy the application:
```bash
git push heroku main
```

5. Scale worker dyno:
```bash
heroku ps:scale worker=1
```

## Project Structure

```
rfp-analysis/
├── api/              # API routes and request handlers 
│   ├── __init__.py   # API initialization
│   └── routes.py     # API endpoints definition
├── executors/        # Workflow execution handlers
│   ├── run_compliance.py    # Compliance matrix workflow
│   └── run_feasibility.py   # Feasibility analysis workflow  
├── graphs/           # LangGraph workflow definitions
│   ├── compliance.py # Compliance matrix graph nodes
│   └── feasibility.py # Feasibility analysis graph nodes
├── models/           # Data models
│   └── state.py      # State models for workflows
├── prompts/          # LLM prompts
│   ├── compliance_matrix.py # Compliance analysis prompts
│   └── feasibility.py       # Feasibility analysis prompts  
├── services/         # Service integrations
│   ├── llm_service.py      # LLM service
│   └── vector_service.py   # Vector store service
├── main.py           # Application entry point
├── worker.py         # Background worker configuration
├── db.py             # Database connection
├── file_storage/     # Place RFP PDFs here
├── requirements.txt  # Project dependencies
├── Procfile          # Heroku process configuration
└── .env              # API keys (not in git)
```

## Dependency Management

If you encounter dependency conflicts, you may need to manage versions carefully:
- langchain-pinecone 0.2.6 requires aiohttp<3.11
- supabase 2.15.0+ (through realtime) requires aiohttp>=3.11

When deploying, you may need to use `--no-deps` or pin specific versions.

## Troubleshooting

### Local Development
- **Redis Connection Error**: Ensure Redis is running (`redis-cli ping` should return PONG)
- **Pinecone Authentication Errors**: Verify your API key is correct and has access to the z-bids index
- **macOS Fork Safety Error**: Use `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` before starting the worker or switch to SpawnWorker
- **Vector Dimension Mismatch**: Ensure your embedding model and index dimensions match (1536 or 3072)

### Heroku
- **Worker Not Processing Jobs**: Check `heroku logs --tail --dyno worker` for errors
- **Dependency Conflicts**: Review `heroku logs --tail` during build to identify package conflicts
- **Resource Limitations**: Monitor dyno resource usage and upgrade if needed

## License

[Add your license information here] 
