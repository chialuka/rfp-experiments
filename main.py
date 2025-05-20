from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from rq import Queue
from worker import conn
from api import create_router
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Queue
queue = Queue(connection=conn)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI application."""
    logger.info("Starting RFP Analysis API")
    yield
    logger.info("Shutting down RFP Analysis API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RFP Analysis API",
        description="API for analyzing RFP documents using LangGraph and Claude",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create file_storage directory if it doesn't exist
    STORAGE_DIR = Path("file_storage")
    STORAGE_DIR.mkdir(exist_ok=True)
    
    # Include router with dependency injection
    app.include_router(create_router(queue))
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=2500, reload=True)
