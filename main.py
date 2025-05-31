from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional, Union
import uvicorn
import os
import redis
import json # For SSE
import asyncio # For SSE
import logging
import sys # For logging to stdout
from prometheus_fastapi_instrumentator import Instrumentator

# Import necessary Manus agent components from openmanus_core
from openmanus_core.app.agent.manus import ManusAgent # Example
from openmanus_core.app.config import load_config # Example

# Import the agent service
from pharma_agent.agent_service import process_research_request

# --- Logger Setup ---
# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set default logging level

# Remove existing handlers to avoid duplicate logs if this module is reloaded
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create a handler for stdout
stream_handler = logging.StreamHandler(sys.stdout)

# Basic JSON-like formatter (can be replaced with a proper JSON library later if needed)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno
        }
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

formatter = JsonFormatter()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# --- End Logger Setup ---

app = FastAPI(
    title="PharmaDB Research Agent",
    description="An AI agent for deep research in PharmaDB using OpenManus.",
    version="0.1.0"
)

# --- Prometheus Metrics Setup ---
Instrumentator().instrument(app).expose(app, include_in_schema=True, endpoint="/metrics")
# --- End Prometheus Metrics Setup ---

# --- API Key Authentication --- 
API_KEY = os.getenv("RESEARCH_AGENT_API_KEY", "your_default_secret_api_key") # TODO: Change default in prod
API_KEY_NAME = "X-API-Key"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header_auth)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials",
        )
# --- End API Key Authentication ---

# --- Pydantic Models for /ask endpoint ---
class FileReference(BaseModel):
    url: HttpUrl
    # Potentially add other metadata like file_type, size, etc.

class AskRequest(BaseModel):
    conversation_id: Optional[str] = None
    question: str
    file_refs: Optional[List[FileReference]] = []
    history: Optional[List[Dict[str, Any]]] = [] # Each dict could be a previous Q/A turn

class Citation(BaseModel):
    source: str # e.g., filename, URL
    text: str   # Relevant snippet

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    trace: Dict[str, Any]
    conversation_id: Optional[str] = None

# Global variables to hold the initialized components
config = None
manus_agent = None
redis_client = None

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred.", "error_type": type(exc).__name__},
    )
# --- End Global Exception Handler ---

@app.on_event("startup")
async def startup_event():
    """Handles application startup tasks."""
    global manus_agent, config, redis_client

    logger.info("Starting application setup...")

    # Load configuration (e.g., from openmanus_core/config/config.toml)
    config_path = os.getenv("CONFIG_PATH", os.path.join("openmanus_core", "config", "config.toml"))
    try:
        config = load_config(config_path)
        logger.info(f"Configuration loaded successfully from {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Trying config.example.toml...")
        config_path = os.path.join("openmanus_core", "config", "config.example.toml")
        try:
            config = load_config(config_path)
            logger.info(f"Successfully loaded example configuration from {config_path}")
        except Exception as e:
            logger.error(f"Could not load any configuration: {e}", exc_info=True)
            # Potentially raise an error here or exit if config is critical
            config = None # Ensure config is None if loading fails
    except Exception as e:
        logger.error(f"Could not load configuration from {config_path}: {e}", exc_info=True)
        config = None

    # Initialize Manus Agent
    if config:
        try:
            # Ensure all necessary LLM API keys and other configs are available in the environment or config file
            manus_agent = ManusAgent(config)
            logger.info("ManusAgent initialized successfully.")
        except Exception as e:
            logger.error(f"Could not initialize ManusAgent: {e}", exc_info=True)
            manus_agent = None # Ensure agent is None if init fails
    else:
        logger.info("ManusAgent initialization skipped due to missing configuration.")
        manus_agent = None

    # Initialize Redis client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping() # Check connection
        logger.info(f"Connected to Redis at {redis_url}")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Could not connect to Redis at {redis_url}: {e}", exc_info=True)
        redis_client = None # Ensure it's None if connection failed
    except Exception as e: # Catch any other potential errors from redis.from_url
        logger.error(f"An unexpected error occurred during Redis initialization: {e}", exc_info=True)
        redis_client = None
    
    logger.info("Application setup finished.")

@app.get("/healthz", summary="Health Check")
async def health_check():
    """Provides a basic health check endpoint."""
    # TODO: Add more sophisticated health checks (e.g., check LLM connectivity, Redis)
    return {"status": "ok"}

# Define /ask endpoint (T11)
@app.post("/ask", response_model=AskResponse, summary="Ask the Research Agent")
async def ask_agent(request: AskRequest, api_key: str = Depends(get_api_key)):
    """Receives a research question, file references, and conversation history, then returns an answer."""
    logger.info(f"Received /ask request with question: {request.question[:50]}...")
    if not manus_agent:
        logger.error("ManusAgent not initialized. Returning 503.")
        raise HTTPException(status_code=503, detail="ManusAgent not initialized. Check server logs.")
    
    # For now, directly call the placeholder. Later, this will involve SSE.
    response_data = await process_research_request(
        question=request.question,
        file_refs=request.file_refs,
        history=request.history,
        agent=manus_agent,
        redis_client=redis_client
    )
    response_data["conversation_id"] = request.conversation_id
    return AskResponse(**response_data)

# Placeholder for SSE streaming version of /ask
# @app.post("/ask-stream", dependencies=[Depends(get_api_key)])
# async def ask_agent_stream(request: AskRequest):
#     if not manus_agent:
#         raise HTTPException(status_code=503, detail="ManusAgent not initialized. Check server logs.")

#     async def event_generator():
#         # Mock streaming data for now
#         yield f"data: {json.dumps({'type': 'thought', 'content': 'Starting research...'})}\n\n"
#         await asyncio.sleep(1)
#         yield f"data: {json.dumps({'type': 'action', 'tool': 'some_tool', 'input': 'some_input'})}\n\n"
#         await asyncio.sleep(1)
#         # ... call actual agent logic that yields events ...
#         final_result = await process_research_request(
#             question=request.question, file_refs=request.file_refs, history=request.history,
#             agent=manus_agent, redis_client=redis_client
#         )
#         final_result["conversation_id"] = request.conversation_id
#         yield f"data: {json.dumps({'type': 'result', 'data': final_result})}\n\n"

#     return StreamingResponse(event_generator(), media_type="text/event-stream")

# TODO: Define /metrics endpoint (T12) if needed

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 