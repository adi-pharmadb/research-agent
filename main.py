from fastapi import FastAPI, HTTPException, Request, Security, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Any, Optional, Union
import uvicorn
import os
import redis
import json # For SSE
import asyncio # For SSE
import logging
import sys # For logging to stdout
from prometheus_fastapi_instrumentator import Instrumentator
import uuid # For generating task IDs
from datetime import datetime, timezone # Added timezone
from contextlib import asynccontextmanager # For lifespan

# Import necessary Manus agent components from openmanus_core
from openmanus_core.app.agent.manus import Manus as ResearchAgent # Renamed and using Manus directly
from openmanus_core.app.config import config # Correct way to access config

# Import the agent service (we'll call this within a background task)
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

# Global variables to hold the initialized components
research_agent_instance = None
redis_client: Optional[redis.Redis] = None # Typed redis_client

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global research_agent_instance, redis_client
    logger.info("Starting application setup (lifespan)...")

    try:
        research_agent_instance = await ResearchAgent.create()
        logger.info("ResearchAgent initialized successfully (lifespan).")
    except Exception as e:
        logger.error(f"Could not initialize ResearchAgent (lifespan): {e}", exc_info=True)
        research_agent_instance = None

    redis_url = os.getenv("REDIS_URL") # Get REDIS_URL, will be None if not set
    if redis_url:
        logger.info(f"REDIS_URL found, attempting connection: {redis_url}")
        try:
            redis_client = redis.from_url(redis_url, decode_responses=False) # decode_responses=False for storing JSON bytes
            redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url} (lifespan)")
        except redis.exceptions.ConnectionError as e:
            logger.warning(f"Could not connect to Redis at {redis_url} (lifespan): {e}. Proceeding without Redis (tasks will be in-memory and non-persistent).")
            redis_client = None
        except Exception as e:
            logger.warning(f"An unexpected error occurred during Redis initialization (lifespan): {e}. Proceeding without Redis.")
            redis_client = None
    else:
        logger.info("REDIS_URL not set. Proceeding without Redis (tasks will be in-memory and non-persistent).")
        redis_client = None
    
    logger.info("Application setup finished (lifespan).")
    
    yield # Application runs here
    
    # Shutdown logic (if any)
    logger.info("Application shutting down (lifespan)...")
    if hasattr(research_agent_instance, 'cleanup') and asyncio.iscoroutinefunction(research_agent_instance.cleanup):
        try:
            await research_agent_instance.cleanup()
            logger.info("ResearchAgent cleaned up successfully (lifespan).")
        except Exception as e:
            logger.error(f"Error during ResearchAgent cleanup (lifespan): {e}", exc_info=True)
    if redis_client:
        try:
            redis_client.close()
            logger.info("Redis client closed (lifespan).")
        except Exception as e:
            logger.error(f"Error closing Redis client (lifespan): {e}", exc_info=True)
    logger.info("Application shutdown complete (lifespan).")

app = FastAPI(
    title="PharmaDB Research Agent",
    description="An AI agent for deep research in PharmaDB using OpenManus.",
    version="0.1.0",
    lifespan=lifespan # Register lifespan handler
)

# --- Prometheus Metrics Setup ---
Instrumentator().instrument(app).expose(app, include_in_schema=True, endpoint="/metrics")
# --- End Prometheus Metrics Setup ---

# --- API Key Authentication --- 
API_KEY = os.getenv("RESEARCH_AGENT_API_KEY", "your_default_secret_api_key")
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

# --- Pydantic Models for Agent Interaction ---
class ResearchQueryRequest(BaseModel):
    query: str = Field(..., description="The research question or task for the agent.")
    # file_refs: Optional[List[FileReference]] = Field(default_factory=list, description="Optional list of file references for context.")
    # history: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Optional conversation history.")
    # conversation_id: Optional[str] = None # We'll use task_id for tracking

class TaskCreationResponse(BaseModel):
    task_id: str
    status: str = "pending"
    detail: str = "Task accepted and queued for processing."

class TaskStatus(BaseModel):
    status: str
    timestamp: str
    result: Optional[Any] = None
    error: Optional[str] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    task_status: TaskStatus

# Remove old models if they are fully replaced or adapt them
# class FileReference(BaseModel):
#     url: HttpUrl
#     # Potentially add other metadata like file_type, size, etc.

# class Citation(BaseModel):
#     source: str # e.g., filename, URL
#     text: str   # Relevant snippet

# class AskResponse(BaseModel): # This will be effectively replaced by the result in TaskStatus
#     answer: str
#     citations: List[Citation]
#     trace: Dict[str, Any]
#     conversation_id: Optional[str] = None


# In-memory store for task status and results
# For production, use Redis or a database
memory_task_store: Dict[str, TaskStatus] = {}
REDIS_TASK_PREFIX = "task:"
TASK_EXPIRY_SECONDS = 3600 * 24 # 1 day


# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred.", "error_type": type(exc).__name__},
    )
# --- End Global Exception Handler ---

# @app.on_event("startup") # This is now handled by lifespan
# async def startup_event():
#     """Handles application startup tasks."""
#     pass # Logic moved to lifespan

@app.get("/healthz", summary="Health Check")
async def health_check():
    """Provides a basic health check endpoint."""
    # TODO: Add more sophisticated health checks (e.g., check LLM connectivity, Redis)
    return {"status": "ok"}

# --- Background Task for Agent Processing ---
async def run_research_task(task_id: str, query: str):
    """Runs the research agent task in the background and updates the task_store."""
    logger.info(f"Task {task_id}: Starting research for query: {query[:50]}...")
    current_status = TaskStatus(
        status="in_progress", 
        timestamp=datetime.now(timezone.utc).isoformat(), # Changed to timezone-aware UTC
        result=None,
        error=None
    )

    if redis_client:
        redis_client.setex(f"{REDIS_TASK_PREFIX}{task_id}", TASK_EXPIRY_SECONDS, current_status.model_dump_json())
    else:
        memory_task_store[task_id] = current_status

    if not research_agent_instance:
        error_msg = "ResearchAgent not initialized. Cannot process task."
        logger.error(f"Task {task_id}: {error_msg}")
        current_status = TaskStatus(
            status="failed", 
            timestamp=datetime.now(timezone.utc).isoformat(), # Changed to timezone-aware UTC
            result=None, 
            error=error_msg
        )
        if redis_client:
            redis_client.setex(f"{REDIS_TASK_PREFIX}{task_id}", TASK_EXPIRY_SECONDS, current_status.model_dump_json())
        else:
            memory_task_store[task_id] = current_status
        return

    try:
        # Here, we directly use the agent's run method.
        # This assumes agent.run() is an async method and takes the query as input.
        # The output of agent.run() will be stored as the result.
        # You might need to adapt this based on the actual signature and 
        # return type of your ResearchAgent's primary execution method.
        # For Manus, it's usually an `async_run` or `run` method that takes a list of messages or a string.
        
        # Example: agent_result = await research_agent_instance.async_run(prompt=query)
        # For Manus, the `run` method might expect a list of messages
        # For simplicity, let's assume a simplified interaction or adapt if needed
        # based on how `process_research_request` was using the agent.
        
        # If process_research_request was a simple wrapper around agent.run, we can replicate its core logic.
        # Let's assume research_agent_instance.run() is the method to call and it's synchronous.
        # To make it async compatible for background task, we can run it in a thread pool if it's blocking.
        # However, Manus agent's run method is often async.
        
        # Assuming agent expects a list of messages, like: `[{"role": "user", "content": query}]`
        # The actual structure might differ for OpenManus.
        # For the demo, Manus takes a string prompt.
        agent_response = await research_agent_instance.run(query) 

        logger.info(f"Task {task_id}: Research completed. Result: {str(agent_response)[:100]}...")
        current_status = TaskStatus(
            status="completed", 
            timestamp=datetime.now(timezone.utc).isoformat(), # Changed to timezone-aware UTC
            result=agent_response, # Store the raw agent response
            error=None
        )
    except Exception as e:
        error_msg = f"Error during research task: {e}"
        logger.error(f"Task {task_id}: {error_msg}", exc_info=True)
        current_status = TaskStatus(
            status="failed", 
            timestamp=datetime.now(timezone.utc).isoformat(), # Changed to timezone-aware UTC
            result=None, 
            error=error_msg
        )
    
    if redis_client:
        redis_client.setex(f"{REDIS_TASK_PREFIX}{task_id}", TASK_EXPIRY_SECONDS, current_status.model_dump_json())
    else:
        memory_task_store[task_id] = current_status
# --- End Background Task ---

# New Agent Invocation Endpoint
@app.post("/agent/invoke", response_model=TaskCreationResponse, summary="Invoke the Research Agent Task")
async def invoke_agent_task(
    request: ResearchQueryRequest, 
    background_tasks: BackgroundTasks, 
    api_key: str = Depends(get_api_key)
):
    """Accepts a research query, starts a background task for the agent, and returns a task ID."""
    logger.info(f"Received /agent/invoke request with query: {request.query[:50]}...")
    
    if not research_agent_instance:
        logger.error("ResearchAgent not initialized. Returning 503.")
        raise HTTPException(status_code=503, detail="ResearchAgent not initialized. Check server logs.")

    task_id = str(uuid.uuid4())
    initial_status = TaskStatus(
        status="pending", 
        timestamp=datetime.now(timezone.utc).isoformat(), # Changed to timezone-aware UTC
        result=None,
        error=None
    )
    
    if redis_client:
        redis_client.setex(f"{REDIS_TASK_PREFIX}{task_id}", TASK_EXPIRY_SECONDS, initial_status.model_dump_json())
    else:
        memory_task_store[task_id] = initial_status
    
    background_tasks.add_task(run_research_task, task_id, request.query)
    
    logger.info(f"Task {task_id} created and queued for query: {request.query[:50]}...")
    return TaskCreationResponse(task_id=task_id)

# New Task Status Endpoint
@app.get("/agent/tasks/{task_id}/status", response_model=TaskStatusResponse, summary="Get Research Agent Task Status")
async def get_agent_task_status(task_id: str, api_key: str = Depends(get_api_key)):
    """Retrieves the status and result (if available) of a previously submitted agent task."""
    logger.info(f"Received status request for task ID: {task_id}")
    task_info_json = None
    if redis_client:
        task_info_json = redis_client.get(f"{REDIS_TASK_PREFIX}{task_id}")
        if task_info_json:
            task_info = TaskStatus.model_validate_json(task_info_json)
        else:
            task_info = None
    else:
        task_info = memory_task_store.get(task_id)

    if not task_info:
        logger.warning(f"Task ID {task_id} not found.")
        raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found")
    
    return TaskStatusResponse(task_id=task_id, task_status=task_info)


# Remove the old /ask endpoint as it's replaced by /agent/invoke and /agent/tasks/... 
# @app.post("/ask", response_model=AskResponse, summary="Ask the Research Agent")
# async def ask_agent(request: AskRequest, api_key: str = Depends(get_api_key)):
#     logger.info(f"Received /ask request with question: {request.question[:50]}...")
#     if not research_agent_instance: # Check renamed instance
#         logger.error("ResearchAgent not initialized. Returning 503.")
#         raise HTTPException(status_code=503, detail="ResearchAgent not initialized. Check server logs.")
#     
#     # For now, directly call the placeholder. Later, this will involve SSE.
#     response_data = await process_research_request(
#         question=request.question,
#         # file_refs=request.file_refs, # Assuming these might not be used directly by ResearchAgent or handled by process_research_request
#         # history=request.history,
#         agent=research_agent_instance, # Pass renamed instance
#         redis_client=redis_client
#     )
#     # The structure of response_data from process_research_request needs to align with AskResponse
#     # This endpoint will be replaced by the new /agent/invoke and /agent/tasks/... structure
#     
#     # Temporary adaptation to old AskResponse if process_research_request returns a simple string or dict
#     if isinstance(response_data, str):
#         ask_response_data = {"answer": response_data, "citations": [], "trace": {}}
#     elif isinstance(response_data, dict) and "answer" in response_data:
#         ask_response_data = response_data
#     else: # Fallback if the structure is unexpected
#         ask_response_data = {"answer": "Processed, but response format is unexpected.", "citations": [], "trace": {}}
# 
#     ask_response_data["conversation_id"] = request.conversation_id
#     return AskResponse(**ask_response_data)

# Placeholder for SSE streaming version of /ask (can be adapted for tasks later if needed)
# @app.post("/ask-stream", dependencies=[Depends(get_api_key)])
# async def ask_agent_stream(request: AskRequest):
#     if not research_agent_instance: # Check renamed instance
#         raise HTTPException(status_code=503, detail="ResearchAgent not initialized. Check server logs.")


# TODO: Define /metrics endpoint (T12) if needed -- This is already present from previous steps
# Ensure uvicorn command is present for running the app, if it was removed.
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000")) # Default to 10000 if PORT not set
    uvicorn.run(app, host="0.0.0.0", port=port) 