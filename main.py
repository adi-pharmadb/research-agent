from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional, Union
import uvicorn
import os
import redis
import json # For SSE
import asyncio # For SSE

# Import necessary Manus agent components from openmanus_core
from openmanus_core.app.agent.manus import ManusAgent # Example
from openmanus_core.app.config import load_config # Example

# Import the agent service
from pharma_agent.agent_service import process_research_request

app = FastAPI(
    title="PharmaDB Research Agent",
    description="An AI agent for deep research in PharmaDB using OpenManus.",
    version="0.1.0"
)

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

@app.on_event("startup")
async def startup_event():
    """Handles application startup tasks."""
    global manus_agent, config, redis_client

    # Load configuration (e.g., from openmanus_core/config/config.toml)
    config_path = os.getenv("CONFIG_PATH", os.path.join("openmanus_core", "config", "config.toml"))
    try:
        config = load_config(config_path)
        print(f"INFO: Configuration loaded successfully from {config_path}")
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}. Trying config.example.toml...")
        config_path = os.path.join("openmanus_core", "config", "config.example.toml")
        try:
            config = load_config(config_path)
            print(f"INFO: Successfully loaded example configuration from {config_path}")
        except Exception as e:
            print(f"ERROR: Could not load any configuration: {e}")
            # Potentially raise an error here or exit if config is critical
            config = None # Ensure config is None if loading fails
    except Exception as e:
        print(f"ERROR: Could not load configuration from {config_path}: {e}")
        config = None

    # Initialize Manus Agent
    if config:
        try:
            # Ensure all necessary LLM API keys and other configs are available in the environment or config file
            manus_agent = ManusAgent(config)
            print("INFO: ManusAgent initialized successfully.")
        except Exception as e:
            print(f"ERROR: Could not initialize ManusAgent: {e}")
            manus_agent = None # Ensure agent is None if init fails
    else:
        print("INFO: ManusAgent initialization skipped due to missing configuration.")
        manus_agent = None

    # Initialize Redis client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping() # Check connection
        print(f"INFO: Connected to Redis at {redis_url}")
    except redis.exceptions.ConnectionError as e:
        print(f"ERROR: Could not connect to Redis at {redis_url}: {e}")
        redis_client = None # Ensure it's None if connection failed
    except Exception as e: # Catch any other potential errors from redis.from_url
        print(f"ERROR: An unexpected error occurred during Redis initialization: {e}")
        redis_client = None

@app.get("/healthz", summary="Health Check")
async def health_check():
    """Provides a basic health check endpoint."""
    # TODO: Add more sophisticated health checks (e.g., check LLM connectivity, Redis)
    return {"status": "ok"}

# Define /ask endpoint (T11)
@app.post("/ask", response_model=AskResponse, summary="Ask the Research Agent")
async def ask_agent(request: AskRequest):
    """Receives a research question, file references, and conversation history, then returns an answer."""
    if not manus_agent:
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
# @app.post("/ask-stream")
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