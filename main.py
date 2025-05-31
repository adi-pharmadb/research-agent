from fastapi import FastAPI
import uvicorn
import os
import redis

# Import necessary Manus agent components from openmanus_core
from openmanus_core.app.agent.manus import ManusAgent # Example
from openmanus_core.app.config import load_config # Example

app = FastAPI(
    title="PharmaDB Research Agent",
    description="An AI agent for deep research in PharmaDB using OpenManus.",
    version="0.1.0"
)

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

# TODO: Define /ask endpoint (T11)
# @app.post("/ask")
# async def ask_agent():
#     # Placeholder for T11
#     return {"message": "Ask endpoint not yet implemented"}

# TODO: Define /metrics endpoint (T12) if needed

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 