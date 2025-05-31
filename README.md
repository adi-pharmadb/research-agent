# PharmaDB Research Agent

This project implements an AI-powered research agent microservice for PharmaDB, utilizing the OpenManus framework. It provides a FastAPI interface to receive research questions and relevant file URLs, and returns a citation-rich answer.

## Overview

The agent is designed to:
- Integrate with an existing PharmaDB web application.
- Leverage OpenManus for its core agent capabilities and tool usage (e.g., CSV querying, PDF search).
- Be deployed as a Dockerized service on Render.

## Features (based on `specs.txt`)

- FastAPI backend with `/ask` endpoint.
- Dockerized for deployment.
- Configuration management for OpenManus.
- Redis integration (placeholder for now, for caching/history).
- Structured logging.
- Prometheus metrics endpoint (`/metrics`).
- API Key authentication for `/ask`.
- Basic CI via GitHub Actions (linting, testing).

## Project Structure

```
pharma-research-agent/
├── .github/workflows/         # GitHub Actions CI
├── openmanus_core/            # The core OpenManus framework (forked)
├── pharma_agent/              # Business logic specific to this research agent
│   ├── __init__.py
│   └── agent_service.py
├── tests/                     # Unit and integration tests
│   ├── __init__.py
│   └── test_agent_service.py
├── .gitignore
├── main.py                    # FastAPI application entry point
├── Dockerfile                 # (Located in openmanus_core/Dockerfile, used by render.yaml)
├── render.yaml                # Render deployment configuration
├── requirements.txt           # Main Python dependencies
├── requirements-dev.txt       # Development/test dependencies
├── specs.txt                  # Project specifications and task list
└── README.md                  # This file
```

## Setup and Running Locally

1.  **Prerequisites:**
    *   Python 3.12+
    *   Docker Desktop (running)
    *   Access to a Redis instance (optional for basic startup, required for full functionality later)
    *   OpenAI API Key (and potentially other keys like Bing Search, set as environment variables or in `openmanus_core/config/config.toml`)

2.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd pharma-research-agent
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate 
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    # Install OpenManus core dependencies (if not handled by root requirements/Dockerfile in future)
    # pip install -r openmanus_core/requirements.txt 
    ```

5.  **Configuration:**
    *   Copy `openmanus_core/config/config.example.toml` to `openmanus_core/config/config.toml`.
    *   Update `openmanus_core/config/config.toml` with your API keys (e.g., `OPENAI_API_KEY`).
    *   Alternatively, set environment variables (e.g., `OPENAI_API_KEY`, `BING_KEY`, `REDIS_URL`).
    *   Set `RESEARCH_AGENT_API_KEY` environment variable for the `/ask` endpoint access.

6.  **Run the FastAPI application directly (for development):**
    ```bash
    # Ensure RESEARCH_AGENT_API_KEY is set in your environment
    # Ensure OPENAI_API_KEY (and others) are set or in config.toml
    # Ensure REDIS_URL is set (or defaults to localhost)
    python main.py
    ```
    The application will be available at `http://localhost:8000`.

7.  **Build and Run with Docker (simulates Render deployment):**
    ```bash
    # Build the image
    docker build -t pharma-research-agent -f openmanus_core/Dockerfile .

    # Run the container
    # Replace your_api_key_value and other ENV vars as needed
    docker run -d -p 8000:8000 \
        -e RESEARCH_AGENT_API_KEY="your_secret_api_key_for_ask_endpoint" \
        -e OPENAI_API_KEY="your_openai_key" \
        -e REDIS_URL="redis://your_redis_host:your_redis_port" \
        --name pharma-agent-container pharma-research-agent
    ```
    The application will be available at `http://localhost:8000`.

## API Endpoints

*   `GET /healthz`: Health check.
*   `GET /metrics`: Prometheus metrics.
*   `POST /ask`: 
    *   Submit a research question. 
    *   Requires `X-API-Key` header for authentication.
    *   Request body (JSON):
        ```json
        {
          "conversation_id": "optional-uuid",
          "question": "What are the latest treatments for X?",
          "file_refs": [
            { "url": "http://example.com/document1.pdf" }
          ],
          "history": []
        }
        ```
    *   Response body (JSON, placeholder for now):
        ```json
        {
          "answer": "This is a placeholder answer...",
          "citations": [],
          "trace": {},
          "conversation_id": "optional-uuid"
        }
        ```

## Testing

Run tests using pytest:
```bash
source .venv/bin/activate
pytest
```

## Deployment

This service is configured for deployment to Render using the `render.yaml` file. Pushes to the `main` branch (if auto-deploy is enabled in Render) will trigger a new build and deployment based on the `openmanus_core/Dockerfile`.

Environment variables (secrets) like `OPENAI_API_KEY`, `BING_KEY`, `REDIS_URL`, `RESEARCH_AGENT_API_KEY` should be configured in the Render dashboard.

## TODO / Next Steps (from specs.txt)

-   Flesh out `pharma_agent.agent_service.process_research_request` with actual OpenManus agent calls.
-   Implement Server-Sent Events (SSE) for the `/ask` endpoint for streaming thought processes.
-   More sophisticated health checks.
-   Expand unit and integration tests.
-   Further refine error handling and logging. 