# PharmaDB Research Agent Microservice API

This document provides an overview of the API endpoints for the PharmaDB Research Agent microservice. This agent uses OpenManus الايهto perform research tasks based on user queries.

## Base URL

*   **Local Development:** `http://localhost:8000` (or the port configured when running with Uvicorn)
*   **Deployment:** (To be filled in once deployed, e.g., `https://your-render-app-name.onrender.com`)

## Authentication

All API endpoints (except `/healthz` and potentially `/metrics`) require an API key to be passed in the `X-API-Key` header.

*   **Header Name:** `X-API-Key`
*   **Default Key (for local development):** `your_default_secret_api_key` 
    *   This value is set by the `RESEARCH_AGENT_API_KEY` environment variable in `main.py`. **It is crucial to change this default for any production or publicly accessible deployment.**

## Endpoints

### 1. Health Check

*   **Endpoint:** `GET /healthz`
*   **Description:** Provides a basic health check for the service.
*   **Authentication:** Not required.
*   **Request:** None
*   **Success Response (200 OK):**
    ```json
    {
        "status": "ok"
    }
    ```

### 2. Invoke Agent Research Task

*   **Endpoint:** `POST /agent/invoke`
*   **Description:** Accepts a research query, queues it for processing by the AI agent in the background, and returns a task ID for status tracking.
*   **Authentication:** Required (`X-API-Key`).
*   **Request Body (application/json):**
    ```json
    {
        "query": "Your detailed research question or task for the agent."
    }
    ```
    *   `query` (string, required): The research question or task.
*   **Success Response (200 OK):** `TaskCreationResponse`
    ```json
    {
        "task_id": "some-unique-uuid-string",
        "status": "pending",
        "detail": "Task accepted and queued for processing."
    }
    ```
*   **Error Responses:**
    *   `403 Forbidden`: Invalid or missing API key.
    *   `503 Service Unavailable`: If the ResearchAgent is not initialized (e.g., due to configuration issues on startup).

### 3. Get Agent Task Status

*   **Endpoint:** `GET /agent/tasks/{task_id}/status`
*   **Description:** Retrieves the current status and result (if completed or failed) of a previously submitted agent task.
*   **Authentication:** Required (`X-API-Key`).
*   **Path Parameters:**
    *   `task_id` (string, required): The unique ID of the task, obtained from the `/agent/invoke` endpoint.
*   **Success Response (200 OK):** `TaskStatusResponse`
    ```json
    {
        "task_id": "some-unique-uuid-string",
        "task_status": {
            "status": "completed", // or "pending", "in_progress", "failed"
            "timestamp": "YYYY-MM-DDTHH:MM:SS.ffffffZ", // ISO 8601 UTC timestamp
            "result": { /* Agent's response if status is "completed" */ }, 
            "error": null // or error message string if status is "failed"
        }
    }
    ```
    *   `task_status.status` can be:
        *   `pending`: Task is in the queue, not yet started.
        *   `in_progress`: Task is currently being processed by the agent.
        *   `completed`: Task finished successfully. The `result` field will contain the agent's findings.
        *   `failed`: Task failed during processing. The `error` field will contain an error message.
*   **Error Responses:**
    *   `403 Forbidden`: Invalid or missing API key.
    *   `404 Not Found`: If the specified `task_id` does not exist.

### 4. Prometheus Metrics

*   **Endpoint:** `GET /metrics`
*   **Description:** Exposes application metrics in Prometheus format.
*   **Authentication:** Not required by default (can be configured).
*   **Request:** None
*   **Success Response (200 OK):** Text-based Prometheus metrics.

## FastAPI Automatic Documentation

When the service is running, interactive API documentation (Swagger UI and ReDoc) is automatically available:

*   **Swagger UI:** `GET /docs`
*   **ReDoc:** `GET /redoc`
*   **OpenAPI Schema:** `GET /openapi.json`

These interfaces provide detailed information about request/response models, parameters, and allow for direct interaction with the API endpoints.

## Example Workflow

1.  **Submit a research task:**
    ```bash
    curl -X POST "http://localhost:8000/agent/invoke" \
         -H "X-API-Key: your_default_secret_api_key" \
         -H "Content-Type: application/json" \
         -d '{"query": "What are the latest treatments for Alzheimer's disease focusing on amyloid plaques?"}'
    ```
    This will return a JSON response with a `task_id`.

2.  **Check task status:**
    Replace `{your_task_id}` with the ID obtained from the previous step.
    ```bash
    curl -X GET "http://localhost:8000/agent/tasks/{your_task_id}/status" \
         -H "X-API-Key: your_default_secret_api_key"
    ```
    Poll this endpoint until the status is `completed` or `failed`.

3.  **Review results:**
    If the status is `completed`, the `result` field in the response will contain the research findings.

## Running the Service Locally

1.  Ensure you have Python 3.12+ and all dependencies installed (from `requirements.txt` and `openmanus_core/requirements.txt`).
2.  Make sure Playwright browser binaries are installed (`playwright install`).
3.  Set up your API keys and other necessary configurations in `openmanus_core/config/config.toml`.
    *   At a minimum, an LLM API key (e.g., for OpenAI or Anthropic) is required for the agent to function.
4.  Set the `RESEARCH_AGENT_API_KEY` environment variable if you want to use a different key than the default for accessing your microservice.
5.  Run the application using Uvicorn from the project root:
    ```bash
    source .venv/bin/activate  # If using a virtual environment
    uvicorn main:app --reload --port 8000
    ```
    The API will then be available at `http://localhost:8000`. 