import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import time
import os
import sys

# Add the project root to the Python path to allow importing 'main'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import 'app' from 'main'
# Import memory_task_store specifically, and other needed items
from main import app, memory_task_store, ResearchQueryRequest, API_KEY, API_KEY_NAME

client = TestClient(app)

# Use the actual API_KEY from main.py for successful requests
VALID_API_KEY = API_KEY 
INVALID_API_KEY = "invalid_test_key"

@pytest.fixture(autouse=True)
def ensure_in_memory_task_store(monkeypatch):
    """Ensures that for testing, redis_client is None, so memory_task_store is used, and clears it."""
    monkeypatch.setattr("main.redis_client", None) # Force usage of memory_task_store
    memory_task_store.clear()
    yield
    memory_task_store.clear() # Clear again after test

def test_health_check():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_invoke_agent_task_unauthorized():
    response = client.post("/agent/invoke", 
                           headers={API_KEY_NAME: INVALID_API_KEY},
                           json={"query": "Test query"})
    assert response.status_code == 403 # Expecting Forbidden due to invalid API key

def test_get_task_status_unauthorized():
    response = client.get("/agent/tasks/some-task-id/status", headers={API_KEY_NAME: INVALID_API_KEY})
    assert response.status_code == 403


@patch('main.research_agent_instance', new_callable=AsyncMock) # Mock the global agent instance in main.py
def test_invoke_agent_and_get_status_flow(mock_agent_instance):
    # Configure the mock agent's run method
    mock_agent_response = {"summary": "This is a mock research summary."}
    mock_agent_instance.run = AsyncMock(return_value=mock_agent_response)

    # 1. Invoke the agent
    query = "What is the capital of France?"
    response_invoke = client.post(
        "/agent/invoke",
        headers={API_KEY_NAME: VALID_API_KEY},
        json={"query": query}
    )
    assert response_invoke.status_code == 200
    task_creation_data = response_invoke.json()
    task_id = task_creation_data.get("task_id")
    assert task_id is not None
    assert task_creation_data.get("status") == "pending"

    # 2. Check initial status (might be pending or already in_progress if background task is quick)
    response_status_initial = client.get(f"/agent/tasks/{task_id}/status", headers={API_KEY_NAME: VALID_API_KEY})
    assert response_status_initial.status_code == 200
    status_data_initial = response_status_initial.json()
    assert status_data_initial["task_id"] == task_id
    
    # The background task should have run.
    mock_agent_instance.run.assert_called_once_with(query)

    # 3. Check final status (should be completed because the mock is fast)
    response_status_final = client.get(f"/agent/tasks/{task_id}/status", headers={API_KEY_NAME: VALID_API_KEY})
    assert response_status_final.status_code == 200
    status_data_final = response_status_final.json()
    
    assert status_data_final["task_id"] == task_id
    assert status_data_final["task_status"]["status"] == "completed"
    assert status_data_final["task_status"]["result"] == mock_agent_response
    assert status_data_final["task_status"]["error"] is None

def test_get_status_for_nonexistent_task():
    response = client.get("/agent/tasks/nonexistent-task-id-123/status", headers={API_KEY_NAME: VALID_API_KEY})
    assert response.status_code == 404
    assert response.json() == {"detail": "Task ID nonexistent-task-id-123 not found"}

@patch('main.research_agent_instance', new_callable=AsyncMock)
@patch('main.run_research_task', new_callable=AsyncMock) # Also mock the task runner itself
def test_invoke_agent_initialization_failure(mock_run_research, mock_agent_instance_main):
    # Simulate agent not being initialized
    with patch('main.research_agent_instance', None):
        response_invoke = client.post(
            "/agent/invoke",
            headers={API_KEY_NAME: VALID_API_KEY},
            json={"query": "Test query when agent is None"}
        )
        assert response_invoke.status_code == 503
        assert response_invoke.json() == {"detail": "ResearchAgent not initialized. Check server logs."}

    # Ensure the background task was NOT called because of the early exit
    mock_run_research.assert_not_called()
    

@patch('main.research_agent_instance', new_callable=AsyncMock)
def test_background_task_handles_agent_exception(mock_agent_instance):
    # Configure the mock agent's run method to raise an exception
    error_message = "LLM API is down"
    mock_agent_instance.run = AsyncMock(side_effect=Exception(error_message))

    query = "Query that will cause agent error"
    response_invoke = client.post(
        "/agent/invoke",
        headers={API_KEY_NAME: VALID_API_KEY},
        json={"query": query}
    )
    assert response_invoke.status_code == 200
    task_id = response_invoke.json()["task_id"]

    mock_agent_instance.run.assert_called_once_with(query)

    response_status = client.get(f"/agent/tasks/{task_id}/status", headers={API_KEY_NAME: VALID_API_KEY})
    assert response_status.status_code == 200
    status_data = response_status.json()
    assert status_data["task_status"]["status"] == "failed"
    assert status_data["task_status"]["result"] is None
    assert error_message in status_data["task_status"]["error"] 