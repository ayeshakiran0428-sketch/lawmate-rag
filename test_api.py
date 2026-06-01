import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api import app

client = TestClient(app)

@patch('api.process_query')
def test_chat_endpoint_success(mock_process_query):
    # Mock the return value of process_query
    mock_process_query.return_value = {
        "query": "test query",
        "answer": "1. Case Type: Test\n2. Section: 123",
        "context_found": True,
        "context_used": "Test context"
    }

    response = client.post(
        "/api/chat",
        json={"query": "test query"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert "Test" in data["answer"]
    assert data["context_found"] is True

@patch('api.process_query')
def test_chat_endpoint_empty_query(mock_process_query):
    response = client.post(
        "/api/chat",
        json={"query": "   "}
    )

    assert response.status_code == 400
    assert "detail" in response.json()
    mock_process_query.assert_not_called()
