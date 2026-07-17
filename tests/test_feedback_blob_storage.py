"""
Tests for feedback blob storage integration.

CUSTOM: Tests to verify that feedback is correctly saved to Azure Blob Storage
when the blob container client is configured, and falls back to local storage
when not configured.
"""

import pytest
import json
import os
from unittest import mock
from io import BytesIO
from datetime import datetime, timezone

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_feedback_saves_to_blob_storage(client):
    """Test that feedback is saved to blob storage when container client is configured."""
    
    # Mock the blob container client
    mock_file_client = mock.AsyncMock()
    mock_file_client.upload_data = mock.AsyncMock()
    
    mock_container_client = mock.MagicMock()
    mock_container_client.get_file_client = mock.MagicMock(return_value=mock_file_client)
    mock_container_client.close = mock.AsyncMock()
    
    # Inject the mock blob container client into the app config
    from config import CONFIG_USER_BLOB_CONTAINER_CLIENT
    client.app.config[CONFIG_USER_BLOB_CONTAINER_CLIENT] = mock_container_client
    
    response = await client.post(
        "/api/feedback",
        json={
            "message_id": "test-blob-001",
            "rating": "helpful",
            "issues": [],
            "comment": "Test feedback for blob storage",
            "context_shared": True,
            "user_prompt": "What is CPR?",
            "ai_response": "CPR is...",
            "conversation_history": [],
            "thoughts": [
                {
                    "title": "Search Query",
                    "description": "civil procedure rules",
                    "props": {}
                }
            ]
        },
    )
    
    assert response.status_code == 200
    result = await response.get_json()
    assert result["status"] == "received"
    
    # Verify get_file_client was called
    assert mock_container_client.get_file_client.called
    
    # Verify upload_data was called at least once (for user-visible feedback)
    assert mock_file_client.upload_data.call_count >= 1
    
    # Check that the uploaded content contains the feedback data
    upload_calls = mock_file_client.upload_data.call_args_list
    uploaded_data = json.loads(upload_calls[0][0][0])
    
    assert uploaded_data["event_type"] == "legal_feedback"
    assert uploaded_data["context_shared"] == True
    assert uploaded_data["payload"]["message_id"] == "test-blob-001"
    assert uploaded_data["payload"]["comment"] == "Test feedback for blob storage"
    
    # Verify context is included
    assert "context" in uploaded_data
    assert uploaded_data["context"]["user_prompt"] == "What is CPR?"
    assert uploaded_data["context"]["ai_response"] == "CPR is..."
    
    # Verify thoughts are present (filtered)
    assert "thoughts" in uploaded_data["context"]
    assert len(uploaded_data["context"]["thoughts"]) == 1
    assert uploaded_data["context"]["thoughts"][0]["title"] == "Search Query"


@pytest.mark.asyncio
async def test_feedback_blob_path_structure(client):
    """Test that blob storage uses correct path structure."""
    
    mock_file_client = mock.AsyncMock()
    mock_file_client.upload_data = mock.AsyncMock()
    
    mock_container_client = mock.MagicMock()
    mock_container_client.get_file_client = mock.MagicMock(return_value=mock_file_client)
    mock_container_client.close = mock.AsyncMock()
    
    from config import CONFIG_USER_BLOB_CONTAINER_CLIENT
    client.app.config[CONFIG_USER_BLOB_CONTAINER_CLIENT] = mock_container_client
    
    response = await client.post(
        "/api/feedback",
        json={
            "message_id": "path-test-001",
            "rating": "helpful",
            "context_shared": False
        },
    )
    
    assert response.status_code == 200
    
    # Check the blob path format
    call_args = mock_container_client.get_file_client.call_args_list[0]
    blob_path = call_args[0][0]
    
    # Should match pattern: feedback/<deployment_id>/<timestamp>_<message_id>.json
    assert blob_path.startswith("feedback/")
    assert "_path-test-001.json" in blob_path
    assert "feedback/" in blob_path


@pytest.mark.asyncio
async def test_feedback_deployment_metadata_in_blob(client):
    """Test that deployment metadata is included in blob storage."""
    
    mock_file_client = mock.AsyncMock()
    mock_file_client.upload_data = mock.AsyncMock()
    
    mock_container_client = mock.MagicMock()
    mock_container_client.get_file_client = mock.MagicMock(return_value=mock_file_client)
    mock_container_client.close = mock.AsyncMock()
    
    from config import CONFIG_USER_BLOB_CONTAINER_CLIENT
    client.app.config[CONFIG_USER_BLOB_CONTAINER_CLIENT] = mock_container_client
    
    # Set some environment variables for metadata
    with mock.patch.dict(os.environ, {
        "DEPLOYMENT_ID": "test-deployment-v1",
        "APP_VERSION": "v1.0.0",
        "GIT_SHA": "abc123def456"
    }):
        response = await client.post(
            "/api/feedback",
            json={
                "message_id": "metadata-test-001",
                "rating": "helpful",
                "context_shared": False
            },
        )
    
    assert response.status_code == 200
    
    # Get uploaded data
    upload_calls = mock_file_client.upload_data.call_args_list
    uploaded_data = json.loads(upload_calls[0][0][0])
    
    # Verify metadata is present
    assert "metadata" in uploaded_data
    metadata = uploaded_data["metadata"]
    
    # Should contain deployment info
    assert "deployment_id" in metadata
    assert "timestamp" in metadata


@pytest.mark.asyncio
async def test_feedback_falls_back_to_local_without_blob(client):
    """Test that feedback falls back to local storage when blob client not configured."""
    
    from config import CONFIG_USER_BLOB_CONTAINER_CLIENT
    
    # Ensure blob container client is NOT configured
    client.app.config[CONFIG_USER_BLOB_CONTAINER_CLIENT] = None
    
    # Mock os.makedirs and open to capture file operations
    with mock.patch("os.makedirs") as mock_makedirs, \
         mock.patch("builtins.open", mock.mock_open()) as mock_file:
        
        response = await client.post(
            "/api/feedback",
            json={
                "message_id": "local-test-001",
                "rating": "helpful",
                "context_shared": False
            },
        )
        
        assert response.status_code == 200
        
        # Verify local storage was used
        mock_makedirs.assert_called()
        mock_file.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
