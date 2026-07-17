from quart import Blueprint, request, jsonify, current_app
from opentelemetry import trace
import logging
import json
import os
import re
from datetime import datetime, timezone
from azure.storage.blob.aio import ContainerClient, BlobServiceClient
from config import CONFIG_USER_BLOB_CONTAINER_CLIENT, CONFIG_CREDENTIAL
# CUSTOM: Import deployment metadata
from customizations import get_deployment_metadata

feedback_bp = Blueprint("feedback", __name__)
tracer = trace.get_tracer(__name__)
logger = logging.getLogger("azure")

@feedback_bp.post("/api/feedback")
async def submit_feedback():
    data = await request.get_json()

    message_id = data.get("message_id")
    if not message_id:
        return jsonify({"error": "message_id is required"}), 400

    # Sanitize message_id to prevent path traversal attacks
    message_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(message_id))[:128]

    # Check if user consented to share context
    context_shared = data.get("context_shared", False)

    # Log to Application Insights via OpenTelemetry
    with tracer.start_as_current_span("legal_feedback") as span:
        span.set_attribute("feedback.rating", data.get("rating"))
        span.set_attribute("feedback.issues", ",".join(data.get("issues", [])))
        span.set_attribute("feedback.has_comment", bool(data.get("comment")))
        span.set_attribute("feedback.message_id", data.get("message_id"))
        span.set_attribute("feedback.context_shared", context_shared)

        # Add detailed event
        event_data = {
            "rating": data.get("rating"),
            "comment": data.get("comment", "")[:100]  # Truncate for trace
        }
        
        # Include context data if user consented
        if context_shared:
            span.set_attribute("feedback.user_prompt", data.get("user_prompt", "")[:500])
            span.set_attribute("feedback.has_conversation_history", bool(data.get("conversation_history")))
            span.set_attribute("feedback.has_thoughts", bool(data.get("thoughts")))
            event_data["user_prompt"] = data.get("user_prompt", "")[:200]

        span.add_event("feedback_submitted", event_data)

    # Log full JSON for Blob Storage/Log Analytics ingestion
    log_payload = {
        "event_type": "legal_feedback",
        "context_shared": context_shared,
        "payload": {
            "message_id": data.get("message_id"),
            "rating": data.get("rating"),
            "issues": data.get("issues", []),
            "comment": data.get("comment", "")
        }
    }

    # Only include context data if user explicitly consented
    if context_shared:
        log_payload["context"] = {
            "user_prompt": data.get("user_prompt", ""),
            "ai_response": data.get("ai_response", ""),
            "conversation_history": data.get("conversation_history", []),
            "thoughts": data.get("thoughts", [])
        }

    # Save to Azure Blob Storage (Persistent "Folder") or Local Disk
    try:
        # Determine deployment ID from environment variables
        image_name = os.environ.get("SERVICE_BACKEND_IMAGE_NAME", "")
        deployment_id = "local"
        if "azd-deploy-" in image_name:
            # Extract ID from image tag (e.g., ...:azd-deploy-123456)
            deployment_id = image_name.split("azd-deploy-")[-1]
        elif os.environ.get("AZURE_ENV_NAME"):
                deployment_id = os.environ.get("AZURE_ENV_NAME")

        # Generate timestamp and filename
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dt%H-%M-%S")
        message_id = data.get("message_id", "unknown-id")
        
        # CUSTOM: Add comprehensive deployment metadata for debugging
        deployment_metadata = get_deployment_metadata()
        deployment_metadata["deployment_id"] = deployment_id  # Override with actual deployment ID
        deployment_metadata["timestamp"] = timestamp
        
        log_payload["metadata"] = deployment_metadata

        # Structure: feedback/<deployment_id>/<date>_<id>.json
        json_content = json.dumps(log_payload, indent=2)
        blob_name = f"feedback/{deployment_id}/{timestamp}_{message_id}.json"

        # 1. Save locally (User requested "folder in this repo")
        if deployment_id == "local" or not current_app.config.get(CONFIG_USER_BLOB_CONTAINER_CLIENT):
            local_folder = os.path.join(os.getcwd(), "data", "feedback_data", deployment_id)
            os.makedirs(local_folder, exist_ok=True)
            local_path = os.path.join(local_folder, f"{timestamp}_{message_id}.json")
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(json_content)
            logger.info(f"Feedback saved locally: {local_path}")
        
        # 2. Upload to blob storage
        # Option A: User Storage (Data Lake Gen2 / HNS) - Configured via USE_USER_UPLOAD
        user_container_client = current_app.config.get(CONFIG_USER_BLOB_CONTAINER_CLIENT)
        
        # Option B: Standard Blob Storage (Fallback) - Uses main content storage account
        storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT")
        credential = current_app.config.get(CONFIG_CREDENTIAL)
        
        if user_container_client:
            # Note: CONFIG_USER_BLOB_CONTAINER_CLIENT is a FileSystemClient for reduced latency/HNS
            file_client = user_container_client.get_file_client(blob_name)
            await file_client.upload_data(json_content, overwrite=True)
            logger.info(f"Feedback saved to User Storage (HNS): {blob_name}")
            
        elif storage_account and credential:
            # Fallback to standard blob storage in the main account
            # This is "Option 1" for easiest implementation in public repos
            blob_service_client = BlobServiceClient(
                f"https://{storage_account}.blob.core.windows.net", 
                credential=credential
            )
            # Use 'feedback' container
            container_client = blob_service_client.get_container_client("feedback")
            if not await container_client.exists():
                await container_client.create_container()
                
            blob_client = container_client.get_blob_client(blob_name)
            await blob_client.upload_blob(json_content, overwrite=True)
            logger.info(f"Feedback saved to Standard Blob Storage: {blob_name}")
            
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")

    logger.info(json.dumps(log_payload))

    return jsonify({"status": "received"})

