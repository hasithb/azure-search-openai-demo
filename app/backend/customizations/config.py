# Custom features configuration
# Enable/disable custom features here without modifying upstream code

import os

CUSTOM_FEATURES = {
    # Category filtering feature - adds /api/categories endpoint and UI dropdown
    "category_filter": True,
    
    # Custom citation formatting in prompts
    "legal_domain_prompts": True,
    
    # Frontend citation sanitization
    "citation_sanitizer": True,
    
    # Custom evaluation scripts
    "custom_evals": True,
    
    # Enhanced feedback with deployment metadata and thought filtering
    "enhanced_feedback": True,

    # Force agentic retrieval to always query sources when initial attempt returns no references
    "agentic_force_query_on_empty": True,

    # Fallback to direct search using agentic query plan when references are missing
    "agentic_fallback_search": True,
}

# Security Configuration
# ----------------------
# Security Group ID for "Civil Procedure Copilot Users"
# Automatically assigned to all new documents during ingestion
CIVIL_PROCEDURE_COPILOT_SECURITY_GROUP_ID = "36094ff3-5c6d-49ef-b385-fa37118527e3"


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a custom feature is enabled."""
    return CUSTOM_FEATURES.get(feature_name, False)


def is_deployed_ui_compat_enabled() -> bool:
    """Enable deployed UI-compatible responses without disabling newer upstream features."""
    return os.getenv("DEPLOYED_UI_COMPAT", "false").lower() == "true"


def get_deployment_metadata() -> dict[str, str]:
    """
    Get deployment and version metadata for feedback tracking.
    
    Includes deployment ID, app version, and Git commit hash if available.
    This information is stored with feedback to enable version-specific debugging.
    
    Returns:
        Dictionary containing deployment metadata
        
    Example:
        {
            "deployment_id": "1767305857",
            "app_version": "1.0.0",
            "git_sha": "abc123def456",
            "deployment_timestamp": "2026-01-10T12:00:00Z"
        }
    """
    return {
        "deployment_id": os.getenv("DEPLOYMENT_ID", "unknown"),
        "app_version": os.getenv("APP_VERSION", "0.0.0"),
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "model_name": os.getenv("AZURE_OPENAI_CHATGPT_MODEL", "gpt-4"),
        "environment": os.getenv("RUNNING_IN_PRODUCTION", "false").lower() == "true" and "production" or "development",
    }
