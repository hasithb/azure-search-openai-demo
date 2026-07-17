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

    # Retry retrieval when initial results do not match the user's apparent section intent
    "adaptive_search_retry": True,

    # Allow agentic retrieval to fall back to direct search when references are weak, not just empty
    "agentic_retry_on_weak_matches": True,

    # Supplemental search for related sub-concepts identified by query rewrite
    "related_aspects_search": True,
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


# CUSTOM: Display name mapping shared between categories route and prompt source list
SOURCE_DISPLAY_NAMES = {
    "Commercial Court": "Commercial Court Guide",
    "Circuit Commercial Court": "Circuit Commercial Court Guide",
    "Technology and Construction Court": "Technology and Construction Court Guide",
    "King's Bench Division": "King's Bench Division Guide",
    "Chancery Division": "Chancery Guide",
    "Patents Court": "Patents Court Guide",
    "Civil Procedure Rules and Practice Directions": "Civil Procedure Rules and Practice Directions",
    "Pre-Action Protocols": "Pre-Action Protocols",
    "Court of Appeal Civil Division": "Court of Appeal Civil Division Guide",
    "Senior Courts Costs Office": "Senior Courts Costs Office Guide",
}


async def fetch_available_sources(search_client) -> list[str]:
    """Fetch the list of available document sources from the search index.

    Uses a faceted search on the 'category' field — no documents returned, just
    the unique category values.  Results are mapped through SOURCE_DISPLAY_NAMES
    so the prompt sees user-friendly names.

    Returns a list of display-name strings, e.g.
      ["Chancery Guide", "Civil Procedure Rules and Practice Directions", ...]
    """
    try:
        results = await search_client.search(
            search_text="*",
            facets=["category,count:1000"],
            top=0,
            select=["id"],
        )
        facets = await results.get_facets()
        sources = []
        if facets and "category" in facets:
            for facet in facets["category"]:
                key = facet.get("value", "")
                if key:
                    sources.append(SOURCE_DISPLAY_NAMES.get(key, key))
        sources.sort()
        return sources
    except Exception:
        return []


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
