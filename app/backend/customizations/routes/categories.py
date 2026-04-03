"""
Custom Categories API Route
===========================
This module provides the /api/categories endpoint for category-based filtering.
Kept in a separate blueprint to avoid merge conflicts with upstream app.py.

Usage:
    Register this blueprint in app.py with a single line:
    app.register_blueprint(categories_bp)
"""

from quart import Blueprint, current_app, jsonify

from config import CONFIG_SEARCH_CLIENT

from ..config import SOURCE_DISPLAY_NAMES, is_deployed_ui_compat_enabled, is_feature_enabled

categories_bp = Blueprint("categories", __name__, url_prefix="/api")


@categories_bp.route("/categories", methods=["GET"])
async def get_categories():
    """
    Fetch available sources from Azure Search index using faceted search.
    
    Returns a list of sources with their document counts, useful for
    building source filter dropdowns in the frontend.
    
    Response format:
    {
        "categories": [
            {"key": "", "text": "All Sources", "count": null},
            {"key": "Commercial Court", "text": "Commercial Court Guide", "count": 42},
            ...
        ]
    }
    """
    # Check if feature is enabled
    if not is_feature_enabled("category_filter"):
        return jsonify({"error": "Category filter feature is disabled"}), 404
    
    try:
        search_client = current_app.config.get(CONFIG_SEARCH_CLIENT)
        
        if not search_client:
            return jsonify({"error": "Search client not configured"}), 500

        # Use faceted search to get unique categories efficiently
        results = await search_client.search(
            search_text="*",
            facets=["category,count:1000"],
            top=0,  # Don't return documents, only facets
            select=["id"]  # Minimal field selection
        )

        use_deployed_ui_labels = is_deployed_ui_compat_enabled()

        # Extract categories from facets and map to display names
        categories = []
        facets = await results.get_facets()
        if facets and "category" in facets:
            for facet in facets["category"]:
                if facet.get("value"):
                    category_key = facet["value"]
                    display_name = category_key if use_deployed_ui_labels else SOURCE_DISPLAY_NAMES.get(category_key, category_key)
                    categories.append({
                        "key": category_key,
                        "text": display_name,
                        "count": facet.get("count")
                    })

        # Sort alphabetically by display name
        categories.sort(key=lambda x: x["text"])

        # Add first option at the beginning
        categories.insert(0, {
            "key": "",
            "text": "All Categories" if use_deployed_ui_labels else "All Sources",
            "count": None
        })

        return jsonify({"categories": categories}), 200

    except Exception as e:
        current_app.logger.error(f"Error fetching categories: {e}")
        return jsonify({"error": str(e)}), 500


@categories_bp.before_app_serving
async def log_categories_route():
    """Log that categories route is registered."""
    if is_feature_enabled("category_filter"):
        current_app.logger.info("Custom categories API route registered at /api/categories")
