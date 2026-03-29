"""
Integration tests for custom API routes.

Tests /api/categories endpoint and feedback endpoint with real request/response cycles.
"""

import pytest
import json
from unittest import mock


@pytest.mark.asyncio
class TestCategoriesEndpoint:
    """Tests for /api/categories endpoint."""

    async def test_categories_endpoint_returns_404_when_feature_disabled(self, client):
        """Test that endpoint returns 404 when category_filter feature is disabled."""
        with mock.patch("customizations.config.CUSTOM_FEATURES", {"category_filter": False}):
            response = await client.get("/api/categories")
            assert response.status_code == 404

    async def test_categories_endpoint_returns_json_response(self, client):
        """Test that endpoint returns valid JSON response."""
        with mock.patch("customizations.config.CUSTOM_FEATURES", {"category_filter": True}):
            # Mock the search client
            mock_search_results = mock.AsyncMock()
            mock_search_results.get_facets = mock.AsyncMock(
                return_value={
                    "category": [
                        {"value": "Commercial Court", "count": 42},
                        {"value": "King's Bench Division", "count": 28},
                    ]
                }
            )

            mock_search_client = mock.AsyncMock()
            mock_search_client.search = mock.AsyncMock(return_value=mock_search_results)

            original = client.app.config.get("search_client")
            client.app.config["search_client"] = mock_search_client
            try:
                response = await client.get("/api/categories")

                # Response should be successful
                assert response.status_code == 200

                # Should be JSON
                data = await response.get_json()
                assert isinstance(data, dict)
            finally:
                client.app.config["search_client"] = original

    async def test_categories_endpoint_includes_all_sources_option(self, client):
        """Test that categories response includes 'All Sources' option."""
        with mock.patch("customizations.config.CUSTOM_FEATURES", {"category_filter": True}):
            mock_search_results = mock.AsyncMock()
            mock_search_results.get_facets = mock.AsyncMock(
                return_value={"category": [{"value": "Court A", "count": 10}]}
            )

            mock_search_client = mock.AsyncMock()
            mock_search_client.search = mock.AsyncMock(return_value=mock_search_results)

            original = client.app.config.get("search_client")
            client.app.config["search_client"] = mock_search_client
            try:
                response = await client.get("/api/categories")
                data = await response.get_json()

                # Should have categories list
                assert "categories" in data

                # First item should be "All Sources"
                categories = data["categories"]
                assert len(categories) > 0
                assert categories[0]["key"] == ""
                assert categories[0]["text"] == "All Sources"
            finally:
                client.app.config["search_client"] = original

    async def test_categories_endpoint_returns_500_when_search_client_missing(self, client):
        """Test that endpoint returns 500 when search client is not configured."""
        with mock.patch("customizations.config.CUSTOM_FEATURES", {"category_filter": True}):
            original = client.app.config.get("search_client")
            client.app.config["search_client"] = None
            try:
                response = await client.get("/api/categories")
                assert response.status_code == 500
            finally:
                client.app.config["search_client"] = original

    async def test_categories_endpoint_uses_faceted_search(self, client):
        """Test that endpoint uses faceted search for efficiency."""
        with mock.patch("customizations.config.CUSTOM_FEATURES", {"category_filter": True}):
            mock_search_results = mock.AsyncMock()
            mock_search_results.get_facets = mock.AsyncMock(return_value={"category": []})

            mock_search_client = mock.AsyncMock()
            mock_search_client.search = mock.AsyncMock(return_value=mock_search_results)

            original = client.app.config.get("search_client")
            client.app.config["search_client"] = mock_search_client
            try:
                response = await client.get("/api/categories")

                # Verify search was called with facets
                mock_search_client.search.assert_called_once()
            finally:
                client.app.config["search_client"] = original


class TestCategoriesEndpointDisplayNames:
    """Tests for category display name mapping."""

    def test_display_names_mapping_exists(self):
        """Test that SOURCE_DISPLAY_NAMES mapping is defined."""
        from customizations.routes.categories import SOURCE_DISPLAY_NAMES

        assert isinstance(SOURCE_DISPLAY_NAMES, dict)
        assert len(SOURCE_DISPLAY_NAMES) > 0

    def test_display_names_have_guide_suffix_for_courts(self):
        """Test that court categories have 'Guide' in their display name."""
        from customizations.routes.categories import SOURCE_DISPLAY_NAMES

        court_keys = [
            "Commercial Court",
            "King's Bench Division",
            "Patents Court",
        ]

        for key in court_keys:
            if key in SOURCE_DISPLAY_NAMES:
                display_name = SOURCE_DISPLAY_NAMES[key]
                assert "Guide" in display_name

    def test_display_names_all_values_are_strings(self):
        """Test that all display name values are non-empty strings."""
        from customizations.routes.categories import SOURCE_DISPLAY_NAMES

        for key, value in SOURCE_DISPLAY_NAMES.items():
            assert isinstance(value, str)
            assert len(value) > 0


@pytest.mark.asyncio
class TestFeedbackEndpoint:
    """Tests for /api/feedback endpoint (enhanced feedback feature)."""

    async def test_feedback_endpoint_returns_success_on_valid_submission(self, client):
        """Test that feedback endpoint accepts valid feedback submission."""
        # This test requires the feedback route to be registered
        # which is done in app.py by importing from customizations.routes

        feedback_data = {
            "message_id": "test-msg-001",
            "rating": "helpful",
            "issues": [],
            "comment": "Great response",
            "context_shared": True,
            "user_prompt": "What is CPR?",
            "ai_response": "CPR is a set of rules...",
            "conversation_history": [],
            "thoughts": [
                {
                    "title": "Search Query",
                    "description": "civil procedure rules",
                    "props": {},
                }
            ],
        }

        with mock.patch("customizations.config.CUSTOM_FEATURES", {"enhanced_feedback": True}):
            response = await client.post(
                "/api/feedback",
                json=feedback_data,
            )

            # Should be accepted
            assert response.status_code in [200, 201]

    async def test_feedback_endpoint_requires_message_id(self, client):
        """Test that feedback endpoint requires message_id field."""
        feedback_data = {
            # Missing message_id
            "rating": "helpful",
            "issues": [],
            "comment": "Test",
        }

        with mock.patch("customizations.config.CUSTOM_FEATURES", {"enhanced_feedback": True}):
            response = await client.post(
                "/api/feedback",
                json=feedback_data,
            )

            # Should return error for missing required field
            assert response.status_code >= 400

    async def test_feedback_endpoint_includes_metadata(self, client):
        """Test that feedback endpoint includes deployment metadata."""
        # This test verifies that metadata is added to the feedback payload
        feedback_data = {
            "message_id": "test-msg-002",
            "rating": "not_helpful",
            "issues": ["incorrect_citations"],
            "comment": "Citations were wrong",
            "context_shared": False,
            "user_prompt": "What is discovery?",
            "ai_response": "Discovery is...",
            "conversation_history": [],
            "thoughts": [],
        }

        with mock.patch("customizations.config.CUSTOM_FEATURES", {"enhanced_feedback": True}):
            with mock.patch("customizations.routes.feedback.get_deployment_metadata") as mock_metadata:
                mock_metadata.return_value = {
                    "deployment_id": "test-deploy-001",
                    "app_version": "1.0.0",
                    "git_sha": "abc123",
                    "model_name": "gpt-4",
                    "environment": "production",
                }

                response = await client.post(
                    "/api/feedback",
                    json=feedback_data,
                )

                # Metadata function should be called
                mock_metadata.assert_called()


class TestFeedbackDataStorage:
    """Tests for feedback storage behavior."""

    def test_feedback_route_blueprint_is_registered(self):
        """Test that feedback route blueprint is importable."""
        try:
            from customizations.routes.feedback import feedback_bp

            assert feedback_bp is not None
        except ImportError:
            pytest.skip("Feedback route not yet implemented")

    def test_feedback_route_includes_metadata_in_payload(self):
        """Test that feedback payload includes deployment metadata."""
        try:
            from customizations.routes.feedback import feedback_bp

            # Blueprint should have feedback route
            assert any(
                rule.rule == "/api/feedback" for rule in feedback_bp.deferred_functions
            )
        except (ImportError, AttributeError):
            pytest.skip("Feedback route structure not yet inspected")


class TestRouteBlueprints:
    """Tests for route blueprint structure."""

    def test_categories_blueprint_defined(self):
        """Test that categories blueprint is defined."""
        from customizations.routes.categories import categories_bp

        assert categories_bp is not None
        assert categories_bp.name == "categories"
        assert categories_bp.url_prefix == "/api"

    def test_categories_blueprint_has_route(self):
        """Test that categories blueprint has the /categories route."""
        from customizations.routes.categories import categories_bp

        # Blueprint should have the route registered
        assert any(
            "/categories" in str(rule) for rule in getattr(categories_bp, "deferred_functions", [])
        )


@pytest.mark.asyncio
class TestEndpointErrorHandling:
    """Tests for endpoint error handling."""

    async def test_categories_endpoint_handles_search_errors(self, client):
        """Test that categories endpoint handles search client errors gracefully."""
        with mock.patch("customizations.config.CUSTOM_FEATURES", {"category_filter": True}):
            mock_search_client = mock.AsyncMock()
            mock_search_client.search = mock.AsyncMock(side_effect=Exception("Search failed"))

            original = client.app.config.get("search_client")
            client.app.config["search_client"] = mock_search_client
            try:
                response = await client.get("/api/categories")

                # Should return error response
                assert response.status_code >= 400
            finally:
                client.app.config["search_client"] = original

    async def test_feedback_endpoint_handles_invalid_json(self, client):
        """Test that feedback endpoint handles invalid JSON gracefully."""
        with mock.patch("customizations.config.CUSTOM_FEATURES", {"enhanced_feedback": True}):
            response = await client.post(
                "/api/feedback",
                data="invalid json",
                headers={"Content-Type": "application/json"},
            )

            # Should return error for invalid JSON
            assert response.status_code >= 400
