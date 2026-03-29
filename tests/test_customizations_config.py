"""
Unit tests for customization configuration module.

Tests feature flag management and deployment metadata functions.
"""

import os
import pytest
from unittest import mock


class TestFeatureFlags:
    """Tests for feature flag management."""

    def test_is_feature_enabled_returns_true_for_enabled_feature(self):
        """Test that is_feature_enabled returns True for enabled features."""
        from customizations.config import is_feature_enabled

        # Features that are enabled by default
        assert is_feature_enabled("category_filter") is True
        assert is_feature_enabled("legal_domain_prompts") is True
        assert is_feature_enabled("citation_sanitizer") is True
        assert is_feature_enabled("custom_evals") is True
        assert is_feature_enabled("enhanced_feedback") is True

    def test_is_feature_enabled_returns_false_for_disabled_feature(self):
        """Test that is_feature_enabled returns False for non-existent features."""
        from customizations.config import is_feature_enabled

        assert is_feature_enabled("non_existent_feature") is False

    def test_is_feature_enabled_returns_false_for_explicitly_disabled_feature(self):
        """Test that is_feature_enabled respects disabled features."""
        from customizations.config import is_feature_enabled, CUSTOM_FEATURES

        # Temporarily disable a feature
        original_value = CUSTOM_FEATURES.get("category_filter")
        try:
            CUSTOM_FEATURES["category_filter"] = False
            assert is_feature_enabled("category_filter") is False
        finally:
            # Restore original state
            CUSTOM_FEATURES["category_filter"] = original_value

    def test_custom_features_dict_contains_all_expected_keys(self):
        """Test that CUSTOM_FEATURES contains all expected feature flags."""
        from customizations.config import CUSTOM_FEATURES

        expected_keys = {
            "category_filter",
            "legal_domain_prompts",
            "citation_sanitizer",
            "custom_evals",
            "enhanced_feedback",
            "agentic_force_query_on_empty",
            "agentic_fallback_search",
            "adaptive_search_retry",
            "agentic_retry_on_weak_matches",
        }
        assert set(CUSTOM_FEATURES.keys()) == expected_keys

    def test_all_feature_flags_are_boolean(self):
        """Test that all feature flags have boolean values."""
        from customizations.config import CUSTOM_FEATURES

        for key, value in CUSTOM_FEATURES.items():
            assert isinstance(value, bool), f"Feature '{key}' should have a boolean value, got {type(value)}"


class TestDeploymentMetadata:
    """Tests for deployment metadata extraction."""

    def test_get_deployment_metadata_structure(self):
        """Test that get_deployment_metadata returns required keys."""
        from customizations.config import get_deployment_metadata

        metadata = get_deployment_metadata()

        required_keys = {
            "deployment_id",
            "app_version",
            "git_sha",
            "model_name",
            "environment",
        }
        assert set(metadata.keys()) == required_keys

    def test_get_deployment_metadata_all_values_are_strings(self):
        """Test that all metadata values are strings."""
        from customizations.config import get_deployment_metadata

        metadata = get_deployment_metadata()

        for key, value in metadata.items():
            assert isinstance(value, str), f"Metadata '{key}' should be a string, got {type(value)}"

    def test_get_deployment_metadata_uses_environment_variables(self):
        """Test that deployment metadata reads from environment variables."""
        from customizations.config import get_deployment_metadata

        with mock.patch.dict(
            os.environ,
            {
                "DEPLOYMENT_ID": "test-deployment-123",
                "APP_VERSION": "2.0.0",
                "GIT_SHA": "abc123def456",
                "AZURE_OPENAI_CHATGPT_MODEL": "gpt-4-turbo",
                "RUNNING_IN_PRODUCTION": "true",
            },
        ):
            metadata = get_deployment_metadata()

            assert metadata["deployment_id"] == "test-deployment-123"
            assert metadata["app_version"] == "2.0.0"
            assert metadata["git_sha"] == "abc123def456"
            assert metadata["model_name"] == "gpt-4-turbo"
            assert metadata["environment"] == "production"

    def test_get_deployment_metadata_uses_defaults_when_env_vars_missing(self):
        """Test that metadata falls back to defaults when env vars are not set."""
        from customizations.config import get_deployment_metadata

        # Clear relevant environment variables
        env_vars_to_clear = {
            "DEPLOYMENT_ID",
            "APP_VERSION",
            "GIT_SHA",
            "AZURE_OPENAI_CHATGPT_MODEL",
            "RUNNING_IN_PRODUCTION",
        }

        with mock.patch.dict(os.environ, {}, clear=False):
            for var in env_vars_to_clear:
                if var in os.environ:
                    del os.environ[var]

            metadata = get_deployment_metadata()

            assert metadata["deployment_id"] == "unknown"
            assert metadata["app_version"] == "0.0.0"
            assert metadata["git_sha"] == "unknown"
            assert metadata["model_name"] == "gpt-4"
            assert metadata["environment"] == "development"

    def test_get_deployment_metadata_environment_detection_production(self):
        """Test that environment is correctly detected as production."""
        from customizations.config import get_deployment_metadata

        with mock.patch.dict(
            os.environ,
            {"RUNNING_IN_PRODUCTION": "true"},
            clear=False,
        ):
            metadata = get_deployment_metadata()
            assert metadata["environment"] == "production"

    def test_get_deployment_metadata_environment_detection_development(self):
        """Test that environment defaults to development."""
        from customizations.config import get_deployment_metadata

        with mock.patch.dict(
            os.environ,
            {"RUNNING_IN_PRODUCTION": "false"},
            clear=False,
        ):
            metadata = get_deployment_metadata()
            assert metadata["environment"] == "development"

        # Also test when variable is not set
        env_copy = os.environ.copy()
        if "RUNNING_IN_PRODUCTION" in env_copy:
            del env_copy["RUNNING_IN_PRODUCTION"]

        with mock.patch.dict(os.environ, env_copy, clear=True):
            metadata = get_deployment_metadata()
            assert metadata["environment"] == "development"

    def test_get_deployment_metadata_case_insensitive_production_check(self):
        """Test that RUNNING_IN_PRODUCTION check is case-insensitive."""
        from customizations.config import get_deployment_metadata

        test_cases = [
            ("TRUE", "production"),
            ("True", "production"),
            ("FALSE", "development"),
            ("False", "development"),
            ("anything_else", "development"),
        ]

        for env_value, expected_env in test_cases:
            with mock.patch.dict(
                os.environ,
                {"RUNNING_IN_PRODUCTION": env_value},
                clear=False,
            ):
                metadata = get_deployment_metadata()
                assert metadata["environment"] == expected_env


class TestSecurityConfiguration:
    """Tests for security configuration."""

    def test_security_group_id_is_defined(self):
        """Test that CIVIL_PROCEDURE_COPILOT_SECURITY_GROUP_ID is defined."""
        from customizations.config import CIVIL_PROCEDURE_COPILOT_SECURITY_GROUP_ID

        assert CIVIL_PROCEDURE_COPILOT_SECURITY_GROUP_ID is not None
        assert isinstance(CIVIL_PROCEDURE_COPILOT_SECURITY_GROUP_ID, str)
        assert len(CIVIL_PROCEDURE_COPILOT_SECURITY_GROUP_ID) > 0

    def test_security_group_id_is_valid_uuid_format(self):
        """Test that security group ID looks like a valid UUID."""
        import re

        from customizations.config import CIVIL_PROCEDURE_COPILOT_SECURITY_GROUP_ID

        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        assert re.match(uuid_pattern, CIVIL_PROCEDURE_COPILOT_SECURITY_GROUP_ID.lower())
