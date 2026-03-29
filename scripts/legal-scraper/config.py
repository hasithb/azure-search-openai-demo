#!/usr/bin/env python
"""
Configuration for legal document scraper.
Reads from azd environment variables when available, falls back to .env or defaults.
"""
import os
import subprocess
from typing import Optional
from pathlib import Path

def get_azd_env_var(var_name: str) -> Optional[str]:
    """Get environment variable from azd context."""
    try:
        # Try to get from azd environment
        result = subprocess.run(
            ["azd", "env", "get-values"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() == var_name:
                        return value.strip().strip('"')
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None

def get_env_var(var_name: str, default: str = "", env_mapping: Optional[dict] = None) -> str:
    """
    Get environment variable with fallback chain:
    1. Direct environment variable
    2. AZD environment
    3. Mapped azd variable name (if provided)
    4. Default value
    """
    # First try direct environment variable
    value = os.getenv(var_name)
    if value:
        return value
    
    # Try mapped azd variable
    if env_mapping and var_name in env_mapping:
        azd_var = env_mapping[var_name]
        value = get_azd_env_var(azd_var)
        if value:
            return value
    
    # Return default
    return default

# Mapping from local env vars to azd variables
AZD_ENV_MAPPING = {
    'AZURE_SEARCH_SERVICE': 'AZURE_SEARCH_SERVICE',
    'AZURE_SEARCH_KEY': 'AZURE_SEARCH_KEY',
    'AZURE_OPENAI_SERVICE': 'AZURE_OPENAI_SERVICE',
    'AZURE_OPENAI_EMB_DEPLOYMENT': 'AZURE_OPENAI_EMB_DEPLOYMENT',
    'AZURE_OPENAI_KEY': 'AZURE_OPENAI_KEY',
}

class Config:
    """Configuration for legal document scraper."""
    
    # Azure Search settings - read from azd or environment
    AZURE_SEARCH_SERVICE = get_env_var(
        'AZURE_SEARCH_SERVICE',
        '',
        AZD_ENV_MAPPING
    )
    # Prefer RBAC: Don't auto-load keys from AZD. Only use if explicitly set in environment.
    AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY', '')

    AZURE_SEARCH_INDEX = os.getenv('AZURE_SEARCH_INDEX', 'legal-court-rag-index-v3')
    AZURE_SEARCH_INDEX_STAGING = os.getenv('AZURE_SEARCH_INDEX_STAGING', 'legal-court-rag-staging')
    
    # Azure OpenAI settings for embeddings
    AZURE_OPENAI_SERVICE = get_env_var(
        'AZURE_OPENAI_SERVICE',
        '',
        AZD_ENV_MAPPING
    )
    # Prefer RBAC: Don't auto-load keys from AZD. Only use if explicitly set in environment.
    AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY', '')

    AZURE_OPENAI_EMB_DEPLOYMENT = get_env_var(
        'AZURE_OPENAI_EMB_DEPLOYMENT',
        'text-embedding-3-large',
        AZD_ENV_MAPPING
    )
    
    # Embedding settings
    EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large
    
    # Document processing settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    MIN_CONTENT_LENGTH = 100  # Minimum characters for valid document
    
    # Legal terminology validation
    LEGAL_TERMS = {
        "court", "rule", "practice direction", "procedure",
        "claimant", "defendant", "proceedings", "order",
        "judgment", "application", "hearing", "parties",
        "section", "part", "paragraph", "regulation"
    }
    MIN_LEGAL_TERMS = 1  # Minimum legal terms to consider valid
    
    # Paths - relative to project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    SCRAPER_DATA_DIR = os.path.join(DATA_DIR, "legal-scraper")
    CIVIL_RULES_DIR = os.path.join(SCRAPER_DATA_DIR, "civil_rules")
    PROCESSED_DIR = os.path.join(SCRAPER_DATA_DIR, "processed")
    UPLOAD_DIR = os.path.join(PROCESSED_DIR, "Upload")
    CACHE_DIR = os.path.join(SCRAPER_DATA_DIR, "cache")
    VALIDATION_REPORT_DIR = os.path.join(SCRAPER_DATA_DIR, "validation-reports")
    
    # Ensure directories exist
    for directory in [DATA_DIR, SCRAPER_DATA_DIR, CIVIL_RULES_DIR, PROCESSED_DIR, 
                      UPLOAD_DIR, CACHE_DIR, VALIDATION_REPORT_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Validation settings
    VERBOSE = os.getenv('VERBOSE', 'true').lower() == 'true'
    
    @classmethod
    def validate_azure_config(cls) -> tuple[bool, list[str]]:
        """Validate that all required Azure configuration is available."""
        errors = []
        
        if not cls.AZURE_SEARCH_SERVICE:
            errors.append("AZURE_SEARCH_SERVICE not configured")
        # Key is optional if using RBAC
        # if not cls.AZURE_SEARCH_KEY:
        #     errors.append("AZURE_SEARCH_KEY not configured")
        if not cls.AZURE_OPENAI_SERVICE:
            errors.append("AZURE_OPENAI_SERVICE not configured")
        # Key is optional if using RBAC
        # if not cls.AZURE_OPENAI_KEY:
        #     errors.append("AZURE_OPENAI_KEY not configured")
        
        return len(errors) == 0, errors
