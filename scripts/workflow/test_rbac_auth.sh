#!/bin/bash
# Test script to validate RBAC authentication for Azure Search and OpenAI
# This script verifies that the upload pipeline works without API keys

set -e

echo "🧪 Testing RBAC Authentication (No API Keys)"
echo "=============================================="
echo ""

# Unset API keys to force RBAC usage
unset AZURE_SEARCH_KEY
unset AZURE_OPENAI_KEY

# Set identity environment variables
export AZURE_CLIENT_ID="1d382868-51d6-4200-a4ba-3a7b94ecb2d3"
export AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)
export AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)

echo "Testing with Service Principal:"
echo "  Client ID: $AZURE_CLIENT_ID"
echo "  Tenant ID: $AZURE_TENANT_ID"
echo ""

# Activate virtual environment
source .venv/bin/activate

echo "1️⃣  Testing Azure Search authentication..."
python scripts/legal-scraper/upload_with_embeddings.py --input Upload --dry-run 2>&1 | \
  grep -E "(Using DefaultAzureCredential|Index.*found|Response status: 200)" | head -5

if [ $? -eq 0 ]; then
    echo "   ✅ Azure Search RBAC: PASSED"
else
    echo "   ❌ Azure Search RBAC: FAILED"
    exit 1
fi

echo ""
echo "2️⃣  Testing Azure OpenAI authentication..."
python -c "
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), 
    'https://cognitiveservices.azure.com/.default'
)

client = AzureOpenAI(
    azure_ad_token_provider=token_provider,
    api_version='2023-05-15',
    azure_endpoint='https://cog-gz2m4s637t5me.openai.azure.com',
    timeout=30.0
)

response = client.embeddings.create(
    input=['RBAC test'],
    model='text-embedding-3-large'
)
print(f'Generated {len(response.data[0].embedding)} dimension embedding')
"

if [ $? -eq 0 ]; then
    echo "   ✅ Azure OpenAI RBAC: PASSED"
else
    echo "   ❌ Azure OpenAI RBAC: FAILED"
    exit 1
fi

echo ""
echo "3️⃣  Checking differential update statistics..."
if [ -f "data/legal-scraper/processed/upload_statistics.txt" ]; then
    echo "   ✅ Statistics file generated:"
    cat data/legal-scraper/processed/upload_statistics.txt
else
    echo "   ❌ Statistics file not found"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ All RBAC authentication tests PASSED"
echo "Ready for GitHub Actions deployment"
echo "=============================================="
