#!/bin/bash
export AZURE_SEARCH_SERVICE="https://cpr-rag.search.windows.net"
export AZURE_SEARCH_INDEX="legal-court-rag-index"
export AZURE_OPENAI_SERVICE="$(azd env get-values | grep AZURE_OPENAI_SERVICE | cut -d'=' -f2 | tr -d '"')"
export AZURE_OPENAI_EMB_DEPLOYMENT="$(azd env get-values | grep AZURE_OPENAI_EMB_DEPLOYMENT | cut -d'=' -f2 | tr -d '"')"

echo "Testing with configuration:"
echo "  Search Service: $AZURE_SEARCH_SERVICE"
echo "  Search Index: $AZURE_SEARCH_INDEX"
echo "  OpenAI Service: $AZURE_OPENAI_SERVICE"
echo "  Embedding Model: $AZURE_OPENAI_EMB_DEPLOYMENT"
echo ""
echo "Running dry-run test..."

/Users/HasithB/Downloads/PROJECTS/azure-search-openai-demo-2/.venv/bin/python scripts/legal-scraper/upload_with_embeddings.py --input Upload --dry-run
