#!/bin/bash
# Local Development: Switch to v2 Index
# This updates the azd environment to use the new index for local testing

set -e

echo "========================================="
echo "Switching Local Environment to v2 Index"
echo "========================================="
echo ""

# Check current index
CURRENT_INDEX=$(azd env get-values 2>/dev/null | grep AZURE_SEARCH_INDEX | cut -d'=' -f2 | tr -d '"')
echo "Current index: $CURRENT_INDEX"

# Update to v2
echo "Updating to: legal-court-rag-index-v2"
azd env set AZURE_SEARCH_INDEX "legal-court-rag-index-v2"

# Verify
NEW_INDEX=$(azd env get-values 2>/dev/null | grep AZURE_SEARCH_INDEX | cut -d'=' -f2 | tr -d '"')
echo "New index: $NEW_INDEX"

echo ""
echo "✅ Local environment updated!"
echo ""
echo "To test the app locally:"
echo "  1. cd app"
echo "  2. ./start.sh"
echo "  3. Open http://localhost:50505"
echo "  4. Test search and chat with legal queries"
echo ""
echo "To switch back to old index:"
echo "  azd env set AZURE_SEARCH_INDEX \"legal-court-rag-index\""
echo ""
