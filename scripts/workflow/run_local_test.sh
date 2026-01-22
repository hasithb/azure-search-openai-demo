#!/bin/bash

# Local test runner for enhanced feedback system
# This script starts the backend with the cpr-rag environment configuration

set -e

echo "=================================="
echo "🚀 Starting Local Development Server"
echo "=================================="
echo ""

# Check if .venv-upgrade exists, if not use .venv
if [ -d ".venv-upgrade" ]; then
    VENV=".venv-upgrade"
else
    VENV=".venv"
fi

echo "✓ Using virtual environment: $VENV"

# Activate virtual environment
source "$VENV/bin/activate"
echo "✓ Virtual environment activated"

# Load environment from azd
echo ""
echo "📝 Loading Azure environment (cpr-rag)..."
eval "$(azd env get-values | grep -E '^AZURE_|^OPENAI_' | sed 's/^/export /')"
echo "✓ Environment loaded"

# Set development-specific variables
export AZURE_USE_AUTHENTICATION="false"
export AZURE_ENABLE_UNAUTHENTICATED_ACCESS="true"
export USE_USER_UPLOAD="true"  # Enable blob storage for feedback
export QUART_ENV="development"
export QUART_DEBUG="true"

echo ""
echo "🔧 Configuration:"
echo "  • Authentication: DISABLED (dev mode)"
echo "  • Unauthenticated access: ENABLED"
echo "  • Blob storage for feedback: ENABLED"
echo "  • Python path: app/backend"
echo ""

# Start the backend server
cd app/backend

echo "=================================="
echo "🏃 Starting Quart backend server..."
echo "=================================="
echo ""
echo "Backend will be available at: http://127.0.0.1:50505"
echo ""
echo "To test the feedback endpoint, use:"
echo ""
echo "  curl -X POST http://127.0.0.1:50505/api/feedback \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message_id\": \"test-001\", \"rating\": \"helpful\", \"comment\": \"Great answer\"}'"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start with hot reload enabled
python -m quart.cli run --host 127.0.0.1 --port 50505 --reload
