#!/bin/bash
# Phase 3 CLI Automation - GitHub Workflow Management
# Repository: adalex-ai/azure-search-openai-demo

REPO="adalex-ai/azure-search-openai-demo"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              PHASE 3: GitHub Workflow Management                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Function to get latest run ID
get_latest_run_id() {
    gh run list -R "$REPO" --workflow="legal-scraper.yml" --limit 1 --json databaseId -q '.[0].databaseId'
}

# Function to check run status
check_status() {
    local run_id=$1
    gh run view "$run_id" -R "$REPO" --json status,conclusion,displayTitle -q '"Status: \(.status) | Conclusion: \(.conclusion // "in_progress")"'
}

# Show current status
echo "📊 Current Workflow Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
gh run list -R "$REPO" --workflow="legal-scraper.yml" --limit 3
echo ""

# Get latest run ID
RUN_ID=$(get_latest_run_id)
echo "Latest Run ID: $RUN_ID"
echo ""

# Menu
echo "Available Actions:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Watch current dry run (live updates)"
echo "2. View dry run logs"
echo "3. Trigger production run (dry_run=false)"
echo "4. View workflow in browser"
echo "5. Check secret configuration"
echo "6. Cancel current run"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "Select action (1-6): " action

case $action in
    1)
        echo "⏳ Watching workflow run $RUN_ID..."
        gh run watch "$RUN_ID" -R "$REPO"
        ;;
    2)
        echo "📋 Viewing logs for run $RUN_ID..."
        gh run view "$RUN_ID" -R "$REPO" --log
        ;;
    3)
        echo "⚠️  About to trigger PRODUCTION run (dry_run=false)"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            gh workflow run "Legal Document Scraper Pipeline" -R "$REPO" -f dry_run=false -f force_upload=false
            echo "✅ Production workflow triggered!"
            echo "   Monitor with: gh run watch \$(gh run list -R $REPO --workflow=legal-scraper.yml --limit 1 --json databaseId -q '.[0].databaseId') -R $REPO"
        else
            echo "Cancelled."
        fi
        ;;
    4)
        echo "🌐 Opening workflow in browser..."
        gh run view "$RUN_ID" -R "$REPO" --web
        ;;
    5)
        echo "🔑 Current Secrets:"
        gh secret list -R "$REPO" | grep AZURE_SEARCH
        echo ""
        echo "To verify AZURE_SEARCH_INDEX value, check the workflow logs or Azure portal."
        ;;
    6)
        echo "⚠️  Cancelling run $RUN_ID..."
        gh run cancel "$RUN_ID" -R "$REPO"
        echo "✅ Run cancelled."
        ;;
    *)
        echo "Invalid selection."
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Quick Commands:"
echo "  Watch:      gh run watch $RUN_ID -R $REPO"
echo "  View logs:  gh run view $RUN_ID -R $REPO --log"
echo "  Browser:    gh run view $RUN_ID -R $REPO --web"
echo "  Prod run:   gh workflow run \"Legal Document Scraper Pipeline\" -R $REPO -f dry_run=false"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
