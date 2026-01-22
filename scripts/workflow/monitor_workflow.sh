#!/bin/bash
# Monitor the legal scraper workflow run

RUN_ID=21048944394
REPO="adalex-ai/azure-search-openai-demo"

echo "🔍 Monitoring workflow run: $RUN_ID"
echo "=================================="
echo ""

while true; do
    clear
    echo "🔍 Workflow Status Monitor"
    echo "=================================="
    echo "Run ID: $RUN_ID"
    echo "Time: $(date '+%H:%M:%S')"
    echo ""
    
    # Get current status
    STATUS=$(gh run view $RUN_ID -R $REPO --json status,conclusion,displayTitle,createdAt,startedAt | jq -r '
        "Status: \(.status)
Conclusion: \(.conclusion // "N/A")
Started: \(.startedAt // "Not yet started")
Created: \(.createdAt)"
    ')
    
    echo "$STATUS"
    echo ""
    echo "=================================="
    echo ""
    
    # Get job status
    echo "📋 Jobs:"
    gh run view $RUN_ID -R $REPO --json jobs | jq -r '.jobs[] | "  \(.name): \(.status) \(if .conclusion then "(\(.conclusion))" else "" end)"'
    echo ""
    
    # Check if completed
    WORKFLOW_STATUS=$(gh run view $RUN_ID -R $REPO --json status -q '.status')
    if [ "$WORKFLOW_STATUS" = "completed" ]; then
        echo "✅ Workflow completed!"
        echo ""
        echo "📊 Viewing recent logs:"
        gh run view $RUN_ID -R $REPO --log 2>&1 | grep -E "documents to upload|unchanged|Processing batch|Successfully uploaded|DIFFERENTIAL" | tail -30
        break
    fi
    
    # Show last few log lines if running
    if [ "$WORKFLOW_STATUS" = "in_progress" ]; then
        echo "📝 Recent activity:"
        gh run view $RUN_ID -R $REPO --log 2>&1 | tail -5
        echo ""
    fi
    
    echo "Refreshing in 15 seconds... (Ctrl+C to stop)"
    sleep 15
done
