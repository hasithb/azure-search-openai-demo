#!/usr/bin/env python3
"""
Script to check if feedback JSON files are being saved to Azure Blob Storage.
This script:
1. Checks the current configuration for user storage
2. Lists any existing feedback files in blob storage
3. Optionally sends a test feedback to verify storage is working
"""

import os
import asyncio
import json
from datetime import datetime, timezone
from azure.identity import DefaultAzureCredential
from azure.storage.blob.aio import ContainerClient
from azure.storage.filedatalake.aio import FileSystemClient

# Load environment from azd
def load_azd_env():
    """Load environment variables from azd."""
    import subprocess
    try:
        result = subprocess.run(
            ["azd", "env", "get-values"],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value.strip('"')
    except Exception as e:
        print(f"Warning: Could not load azd env: {e}")


async def check_blob_storage_feedback():
    """Check blob storage for feedback files."""
    print("=" * 60)
    print("AZURE FEEDBACK STORAGE CHECKER")
    print("=" * 60)
    
    # Load azd environment
    load_azd_env()
    
    # Get storage configuration
    storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
    storage_container = os.environ.get("AZURE_STORAGE_CONTAINER", "content")
    user_storage_account = os.environ.get("AZURE_USERSTORAGE_ACCOUNT", "")
    user_storage_container = os.environ.get("AZURE_USERSTORAGE_CONTAINER", "user-content")
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
    
    print("\n📋 CURRENT CONFIGURATION:")
    print(f"  Main Storage Account: {storage_account or '(not set)'}")
    print(f"  Main Container: {storage_container}")
    print(f"  User Storage Account: {user_storage_account or '(NOT SET - feedback saves locally!)'}")
    print(f"  User Storage Container: {user_storage_container}")
    print(f"  Subscription ID: {subscription_id}")
    
    # Check if user storage is configured
    if not user_storage_account:
        print("\n⚠️  WARNING: AZURE_USERSTORAGE_ACCOUNT is not set!")
        print("   Feedback is being saved LOCALLY only, not to Azure Blob Storage.")
        print("   To enable blob storage:")
        print("   1. Create a storage account with hierarchical namespace (Data Lake Gen2)")
        print("   2. Set AZURE_USERSTORAGE_ACCOUNT in your azd environment")
        print("   3. Ensure the container exists")
        
        # Check local feedback files
        print("\n📁 LOCAL FEEDBACK FILES:")
        local_path = os.path.join(os.getcwd(), "data", "feedback_data")
        if os.path.exists(local_path):
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    if file.endswith(".json"):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, local_path)
                        file_size = os.path.getsize(full_path)
                        print(f"   ✓ {rel_path} ({file_size} bytes)")
        else:
            print("   No local feedback folder found.")
        # Continue to check fallbacks
    
    # Try to list feedback from user blob storage (if configured)
    if user_storage_account:
        print("\n🔍 CHECKING USER BLOB STORAGE FOR FEEDBACK...")
        
        try:
            credential = DefaultAzureCredential()
            
            # Try as Data Lake Gen2 (FileSystemClient)
            print(f"\n   Connecting to: https://{user_storage_account}.dfs.core.windows.net/{user_storage_container}")
            fs_client = FileSystemClient(
                f"https://{user_storage_account}.dfs.core.windows.net",
                user_storage_container,
                credential=credential
            )
            
            feedback_files = []
            async for path in fs_client.get_paths(path="feedback/"):
                feedback_files.append(path)
            
            if feedback_files:
                print(f"\n✅ FOUND {len(feedback_files)} FEEDBACK FILE(S):")
                for f in feedback_files[:20]:  # Show first 20
                    print(f"   📄 {f.name} ({f.content_length or 'dir'} bytes)")
                    
                    # Read and display the most recent feedback
                    if f.is_directory == False and f.name.endswith(".json"):
                        file_client = fs_client.get_file_client(f.name)
                        download = await file_client.download_file()
                        content = await download.readall()
                        data = json.loads(content)
                        print(f"      Rating: {data.get('payload', {}).get('rating', 'N/A')}")
                        print(f"      Deployment: {data.get('metadata', {}).get('deployment_id', 'N/A')}")
                
                if len(feedback_files) > 20:
                    print(f"   ... and {len(feedback_files) - 20} more files")
            else:
                print("\n⚠️  No feedback files found in blob storage.")
                print("   The feedback/ prefix may not exist yet or no feedback has been submitted.")
                
            await fs_client.close()
            
        except Exception as e:
            print(f"\n❌ ERROR accessing blob storage: {e}")
            print("   This could mean:")
            print("   - The storage account doesn't have hierarchical namespace enabled")
            print("   - Authentication failed (run 'az login' first)")
            print("   - The container doesn't exist")
            
            # Fallback: Try as regular blob storage
            print("\n   Trying as regular Blob Storage...")
            try:
                blob_client = ContainerClient(
                    f"https://{user_storage_account}.blob.core.windows.net",
                    user_storage_container,
                    credential=credential
                )
                
                blobs = []
                async for blob in blob_client.list_blobs(name_starts_with="feedback/"):
                    blobs.append(blob)
                
                if blobs:
                    print(f"\n✅ FOUND {len(blobs)} FEEDBACK BLOB(S):")
                    for b in blobs[:20]:
                        print(f"   📄 {b.name} ({b.size} bytes)")
                else:
                    print("   No feedback blobs found in container.")
                    
                await blob_client.close()
                
            except Exception as e2:
                print(f"   Also failed as Blob Storage: {e2}")
    
    # Also check main storage account for feedback in 'feedback' container (Fallback location)
    print(f"\n📂 CHECKING MAIN STORAGE ({storage_account}) 'feedback' CONTAINER...")
    try:
        credential = DefaultAzureCredential()
        # Check 'feedback' container specifically
        blob_client = ContainerClient(
            f"https://{storage_account}.blob.core.windows.net",
            "feedback",
            credential=credential
        )
        
        blobs = []
        async for blob in blob_client.list_blobs():
             blobs.append(blob)
        
        if blobs:
            print(f"\n✅ FOUND {len(blobs)} FEEDBACK BLOB(S) IN FALLBACK CONTAINER ('feedback'):")
            for b in blobs[:10]:
                print(f"   📄 {b.name} ({b.size} bytes)")
        else:
            print("   No feedback files in fallback 'feedback' container.")
            
        await blob_client.close()

    except Exception as e:
        print(f"   Could not check main storage feedback container: {e}")

    # Also check main storage account for configured container (for legacy/other reasons)
    print(f"\n📂 CHECKING MAIN STORAGE ({storage_account}) '{storage_container}' CONTAINER...")
    try:
        credential = DefaultAzureCredential()
        blob_client = ContainerClient(
            f"https://{storage_account}.blob.core.windows.net",
            storage_container,
            credential=credential
        )
        
        blobs = []
        async for blob in blob_client.list_blobs(name_starts_with="feedback/"):
            blobs.append(blob)
        
        if blobs:
            print(f"\n✅ FOUND {len(blobs)} FEEDBACK BLOB(S) IN MAIN STORAGE:")
            for b in blobs[:10]:
                print(f"   📄 {b.name} ({b.size} bytes)")
        else:
            print("   No feedback files in main storage container.")
            
        await blob_client.close()
        
    except Exception as e:
        print(f"   Could not check main storage: {e}")


async def send_test_feedback():
    """Send a test feedback request to verify the endpoint."""
    import aiohttp
    
    print("\n" + "=" * 60)
    print("SENDING TEST FEEDBACK")
    print("=" * 60)
    
    test_data = {
        "message_id": f"storage-test-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        "rating": "helpful",
        "issues": [],
        "comment": "Test feedback for storage verification",
        "context_shared": True,
        "user_prompt": "Test question about CPR rules",
        "ai_response": "Test response about Civil Procedure Rules",
        "thoughts": [
            {"title": "Test Thought", "description": "Storage verification test"}
        ]
    }
    
    # Try local development server first
    urls = [
        "http://localhost:50505/api/feedback",
        "http://127.0.0.1:50505/api/feedback"
    ]
    
    for url in urls:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=test_data, timeout=5) as resp:
                    result = await resp.json()
                    print(f"\n✅ Test feedback sent successfully to {url}")
                    print(f"   Response: {result}")
                    return
        except Exception as e:
            print(f"   Could not reach {url}: {e}")
    
    print("\n⚠️  Could not send test feedback - is the backend running?")
    print("   Start the backend with: ./scripts/start.sh or 'azd up'")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(send_test_feedback())
    else:
        asyncio.run(check_blob_storage_feedback())
        print("\n" + "-" * 60)
        print("TIP: Run with --test flag to send a test feedback request")
        print("-" * 60)
