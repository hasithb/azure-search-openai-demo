#!/usr/bin/env python3
"""
Script to sync feedback files from Azure Blob Storage to a local folder.
This allows you to browse blob storage feedback as regular files in VS Code.

Usage:
    python scripts/sync_feedback_from_blob.py              # One-time sync
    python scripts/sync_feedback_from_blob.py --watch      # Continuous sync (every 30s)
    python scripts/sync_feedback_from_blob.py --clean      # Delete local synced folder first
"""

import os
import asyncio
import json
from datetime import datetime
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.storage.blob.aio import ContainerClient
from azure.storage.filedatalake.aio import FileSystemClient

# Local sync directory
SYNC_DIR = Path(__file__).parent.parent / "data" / "feedback_blob_sync"


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


async def sync_feedback_blobs():
    """Sync all feedback blobs to local folder."""
    print("=" * 60)
    print("SYNCING FEEDBACK FROM AZURE BLOB STORAGE")
    print("=" * 60)
    
    # Load environment
    load_azd_env()
    
    # Get configuration
    storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
    user_storage_account = os.environ.get("AZURE_USERSTORAGE_ACCOUNT", "")
    user_storage_container = os.environ.get("AZURE_USERSTORAGE_CONTAINER", "user-content")
    
    if not storage_account:
        print("❌ ERROR: AZURE_STORAGE_ACCOUNT not set!")
        return
    
    # Create sync directory
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Syncing to: {SYNC_DIR}")
    print(f"📦 Storage Account: {storage_account}")
    
    # Create credential
    credential = DefaultAzureCredential()
    
    total_synced = 0
    total_size = 0
    
    # Sync from User Storage (if configured)
    if user_storage_account:
        print(f"\n📂 Syncing from User Storage: {user_storage_account}/{user_storage_container}")
        try:
            # Try as Data Lake first
            fs_client = FileSystemClient(
                f"https://{user_storage_account}.dfs.core.windows.net",
                user_storage_container,
                credential=credential
            )
            
            paths = []
            async for path in fs_client.get_paths(path="feedback"):
                if not path.is_directory:
                    paths.append(path)
            
            for path in paths:
                # Download file
                file_client = fs_client.get_file_client(path.name)
                download = await file_client.download_file()
                content = await download.readall()
                
                # Save locally
                local_path = SYNC_DIR / "user_storage" / path.name.replace("feedback/", "")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(content)
                
                total_synced += 1
                total_size += len(content)
                print(f"   ✓ {path.name} ({len(content)} bytes)")
            
            await fs_client.close()
            
        except Exception as e:
            print(f"   ⚠️  Could not sync from User Storage: {e}")
    
    # Sync from main storage 'feedback' container
    print(f"\n📂 Syncing from Main Storage: {storage_account}/feedback")
    try:
        blob_client = ContainerClient(
            f"https://{storage_account}.blob.core.windows.net",
            "feedback",
            credential=credential
        )
        
        blobs = []
        async for blob in blob_client.list_blobs():
            blobs.append(blob)
        
        for blob in blobs:
            # Download blob
            blob_data_client = blob_client.get_blob_client(blob.name)
            download = await blob_data_client.download_blob()
            content = await download.readall()
            
            # Save locally
            local_path = SYNC_DIR / "feedback_container" / blob.name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(content)
            
            total_synced += 1
            total_size += len(content)
            print(f"   ✓ {blob.name} ({blob.size} bytes)")
        
        await blob_client.close()
        
    except Exception as e:
        print(f"   ⚠️  Could not sync from feedback container: {e}")
    
    # Create a summary file
    summary = {
        "last_sync": datetime.utcnow().isoformat() + "Z",
        "total_files": total_synced,
        "total_size_bytes": total_size,
        "storage_account": storage_account,
        "user_storage_account": user_storage_account or None,
    }
    
    summary_path = SYNC_DIR / "_sync_info.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    
    print(f"\n✅ SYNC COMPLETE")
    print(f"   Total files: {total_synced}")
    print(f"   Total size: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
    print(f"   Local path: {SYNC_DIR}")
    print(f"\n💡 TIP: Open '{SYNC_DIR}' in VS Code to browse feedback files")


async def watch_and_sync(interval=30):
    """Continuously sync feedback blobs at regular intervals."""
    print(f"👀 Watching for new feedback (sync every {interval}s)...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            await sync_feedback_blobs()
            print(f"\n⏱️  Next sync in {interval} seconds...")
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n🛑 Sync stopped by user")


def clean_sync_dir():
    """Remove all synced files."""
    import shutil
    if SYNC_DIR.exists():
        shutil.rmtree(SYNC_DIR)
        print(f"🗑️  Cleaned sync directory: {SYNC_DIR}")
    else:
        print(f"✓ Sync directory doesn't exist: {SYNC_DIR}")


if __name__ == "__main__":
    import sys
    
    if "--clean" in sys.argv:
        clean_sync_dir()
    elif "--watch" in sys.argv:
        asyncio.run(watch_and_sync())
    else:
        asyncio.run(sync_feedback_blobs())
