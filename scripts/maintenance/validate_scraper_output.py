import os
import json
import glob
from datetime import datetime

UPLOAD_DIR = "data/legal-scraper/processed/Upload"
REQUIRED_OIDS = ["all"]
REQUIRED_GROUPS = ["all", "36094ff3-5c6d-49ef-b385-fa37118527e3"]

def validate_files():
    files = glob.glob(os.path.join(UPLOAD_DIR, "*.json"))
    print(f"Scanning {len(files)} files in {UPLOAD_DIR}...")
    
    errors = []
    valid_count = 0
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            
            # Check OIDS
            if doc.get("oids") != REQUIRED_OIDS:
                errors.append(f"{file_name}: Invalid oids. Expected {REQUIRED_OIDS}, got {doc.get('oids')}")
                continue
                
            # Check GROUPS
            if doc.get("groups") != REQUIRED_GROUPS:
                errors.append(f"{file_name}: Invalid groups. Expected {REQUIRED_GROUPS}, got {doc.get('groups')}")
                continue
                
            # Check Updated
            updated = doc.get("updated")
            if not updated:
                errors.append(f"{file_name}: Missing updated date")
                continue
            
            # Basic content check
            if not doc.get("content"):
                errors.append(f"{file_name}: Empty content")
                continue

            valid_count += 1
            
        except Exception as e:
            errors.append(f"{file_name}: JSON Load Error - {str(e)}")

    print(f"\nModel Validation Complete.")
    print(f"Total Files: {len(files)}")
    print(f"Valid Files: {valid_count}")
    print(f"Invalid Files: {len(errors)}")
    
    if errors:
        print("\nErrors Found:")
        for err in errors[:20]: # Show first 20 errors
            print(f" - {err}")
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more.")
    else:
        print("\n✅ All files match the required V2 Schema (OIDs, Groups, Updated).")

if __name__ == "__main__":
    validate_files()
