import json
import glob

files = glob.glob("court_guides_processing_pipeline/outputs/*.json")
if files:
    with open(files[0], 'r') as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            print(f"Keys in first item of {files[0]}: {list(data[0].keys())}")
        else:
            print("File is not a list or empty.")
else:
    print("No files found.")
