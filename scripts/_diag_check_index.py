#!/usr/bin/env python3
"""Quick diagnostic: check what the index stores for the 4 false-NEW entries."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from load_azd_env import load_azd_env
load_azd_env()

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

ep = f'https://{os.environ["AZURE_SEARCH_SERVICE"]}.search.windows.net'
idx = os.environ.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v3")
c = SearchClient(endpoint=ep, index_name=idx, credential=DefaultAzureCredential())

# The 4 false-NEW entries
targets = [
    ("Practice Direction 16", "Practice_Direction_16"),
    ("Practice Direction 51Z", "Practice_Direction_51Z___Online_Civil_Money_Claims"),
    ("Practice Direction 51R", "Practice_Direction_51R"),
    ("Practice Direction 63", "Practice_Direction_63"),
]

for label, azure_id in targets:
    print(f"\n=== {label} (azure_id={azure_id}) ===")
    
    # Try direct key lookup
    for key_try in [azure_id, azure_id + "_chunk_000"]:
        try:
            doc = c.get_document(key=key_try, selected_fields=["id", "sourcefile", "sourcepage"])
            print(f"  KEY FOUND: id={doc['id']}, sourcefile={doc['sourcefile']}, sourcepage={doc.get('sourcepage','')}")
        except Exception:
            print(f"  KEY NOT FOUND: {key_try}")
    
    # Full-text search
    results = c.search(
        search_text=label,
        select=["id", "sourcefile", "sourcepage"],
        top=10,
        filter="category eq 'Civil Procedure Rules'",
    )
    found_any = False
    for r in results:
        sf_lower = r["sourcefile"].lower().replace(" ", "")
        sp_lower = r.get("sourcepage", "").lower().replace(" ", "")
        target_lower = label.lower().replace(" ", "")
        if target_lower in sf_lower or target_lower in sp_lower:
            found_any = True
            print(f"  SEARCH HIT: id={r['id']}, sourcefile={r['sourcefile']}, sourcepage={r.get('sourcepage','')}")
    if not found_any:
        print(f"  No search hits matching '{label}'")
