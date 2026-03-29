import json

data = json.load(open("legal-court-rag-index-v3_backup.json"))
docs = data["documents"]
print(f"Total: {len(docs)}")

cpr = [d for d in docs if d.get("category") == "Civil Procedure Rules and Practice Directions"]

# CPR Parts
parts = [d for d in cpr if d.get("sourcefile", "").startswith("Part ")]
seen = set()
print("\n=== CPR PARTS ===")
for d in parts:
    sf = d.get("sourcefile", "")
    if sf not in seen:
        seen.add(sf)
        print(f"  sp={d.get('sourcepage','')!r}")
        print(f"    sf={sf!r}  sid={d.get('subsection_id','')!r}")
        if len(seen) >= 15:
            break

# Practice Directions
pds = [d for d in cpr if "Practice Direction" in d.get("sourcefile", "") or "Practice Direction" in d.get("sourcepage", "")]
seen2 = set()
print("\n=== PRACTICE DIRECTIONS ===")
for d in pds:
    sf = d.get("sourcefile", "")
    if sf not in seen2:
        seen2.add(sf)
        print(f"  sp={d.get('sourcepage','')!r}")
        print(f"    sf={sf!r}  sid={d.get('subsection_id','')!r}")
        if len(seen2) >= 15:
            break

# Pre-Action Protocols
protos = [d for d in cpr if "Protocol" in d.get("sourcepage", "")]
seen3 = set()
print("\n=== PRE-ACTION PROTOCOLS ===")
for d in protos:
    sp = d.get("sourcepage", "")
    if sp not in seen3:
        seen3.add(sp)
        print(f"  sp={sp!r}")
        print(f"    sf={d.get('sourcefile','')!r}  sid={d.get('subsection_id','')!r}")
        if len(seen3) >= 10:
            break

# All subsection_id patterns
print("\n=== DIVERSE subsection_id VALUES ===")
sids = set()
for d in docs:
    sid = d.get("subsection_id", "")
    if sid:
        sids.add(sid)
# Sort and show diverse examples
import re
numeric = sorted([s for s in sids if re.match(r"^\d+\.?\d*$", s)], key=lambda x: float(x.replace(".","",1) if x.count(".")==1 else x))
alpha = sorted([s for s in sids if re.match(r"^[A-Z]", s)])
other = sorted([s for s in sids if not re.match(r"^\d", s) and not re.match(r"^[A-Z]", s)])
print(f"  Numeric({len(numeric)}): {numeric[:20]}")
print(f"  Alpha({len(alpha)}): {alpha[:30]}")
print(f"  Other({len(other)}): {other[:10]}")

# Court Guides - show sourcepage patterns
for cat in ["Commercial Court", "Technology and Construction Court", "Chancery Division", "King's Bench Division", "Patents Court"]:
    guides = [d for d in docs if d.get("category") == cat]
    print(f"\n=== {cat} ({len(guides)} docs) ===")
    seeng = set()
    for d in guides[:8]:
        sp = d.get("sourcepage", "")
        if sp not in seeng:
            seeng.add(sp)
            print(f"  sp={sp!r}")
            print(f"    sf={d.get('sourcefile','')!r}  sid={d.get('subsection_id','')!r}")

# Circuit Commercial
cc = [d for d in docs if d.get("category") == "Circuit Commercial Court"]
print(f"\n=== Circuit Commercial Court ({len(cc)} docs) ===")
for d in cc[:3]:
    print(f"  sp={d.get('sourcepage','')!r}  sf={d.get('sourcefile','')!r}")
