import json, os, glob
from collections import Counter

base = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(base, "..", "outputs_azure_di", "*_processed.json")))

for f in files:
    with open(f) as fh:
        docs = json.load(fh)
    name = os.path.basename(f)[:55]
    print("\n" + "=" * 100)
    print("%s - %d docs" % (name, len(docs)))
    print("=" * 100)

    # First 5 docs
    print("\n  FIRST 5:")
    for i in range(min(5, len(docs))):
        d = docs[i]
        sp = d["sourcepage"]
        clen = len(d["content"])
        preview = d["content"][:140].replace("\n", " ")
        print("    [%3d] (%5d chars) sp=%s" % (i, clen, sp))
        print("          content: %s..." % preview)

    # Last 3 docs
    print("\n  LAST 3:")
    for i in range(max(0, len(docs)-3), len(docs)):
        d = docs[i]
        sp = d["sourcepage"]
        clen = len(d["content"])
        preview = d["content"][:140].replace("\n", " ")
        print("    [%3d] (%5d chars) sp=%s" % (i, clen, sp))
        print("          content: %s..." % preview)

    # Find problematic docs
    problems = []
    for i, d in enumerate(docs):
        sp = d["sourcepage"]
        c = d["content"]
        clen = len(c)
        issues = []
        if sp in ("Untitled", "OGL"):
            issues.append("BAD_SOURCEPAGE=%s" % sp)
        if clen < 100:
            issues.append("TOO_SHORT=%d" % clen)
        if "PageFooter" in c or "PageHeader" in c:
            issues.append("HAS_PAGE_ARTIFACTS")
        if "<!-- " in c:
            issues.append("HAS_HTML_COMMENTS")
        if "Open Government Licence" in c or "Crown copyright" in c:
            issues.append("OGL_BOILERPLATE")
        if c.strip().startswith("| ") and c.count("|") > 20 and clen < 300:
            issues.append("TABLE_ONLY_SHORT")
        if issues:
            problems.append((i, sp, clen, issues, c[:180].replace("\n", " ")))

    if problems:
        print("\n  PROBLEMS (%d):" % len(problems))
        for idx, sp, clen, iss, preview in problems:
            print("    [%3d] (%5d chars) %s" % (idx, clen, ", ".join(iss)))
            print("          sp=%s" % sp)
            print("          content: %s..." % preview)
    else:
        print("\n  PROBLEMS: None found")

    # Check for duplicate sourcepages
    sps = [d["sourcepage"] for d in docs]
    dupes = {sp: cnt for sp, cnt in Counter(sps).items() if cnt > 1}
    if dupes:
        print("\n  DUPLICATE SOURCEPAGES (%d):" % len(dupes))
        for sp, cnt in sorted(dupes.items(), key=lambda x: -x[1])[:8]:
            # find which doc indices
            indices = [j for j, d in enumerate(docs) if d["sourcepage"] == sp]
            print("    x%d: %s (docs %s)" % (cnt, sp, indices))

    # Metadata check
    cats = set(d["category"] for d in docs)
    sfiles = set(d["sourcefile"] for d in docs)
    updated = set(d["updated"] for d in docs)
    print("\n  METADATA: categories=%s" % cats)
    print("            sourcefiles=%s" % sfiles)
    print("            updated=%s" % updated)
    
    # parent_id check
    parent_ids = [d["parent_id"] for d in docs]
    null_parents = sum(1 for p in parent_ids if p is None or p == "")
    print("            parent_ids: %d null, %d set" % (null_parents, len(parent_ids) - null_parents))
