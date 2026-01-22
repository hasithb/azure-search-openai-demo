import json
from pathlib import Path
import re


def extract_source_filename(content: str) -> str | None:
    if not content:
        return None
    match = re.search(r"^SOURCE:\s*(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def main() -> None:
    junk_path = Path("evals/results/v2_highlight_accuracy_junks_full.json")
    junk = json.loads(junk_path.read_text())

    missing_docs = junk.get("missing_without_subsection", [])
    excluded_docs = junk.get("excluded_docs", [])

    source_files: set[str] = set()

    for doc in missing_docs + excluded_docs:
        sourcefile = doc.get("sourcefile")
        if sourcefile:
            source_files.add(str(sourcefile))
        content = doc.get("content") or ""
        source = extract_source_filename(content)
        if source:
            source_files.add(source)

    processed_dir = Path("data/legal-scraper/processed")
    upload_dir = processed_dir / "Upload"

    deleted = []
    missing = []

    for name in sorted(source_files):
        candidates = [upload_dir / name, processed_dir / name]
        found = False
        for path in candidates:
            if path.exists():
                path.unlink()
                deleted.append(str(path))
                found = True
        if not found:
            missing.append(name)

    print(f"Deleted files: {len(deleted)}")
    print(f"Missing files: {len(missing)}")

    deleted_path = Path("evals/results/junk_uploads_deleted.json")
    deleted_path.write_text(json.dumps({"deleted": deleted, "missing": missing}, indent=2))
    print(f"Report saved: {deleted_path}")


if __name__ == "__main__":
    main()
