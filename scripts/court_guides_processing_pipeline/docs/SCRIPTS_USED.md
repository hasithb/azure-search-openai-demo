# Scripts Used — Court Guide Processing (summary)

This is a short reference to the scripts involved in processing and verifying the court guides, and what to check if they fail.

- `scripts/extract_court_guides.py` 🔧
  - Purpose: extract PDFs to parsed JSON using Docling pipeline.
  - Checks: `_extraction_log.json` in `court_guides_parsed/`; run with `--log-level DEBUG` for details.

- `scripts/process_court_guides.py` 🔧
  - Purpose: convert Docling JSON → processed + review JSON files.
  - Important flags: `--toc <toc.json>`, `--export-text-lines`, `--jsonl`.
  - If output is wrong: re-run with `--no-merge` or `--no-group-major` to diagnose grouping errors.

- `scripts/reconstruct_and_compare_all.py` 📊
  - Purpose: reconstruct original documents from parsed JSON and MD and compare processed outputs.
  - Use to generate coverage metrics (char ratio, word coverage, sequence coverage).

- `scripts/rebuild_kings_bench_from_md.py` & `scripts/add_kings_bench_annexes.py` ✳️
  - Purpose: special handling for King's Bench (rebuilt from Markdown and annex stitching).
  - Notes: King's Bench uses Markdown as primary source; these must be re-run if MD changes.

- `scripts/fix_content_join.py`, `scripts/fix_kings_bench_structure.py` 🛠
  - Purpose: post-processing fixes for known join and hierarchy issues.
  - They touch specific processed JSONs (see script headers for file lists).

- Verification & comparison:
  - `scripts/compare_content.py`, `scripts/compare_toc_similarity.py`, `scripts/compare_all_guides.sh` — use to verify content/TOC similarity.
  - `scripts/validate_missing_is_in_titles.py` and `scripts/validate_against_markdown.py` — run these after any move.

If a script fails:

- Re-run with `--log-level DEBUG` and capture stdout/stderr into a log file.
- Check for malformed input JSONs in `court_guides_parsed` or missing TOC `*.toc.json` files.
- For rebuild scripts, ensure the Markdown source exists in `SOURCE_FILES/03_Markdown_Output`.
