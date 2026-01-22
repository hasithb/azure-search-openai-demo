# Migration Checklist — Uploaded Court Guides ✅

Follow these steps when you *move* uploaded court guides to another folder (or archive location):

1. Pre-move validation
   - [ ] Confirm `mapping.json` entries exist for each upload file.
   - [ ] Run `python -m scripts/validate_missing_is_in_titles.py` to catch missing-title inconsistencies.
   - [ ] Run `python -m scripts/validate_against_markdown.py` to ensure MD/JSON sync.

1. Run core verification scripts (in repo root):
   - `python scripts/reconstruct_and_compare_all.py` — ensure char/word coverage metrics remain similar.
   - `python scripts/compare_content.py --original court_guides_parsed/<file>.json --processed court_guides_processed/<file>.processed.json` for each guide (or use `compare_all_guides.sh`).

1. If any guide shows substantial discrepancies (<70% char coverage):
   - Re-run `process_court_guides.py` with `--log-level DEBUG` and `--toc <toc.json>` where available.
   - For King's Bench, re-run `scripts/rebuild_kings_bench_from_md.py` and `scripts/add_kings_bench_annexes.py`.

1. Update references before moving
   - Update `QUICK_START.md` if entries change.
   - Update any scripts that hardcode `SOURCE_FILES/Upload` or `SOURCE_FILES/03_Markdown_Output` paths.

1. Move files
   - Use a single git commit describing the move.
   - After moving, run `pytest` and the validation scripts above.

1. Post-move checks
   - Verify `SOURCE_FILES/03_Markdown_Output` still references any images (`img-*.jpeg`) next to the moved content.
   - Run `scripts/compare_content.py` on moved files to ensure no path breakage.

Notes

- If extraction/processing fail for a file, check `court_guides_parsed/_extraction_log.json` or capture logs by running scripts with `--log-level DEBUG`.
- King's Bench has bespoke rebuild steps; do not skip those when verifying.
