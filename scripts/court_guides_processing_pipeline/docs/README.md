# Uploaded Court Guides — Migration & Documentation ✅

This folder documents the *uploaded* court guide JSON files (from `SOURCE_FILES/Upload`) and how they map to the repository outputs (processed + review JSONs and generated Markdown outputs). It contains:

- **5 Original PDFs** — Source documents from UK Judiciary
- **5 Processed JSONs** — Final court guide data ready for RAG/semantic search
- `mapping.json` — machine-readable mapping (upload → processed → markdown → scripts used).
- `CHECKLIST.md` — step-by-step checklist for validating and moving files.
- `SCRIPTS_USED.md` — short descriptions of the scripts involved and notes about scripts that require attention.

Purpose: collect the information you asked for so the uploaded assets can be safely moved to another folder or archived with minimal risk.

Next step: run the verification checklist in `CHECKLIST.md` and update paths referenced in `QUICK_START.md` or other docs if you move files.

## PDF to JSON Conversion Process

### Overview

The court guides started as PDF files from the UK Judiciary website. They were converted to JSON using a multi-step pipeline optimized for legal document processing.

### Solution Used: Docling + Custom Processing Scripts

**Core Technology:**

- **Docling**: Open-source document processing library (IBM) for extracting structured content from PDFs.
- **Pipeline**: OCR-enabled, table structure extraction, heading hierarchy detection, footnote capture.
- **Backend**: DoclingParseV4DocumentBackend for high-quality text extraction.

**Conversion Steps:**

1. **PDF Extraction** (`scripts/extract_court_guides.py`)
   - Input: PDF files (e.g., `14.341_JO_Commercial_Court_Guide_FINAL.pdf`)
   - Output: Raw Docling JSON with page-level metadata, text elements, tables, sections.
   - Settings: OCR=True, Tables=True, Cell Matching=True, No enrichments (code/formula).

1. **JSON Processing** (`scripts/process_court_guides.py`)
   - Input: Raw Docling JSON
   - Output: Simplified "processed.json" (content-focused) and "review.json" (detailed with metadata)
   - Features: Heading grouping, repeated heading merging, page span capture, TOC integration.

1. **Special Cases:**
   - **King's Bench**: Rebuilt from Markdown (`scripts/rebuild_kings_bench_from_md.py`) + annex stitching (`scripts/add_kings_bench_annexes.py`)
   - **TOC Integration**: Used for better section matching (where available)

### What Worked Well ✅

- **High Content Retention**: 90-100%+ character retention across guides (validated via `scripts/reconstruct_and_compare_all.py`)
- **Structured Output**: JSON format preserves hierarchy, page refs, and metadata for RAG/semantic search
- **Scalable**: Pipeline handles multiple guides with consistent results
- **Validation**: Comprehensive verification scripts ensure accuracy

### Known Issues & Solutions

- **Missing Content**: Some guides had 10-20% missing sections; mitigated by manual review and rebuild scripts
- **King's Bench Anomalies**: Required special Markdown-based rebuild due to parsing issues
- **Table/Appendix Handling**: Improved with cell matching, but some complex tables needed manual fixes

### Key Scripts & Commands

See `SCRIPTS_USED.md` for details. Example command:

```bash
python scripts/extract_court_guides.py --input "Court Guides" --output court_guides_parsed
python scripts/process_court_guides.py --input court_guides_parsed/guide.json --outdir court_guides_processed --court commercial
```

### Validation Results

- All guides passed title consistency checks
- Content coverage: Commercial (97%), Patents (90%), TCC (101%), Chancery (validated), King's Bench (rebuilt from MD)
- Ready for RAG applications with 698 total JSON items across 5 guides
