import json

from scripts.court_guides_processing_pipeline.scripts import (
    extract_court_guides_azure_di as extractor,
)


def test_canonical_capture_inventory_matches_completeness_contract():
    assert len(extractor.GUIDE_METADATA) == 8
    assert "Intellectual-Property-Enterprise-Court-IPEC-Guide-revised-November-2024.pdf" in extractor.GUIDE_METADATA


def test_process_pdf_reuses_hash_matched_persistent_cache(tmp_path, monkeypatch):
    pdf_path = tmp_path / "guide.pdf"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    pdf_path.write_bytes(b"%PDF-test")
    pdf_hash = extractor.sha256_file(pdf_path)
    processed = [{"id": "cached-document", "content": "cached"}]
    markdown = "# Guide\n\nCached markdown"
    (cache_dir / "guide_azure_di.md").parent.mkdir()
    (cache_dir / "guide_azure_di.md").write_text(markdown, encoding="utf-8")
    (cache_dir / "guide_processed.json").write_text(json.dumps(processed), encoding="utf-8")
    (cache_dir / "guide_azure_di.md.provenance.json").write_text(
        json.dumps({"pdf_sha256": pdf_hash}), encoding="utf-8"
    )
    (cache_dir / extractor.EXTRACTION_MANIFEST).write_text(
        json.dumps(
            {
                "guides": {
                    "guide.pdf": {
                        "pdf_sha256": pdf_hash,
                        "markdown_sha256": extractor.hashlib.sha256(markdown.encode()).hexdigest(),
                        "processed_json": "guide_processed.json",
                        "processed_json_sha256": extractor.sha256_file(cache_dir / "guide_processed.json"),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(extractor, "parse_with_azure_di", lambda _: (_ for _ in ()).throw(AssertionError("DI called")))

    assert extractor.process_pdf(str(pdf_path), str(output_dir), cache_dir=str(cache_dir)) == processed
    assert json.loads((output_dir / "guide_processed.json").read_text(encoding="utf-8")) == processed


def test_parse_with_azure_di_retries_transient_timeout(monkeypatch, tmp_path):
    pdf_path = tmp_path / "guide.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    attempts = 0

    class Poller:
        def result(self):
            return type("Result", (), {"pages": [1], "content": "markdown"})()

    class Client:
        def begin_analyze_document(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise extractor.HttpResponseError("timeout")
            return Poller()

    monkeypatch.setattr(extractor, "DefaultAzureCredential", lambda: object())
    monkeypatch.setattr(extractor, "DocumentIntelligenceClient", lambda **kwargs: Client())
    monkeypatch.setattr(extractor.time, "sleep", lambda _: None)

    assert extractor.parse_with_azure_di(str(pdf_path), max_retries=1) == "markdown"
    assert attempts == 2