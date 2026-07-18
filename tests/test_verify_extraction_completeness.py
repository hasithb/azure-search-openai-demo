from scripts.court_guides_processing_pipeline.scripts.extract_court_guides_azure_di import GUIDE_METADATA


def test_canonical_capture_inventory_matches_completeness_contract():
    assert len(GUIDE_METADATA) == 8
    assert "Intellectual-Property-Enterprise-Court-IPEC-Guide-revised-November-2024.pdf" in GUIDE_METADATA