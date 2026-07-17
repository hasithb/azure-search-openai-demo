#!/usr/bin/env python3
"""
Update CPR/PD documents in index v3.

Addresses all items from docs/fresh_vs_index_action_list.md:
  - Section A: 14 docs with content drift (re-scrape & update)
  - Section B: 1 scraper investigation (Part 82)
  - Section C: 25 docs needing manual review (re-scrape & compare)
  - Section D: 2 missing docs (PD 5C, Devolution Issues Welsh)
  - Bonus: Debt Claims PAP (PDF-only on justice.gov.uk)

Usage:
    # Dry run — scrape + compare, no index changes:
    python scripts/update_cpr_index_v3.py --dry-run

    # Live update:
    python scripts/update_cpr_index_v3.py

    # Only specific sections:
    python scripts/update_cpr_index_v3.py --sections A D

    # Verbose diff output:
    python scripts/update_cpr_index_v3.py --dry-run --verbose
"""
import os
import sys
import json
import re
import io
import time
import logging
import argparse
import hashlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import pypdf
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "app" / "backend"
SCRAPER_DIR = SCRIPT_DIR / "legal-scraper"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRAPER_DIR))

from load_azd_env import load_azd_env
from customizations.subsection_extractor import SubsectionExtractor
from token_chunker import LegalDocumentChunker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("update_cpr_index_v3")
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Action list from docs/fresh_vs_index_action_list.md
# Each entry: (sourcefile, azure_id, url, section)
# ---------------------------------------------------------------------------
ACTION_LIST = [
    # ── Section A: 14 docs with content drift ──
    {"sourcefile": "Part 44", "azure_id": "Part_44___General_Rules_about_Costs", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-44-general-rules-about-costs", "section": "A"},
    {"sourcefile": "Part 46", "azure_id": "Part_46___Costs_special_cases", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-46-costs-special-cases", "section": "A"},
    {"sourcefile": "Part 5", "azure_id": "Part_5___Court_Documents", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part05", "section": "A"},
    {"sourcefile": "Part 77", "azure_id": "Part_77___Provision_in_Support_of_Criminal_Justice", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part77", "section": "A"},
    {"sourcefile": "Part 8", "azure_id": "Part_8___Alternative_Procedure_for_Claims", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part08", "section": "A"},
    {"sourcefile": "Practice Direction 16", "azure_id": "Practice_Direction_16", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part16/pd_part16", "section": "A"},
    {"sourcefile": "Practice Direction 27A", "azure_id": "Practice_Direction_27A___Small_Claims_Track", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part27/pd_part27", "section": "A"},
    {"sourcefile": "Practice Direction 41A", "azure_id": "Practice_Direction_41A___Provisional_Damages", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part41/pd_part41a", "section": "A"},
    {"sourcefile": "Practice Direction 51ZH", "azure_id": "Practice_Direction_51ZH___Access_to_Public_Domain_Documents", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/practice-direction-51zh-access-to-public-domain-documents", "section": "A"},
    {"sourcefile": "Practice Direction 54D", "azure_id": "Practice_Direction_54D___Planning_Court_Claims", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part54/practice-direction-54e-planning-court-claims", "section": "A"},
    {"sourcefile": "Practice Direction 57B", "azure_id": "Practice_Direction_57B___Proceedings_under_the_Presumption_of_Death_Act_2013", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part57/practice-direction-57b-proceedings-under-the-presumption-of-death-act-2013", "section": "A"},
    {"sourcefile": "Practice Direction 62", "azure_id": "Practice_Direction_62", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part62/pd_part62", "section": "A"},
    {"sourcefile": "Practice Direction 64B", "azure_id": "Practice_Direction_64B___Applications_to_the_Court_for_Directions_by_Trustees_in_Relation_to_the_Adm", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part64/pd_part64b", "section": "A"},
    {"sourcefile": "Practice Direction 74A", "azure_id": "Practice_Direction_74A___Enforcement_of_Judgments_in_different_Jurisdictions", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part74/pd_part74a", "section": "A"},
    # ── Section B: 1 scraper investigation ──
    {"sourcefile": "Part 82", "azure_id": "Part_82___Closed_material_procedure_chunk_001", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-82-closed-material-procedure", "section": "B"},
    # ── Section C: 25 docs needing manual review ──
    {"sourcefile": "Part 2", "azure_id": "Part_2___Application_and_Interpretation_of_the_Rules", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part02", "section": "C"},
    {"sourcefile": "Part 30", "azure_id": "Part_30___Transfer", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part30", "section": "C"},
    {"sourcefile": "Part 52", "azure_id": "Part_52___Appeals_chunk_001", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part52", "section": "C"},
    {"sourcefile": "Part 53", "azure_id": "Part_53___Media_and_Communications_Claims", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part53", "section": "C"},
    {"sourcefile": "Part 62", "azure_id": "Part_62___Arbitration_Claims", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part62", "section": "C"},
    {"sourcefile": "Part 65", "azure_id": "Part_65___Proceedings_Relating_to_Anti-Social_Behaviour_and_Harassment_chunk_001", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part65", "section": "C"},
    {"sourcefile": "Part 74", "azure_id": "Part_74___Enforcement_of_Judgments_in_Different_Jurisdictions", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part74", "section": "C"},
    {"sourcefile": "Part 79", "azure_id": "Part_79___Proceedings_under_the_counter-terrorism_act_2008__part_1_of_the_terrorist_asset-freezing_e", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part79", "section": "C"},
    {"sourcefile": "Part 80", "azure_id": "Part_80___Proceedings_under_the_Terrorism_Prevention_and_Investigation_Measures_Act_2011", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part80", "section": "C"},
    {"sourcefile": "Practice Direction 30", "azure_id": "Practice_Direction_30", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part30/pd_part30", "section": "C"},
    {"sourcefile": "Practice Direction 31A", "azure_id": "Practice_Direction_31A___Disclosure_and_Inspection", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part31/pd_part31a", "section": "C"},
    {"sourcefile": "Practice Direction 31B", "azure_id": "Practice_Direction_31B___Disclosure_of_Electronic_Documents", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part31/pd_part31b", "section": "C"},
    {"sourcefile": "Practice Direction 31C", "azure_id": "Practice_Direction_31C___Disclosure_and_inspection_in_relation_to_competition_claims", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part31/practice-direction-31c-disclosure-and-inspection-in-relation-to-competition-claims", "section": "C"},
    {"sourcefile": "Practice Direction 34A", "azure_id": "Practice_Direction_34A___Depositions_and_Court_Attendance_by_Witnesses", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part34/pd_part34a", "section": "C"},
    {"sourcefile": "Practice Direction 40B", "azure_id": "Practice_Direction_40B___Judgments___Orders", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part40/pd_part40b", "section": "C"},
    {"sourcefile": "Practice Direction 46", "azure_id": "Practice_Direction_46___Costs_Special_Cases", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-46-costs-special-cases/practice-direction-46-costs-special-cases", "section": "C"},
    {"sourcefile": "Practice Direction 51R", "azure_id": "Practice_Direction_51R___Online_Civil_Money_Claims_Pilot_chunk_001", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/practice-direction-51r-online-court-pilot", "section": "C"},
    {"sourcefile": "Practice Direction 52C", "azure_id": "Practice_Direction_52C___Appeals_to_the_Court_of_Appeal", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part52/practice-direction-52c-appeals-to-the-court-of-appeal", "section": "C"},
    {"sourcefile": "Practice Direction 52D", "azure_id": "Practice_Direction_52D___Statutory_appeals_and_appeals_subject_to_special_provision_chunk_001", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part52/practice-direction-52d-statutory-appeals-and-appeals-subject-to-special-provision", "section": "C"},
    {"sourcefile": "Practice Direction 52E", "azure_id": "Practice_Direction_52E___Appeals_by_way_of_case_stated", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part52/practice-direction-52e-appeals-by-way-of-case-stated", "section": "C"},
    {"sourcefile": "Practice Direction 57", "azure_id": "Practice_Direction_57", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part57/pd_part57", "section": "C"},
    {"sourcefile": "Practice Direction 63", "azure_id": "Practice_Direction_63", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part63/pd_part63", "section": "C"},
    {"sourcefile": "Practice Direction 6B", "azure_id": "Practice_Direction_6B___Service_out_of_the_Jurisdiction", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part06/pd_part06b", "section": "C"},
    {"sourcefile": "Practice Direction 77", "azure_id": "Practice_Direction_77", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part77/pd_part77", "section": "C"},
    {"sourcefile": "Practice Direction 7B", "azure_id": "Practice_Direction_7B-_Production_Centre", "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part07/pd_part07c", "section": "C"},
    # ── Section D: 2 missing docs ──
    {"sourcefile": "Devolution Issues and Crown Office Applications in Wales (Welsh)", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/devolution_issues_welsh", "section": "D"},
    {"sourcefile": "PRACTICE DIRECTION 5C", "azure_id": None, "url": "https://www.justice.gov.uk/practice-direction-5c-ce-file-electronic-filing-and-case-management-system", "section": "D"},
    # ── Bonus: Debt Claims PAP (PDF) ──
    {"sourcefile": "Pre-Action Protocol for Debt Claims", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/pdf/protocols/debt-pap.pdf", "section": "DEBT"},
    {"sourcefile": "Practice Direction 1A", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part01/practice-direction-1a-participation-of-vulnerable-parties-or-witnesses", "section": "E"},
    {"sourcefile": "Practice Direction 3F", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/practice-direction-3g-requests-for-the-appointment-of-an-advocate-to-the-court", "section": "E"},
    {"sourcefile": "Practice Direction 17", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part17/pd_part17", "section": "E"},
    {"sourcefile": "Practice Direction 18", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part18/pd_part18", "section": "E"},
    {"sourcefile": "Practice Direction 20", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part20/pd_part20", "section": "E"},
    {"sourcefile": "Practice Direction 22", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part22/pd_part22", "section": "E"},
    {"sourcefile": "Practice Direction 29", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part29/pd_part29", "section": "E"},
    {"sourcefile": "Practice Direction 32", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part32/pd_part32", "section": "E"},
    {"sourcefile": "Practice Direction 35", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part35/pd_part35", "section": "E"},
    {"sourcefile": "Practice Direction 37", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part37/pd_part37", "section": "E"},
    {"sourcefile": "Practice Direction 42", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part42/pd_part42", "section": "E"},
    {"sourcefile": "Practice Direction 49B", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part49/pd_part49b", "section": "E"},
    {"sourcefile": "Practice Direction 49C", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part49/practice-direction-49c-consumer-credit-act-2006-unfair-relationships", "section": "E"},
    {"sourcefile": "Practice Direction 49D", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part49/practice-direction-49d-claims-for-the-recovery-of-taxes-and-duties", "section": "E"},
    {"sourcefile": "Practice Direction 49E", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part49/practice-direction-49e-alternative-procedure-for-claims", "section": "E"},
    {"sourcefile": "Practice Direction 49F", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part49/practice-direction-49f-pre-action-protocol-for-low-value-personal-injury-claims-in-road-traffic-accidents-and-low-value-personal-injury-employers-liability-and-public-liability-claims-stage-3-procedure", "section": "E"},
    {"sourcefile": "Practice Direction 51ZD", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/practice-direction-51zd-pilot-scheme-for-capping-costs-in-patent-cases-in-the-shorter-trial-scheme", "section": "E"},
    {"sourcefile": "Practice Direction 51ZG1", "azure_id": None, "url": "https://www.justice.gov.uk/practice-direction-51zg1-pilot-scheme-for-cost-budgeting-in-certain-business-and-property-courts-and-certain-business-and-property-work-in-the-county-court", "section": "E"},
    {"sourcefile": "Practice Direction 51ZG2", "azure_id": None, "url": "https://www.justice.gov.uk/practice-direction-51zg2-pilot-scheme-for-costs-budgeting-in-certain-claims-with-a-value-of-less-than-1-million", "section": "E"},
    {"sourcefile": "Practice Direction 51ZG3", "azure_id": None, "url": "https://www.justice.gov.uk/practice-direction-51zg3-pilot-scheme-for-certain-high-court-qualified-one-way-costs-shifting-qocs-cases", "section": "E"},
    {"sourcefile": "Practice Direction 53A", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part53/pd_part53", "section": "E"},
    {"sourcefile": "Practice Direction 53B", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part53/practice-direction-53b-media-and-communications-claims", "section": "E"},
    {"sourcefile": "Practice Direction 54B", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part54/pd_part54c", "section": "E"},
    {"sourcefile": "Practice Direction 54E", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part54/practice-direction-54e-environmental-review-claims", "section": "E"},
    {"sourcefile": "Practice Direction 56", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part56/pd_part56", "section": "E"},
    {"sourcefile": "Practice Direction 56A", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part56/practice-direction-56a-renting-homes-wales-claims", "section": "E"},
    {"sourcefile": "Practice Direction 57C", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/practice-direction-57c-proceedings-under-the-guardianship-missing-persons-act-2017", "section": "E"},
    {"sourcefile": "Practice Direction 57AA", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/practice-direction-business-and-property-courts", "section": "E"},
    {"sourcefile": "Practice Direction 57AB", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/practice-direction-57ab-shorter-and-flexible-trials-schemes", "section": "E"},
    {"sourcefile": "Practice Direction 57AC", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-57a-business-and-property-courts/practice-direction-57ac-trial-witness-statements-in-the-business-and-property-courts", "section": "E"},
    {"sourcefile": "Practice Direction 58", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part58/pd_part58", "section": "E"},
    {"sourcefile": "Practice Direction 59", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part59/pd_part59", "section": "E"},
    {"sourcefile": "Practice Direction 60", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part60/pd_part60", "section": "E"},
    {"sourcefile": "Practice Direction 61", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part61/pd_part61", "section": "E"},
    {"sourcefile": "Practice Direction 63AA", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/financial-list/practice-direction-63aa-financial-list", "section": "E"},
    {"sourcefile": "Practice Direction 65", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part65/pd_part65", "section": "E"},
    {"sourcefile": "Practice Direction 66", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part66/pd_part66", "section": "E"},
    {"sourcefile": "Practice Direction 67", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part67/pd_part67", "section": "E"},
    {"sourcefile": "Practice Direction 69", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part69/pd_part69", "section": "E"},
    {"sourcefile": "Practice Direction 70A", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part70/pd_part70", "section": "E"},
    {"sourcefile": "Practice Direction 70B", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part70/practice-direction-70b-debt-respite-scheme-under-the-financial-guidance-and-claims-act-2018", "section": "E"},
    {"sourcefile": "Practice Direction 71", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part71/pd_part71", "section": "E"},
    {"sourcefile": "Practice Direction 72", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part72/pd_part72", "section": "E"},
    {"sourcefile": "Practice Direction 73", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part73/pd_part73", "section": "E"},
    {"sourcefile": "Practice Direction 75", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part75/pd_part75", "section": "E"},
    {"sourcefile": "Part 83", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-83-writs-and-warrants-general-provisions", "section": "E"},
    {"sourcefile": "Practice Direction 83", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-83-writs-and-warrants-general-provisions/37-practice-direction-83-writs-and-warrants-general-provisions", "section": "E"},
    {"sourcefile": "Part 84", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-84enforcement-by-taking-control-of-goods", "section": "E"},
    {"sourcefile": "Practice Direction 84", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-84enforcement-by-taking-control-of-goods/practice-direction-84-enforcement-by-taking-control-of-goods", "section": "E"},
    {"sourcefile": "Part 85", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-85-claims-on-controlled-goods-and-executed-goods", "section": "E"},
    {"sourcefile": "Part 86", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-86-stakeholder-claims-and-applications", "section": "E"},
    {"sourcefile": "Part 87", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-87-applications-for-writ-of-habeas-corpus", "section": "E"},
    {"sourcefile": "Part 88", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-88-proceedings-under-the-counter-terrorism-and-security-act-2015", "section": "E"},
    {"sourcefile": "Part 89", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/art-89-attachment-of-earnings", "section": "E"},
    {"sourcefile": "Practice Direction – Pre-Action Conduct and Protocols", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/pd_pre-action_conduct", "section": "E"},
    {"sourcefile": "Practice Direction – Competition Law", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/competitionlaw_pd", "section": "E"},
    {"sourcefile": "Practice Direction – Civil Recovery Proceedings", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/civilrecovery_pd", "section": "E"},
    {"sourcefile": "Practice Direction – Enterprise Act 2002 Warrant", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/appforwarrant_comp_act2002", "section": "E"},
    {"sourcefile": "Practice Direction – Proceedings under Enactments Relating to Equality", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/proceedings_under_enactments_equality", "section": "E"},
    {"sourcefile": "Practice Direction – County Court Closures", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/county_court_closures", "section": "E"},
    {"sourcefile": "Practice Direction – Solicitors Negligence in Right to Buy Cases", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/practice-direction-solicitors-negligence-in-right-to-buy-cases", "section": "E"},
    {"sourcefile": "Practice Direction – EU and EEA EFTA Citizens Rights", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/practice-direction-claims-relating-to-eu-and-eea-efta-citizens-rights-under-part-2-of-the-withdrawal-agreement-and-part-2-of-the-eea-efta-separation-agreement", "section": "E"},
    {"sourcefile": "Pre-Action Protocol for Dilapidations", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/protocol/pre-action-protocol-for-claims-for-damages-in-relation-to-the-physical-state-of-commercial-property-at-termination-of-a-tenancy-the-dilapidations-protocol", "section": "E"},
    {"sourcefile": "Pre-Action Protocol for Low Value Personal Injury EL PL Claims", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/protocol/pre-action-protocol-for-low-value-personal-injury-employers-liability-and-public-liability-claims", "section": "E"},
    {"sourcefile": "Pre-Action Protocol for RTA Small Claims", "azure_id": None, "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/protocol/pre-action-protocol-for-personal-injury-claims-below-the-small-claims-limit-in-road-traffic-accidents-the-rta-small-claims-protocol", "section": "E"},
]

CATEGORY = "Civil Procedure Rules and Practice Directions"
PROTOCOL_INDEX_URL = "https://www.justice.gov.uk/courts/procedure-rules/civil/protocol"

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
def make_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    return sess


# ---------------------------------------------------------------------------
# Scraping (adapted from scrape_cpr.py)
# ---------------------------------------------------------------------------
def fetch_soup(session: requests.Session, url: str) -> Optional[BeautifulSoup]:
    """Fetch URL and return BeautifulSoup. Handles PDF responses."""
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if url.lower().endswith(".pdf") or "application/pdf" in content_type:
            logger.info("  PDF detected for %s", url)
            pdf_file = io.BytesIO(response.content)
            reader = pypdf.PdfReader(pdf_file)
            texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
            full = "\n\n".join(texts)
            paras = "".join(f"<p>{p.strip()}</p>" for p in full.split("\n\n") if p.strip())
            slug = url.split("/")[-1].replace(".pdf", "")
            html = f"<html><body><article><h1>{slug}</h1><div>{paras}</div></article></body></html>"
            return BeautifulSoup(html, "html.parser")

        return BeautifulSoup(response.content, "html.parser")
    except Exception as e:
        logger.error("  Failed to fetch %s: %s", url, e)
        return None


def clean_html(soup: BeautifulSoup) -> BeautifulSoup:
    noise = [
        "script", "style", "nav", "header", "footer",
        ".tools", ".back-to-top", ".related-items",
        "#cookie-banner", ".global-cookie-message",
        ".breadcrumb", "#breadcrumb", ".breadcrumbs", ".you-are-here",
    ]
    for sel in noise:
        for el in soup.select(sel):
            el.decompose()
    for el in soup.find_all(string=re.compile(r"^Back to top")):
        if el.parent:
            el.parent.decompose()
    return soup


def scrape_page(
    session: requests.Session,
    action_entry: dict | str,
    prefetched_result: Optional[tuple] = None,
) -> Optional[dict]:
    """
    Scrape a single justice.gov.uk page.
    Returns {"content": str, "title": str, "updated": str} or None.
    """
    if isinstance(action_entry, dict):
        url = action_entry["url"]
    else:
        url = action_entry

    if prefetched_result is not None:
        soup, final_url, redirect_count = prefetched_result
    else:
        soup = fetch_soup(session, url)
        final_url = url
        redirect_count = 0
    if not soup:
        return None

    content_div = (
        soup.find("div", class_="article-content")
        or soup.find("div", id="content")
        or soup.find("main")
        or soup.find("body")
    )
    if not content_div:
        logger.warning("  No content found for %s", url)
        return None

    content_div = clean_html(content_div)

    # Extract updated date
    updated = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    meta_date = soup.find("meta", attrs={"name": "DC.date.modified"})
    if meta_date and meta_date.get("content"):
        try:
            d = datetime.strptime(meta_date["content"], "%Y-%m-%d")
            updated = d.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass

    # DOM traversal with breadcrumb context
    context_part = ""
    context_rule = ""
    paragraphs = []

    elements = content_div.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "li", "table"])
    for elem in elements:
        if not elem.parent:
            continue
        if elem.name == "div" and (elem.find("p") or elem.find("h1") or elem.find("h2")):
            continue

        if elem.name == "table":
            rows = elem.find_all("tr")
            if rows:
                bc = ""
                if context_part or context_rule:
                    parts = [c for c in [context_part, context_rule] if c]
                    bc = f"[{' > '.join(parts)}] "
                for row in rows:
                    cells = row.find_all(["th", "td"])
                    cell_texts = [c.get_text(" ", strip=True) for c in cells]
                    row_text = " | ".join(cell_texts)
                    if row_text.strip():
                        paragraphs.append(f"{bc}{row_text}")
            continue

        text_element = elem
        if elem.name == "li" and elem.find_all(["ol", "ul"]):
            text_element = deepcopy(elem)
            for nested_list in text_element.find_all(["ol", "ul"]):
                nested_list.decompose()
        text = text_element.get_text(" ", strip=True)
        if not text:
            continue

        if elem.name == "h1" or (
            elem.name == "p"
            and re.match(r"^(PART|PRACTICE\s+DIRECTIONS?)\b", text, re.IGNORECASE)
        ):
            context_part = text
            context_rule = ""
            paragraphs.append(f"# {text}")
            continue

        is_generic = re.match(
            r"^(DATA PROTECTION|MISUSE OF PRIVATE|HARASSMENT|DEFAMATION|INTRODUCTION"
            r"|OBJECTIVES|PROPORTIONALITY|EXPERTS|SETTLEMENT|LIMITATION)",
            text, re.IGNORECASE,
        )
        if elem.name in ["h2", "h3", "h4", "h5", "h6"]:
            if len(text) < 100:
                context_rule = text
            paragraphs.append(f"## {text}")
            continue

        if (
            elem.name == "p"
            and (
                re.match(r"^(Rule|Para\.?|Paragraph)\s*\d+|^\d+(\.\d+)?", text, re.IGNORECASE)
                or is_generic
            )
            and len(text) < 100
        ):
            context_rule = text
            paragraphs.append(f"## {text}")
            continue

        if elem.name in ["p", "li"]:
            bc = ""
            if context_part or context_rule:
                parts = [c for c in [context_part, context_rule] if c]
                bc = f"[{' > '.join(parts)}] "
            paragraphs.append(f"{bc}{text}")

    full_content = "\n\n".join(paragraphs)
    full_content = re.sub(r"\n{3,}", "\n\n", full_content)

    if len(full_content) < 50:
        logger.warning("  Content too short (%d chars) for %s", len(full_content), url)
        return None

    # Extract title from H1
    title = ""
    h1_match = re.search(r"^#\s+(.+)", full_content, re.MULTILINE)
    if h1_match:
        title = h1_match.group(1).strip()

    return {
        "content": full_content,
        "title": title,
        "updated": updated,
        "_final_url": final_url,
        "_redirect_count": redirect_count,
    }


# ---------------------------------------------------------------------------
# Document mapping (same schema as upload_with_embeddings.py)
# ---------------------------------------------------------------------------
def sanitize_id(doc_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-=]", "_", doc_id)
    s = re.sub(r"_{2,}", "___", s)
    return s.strip("_")


def generate_id_from_content(title: str, content: str, fallback: str) -> str:
    """Generate ID from content headers (same logic as scrape_cpr.py)."""
    # H1 Practice Direction
    m = re.search(r"^#\s+(PRACTICE\s+DIRECTION\s+\d+[A-Z]*)\s*[-–]\s*([^\n]+)", content, re.MULTILINE | re.IGNORECASE)
    if m:
        return f"{m.group(1).title()} – {m.group(2).strip().title()}"
    # H1 Part
    m = re.search(r"^#\s+(PART\s+\d+[A-Z]?)\s*[-–]\s*([^\n]+)", content, re.MULTILINE)
    if m:
        return f"{m.group(1).title()} – {m.group(2).strip().title()}"
    # Fallback text PD
    m = re.search(r"^(PRACTICE\s+DIRECTION\s+\d+[A-Z]*)\s*[-–]\s*([^\n]+)", content, re.MULTILINE | re.IGNORECASE)
    if m:
        return f"{m.group(1).title()} – {m.group(2).strip().title()}"
    # Fallback text Part
    m = re.search(r"^(PART\s+\d+[A-Z]?)\s*[-–]\s*([^\n]+)", content, re.MULTILINE)
    if m:
        return f"{m.group(1).title()} – {m.group(2).strip().title()}"
    # Use title
    if title:
        t = title.strip().replace(" - ", " – ")
        return t
    return fallback


def has_existing_header(text: str) -> bool:
    if not text:
        return False
    head = [line.strip() for line in text.splitlines()[:6] if line.strip()]
    return any(
        line.startswith(("SOURCE:", "SOURCEPAGE:", "SECTION:"))
        or (line.startswith("[") and ">" in line)
        for line in head
    )


def extract_subsection_from_sourcepage(value: str) -> str:
    if not value:
        return ""
    for pat in [
        r"\b([A-Z]\.\d+(?:\.\d+)?)\b",
        r"\b([A-Z]\d+\.\d+(?:\.\d+)?)\b",
        r"\b(\d+\.\d+(?:\.\d+)?)\b",
        r"\b([A-Z]\d+)\b",
    ]:
        m = re.search(pat, value, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def extract_parent_section(value: str) -> str:
    if not value:
        return ""
    raw = value.strip()
    first_seg = raw.split(",", 1)[0].strip()
    if re.match(r"^[A-Z]\.", first_seg) or re.match(r"^(Section|Appendix|Part|Practice Direction)\b", first_seg, re.IGNORECASE):
        return first_seg
    for pat in [
        r"\b(Practice Direction\s+[0-9A-Z]+)\b",
        r"\b(Part\s+\d+[A-Z]?)\b",
        r"\b(Section\s+\d+)\b",
        r"\b(Appendix\s+[A-Z])\b",
    ]:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def build_index_docs(action_entry: dict, scraped: dict) -> list[dict]:
    """Build index-ready documents from a scraped page. May produce multiple chunks."""
    content = scraped["content"]
    title = scraped["title"] or action_entry["sourcefile"]
    updated = scraped["updated"]
    url = action_entry["url"]
    sourcefile = action_entry["sourcefile"]

    # Generate document ID
    fallback_id = sanitize_id(sourcefile)
    doc_id = generate_id_from_content(title, content, fallback_id)

    # ACTION_LIST sourcefile is the canonical identifier; keep it unchanged.
    sf = sourcefile

    # Chunk if needed
    chunker = LegalDocumentChunker(max_tokens=8000, overlap_tokens=200)
    chunks = chunker.chunk_legal_document(content, doc_id, title)

    docs = []
    for chunk in chunks:
        chunk_text = chunk["text"]
        idx = chunk["chunk_index"]
        total = chunk["total_chunks"]

        if total > 1:
            chunk_id = f"{doc_id}_chunk_{idx:03d}"
        else:
            chunk_id = doc_id

        # Subsection extraction
        extracted_sub = SubsectionExtractor.extract_first_subsection(chunk_text)
        all_subs = list(SubsectionExtractor.extract_all_subsections(chunk_text))
        derived_sub = extract_subsection_from_sourcepage(title)
        parent = extract_parent_section(title)
        subsection_id = extracted_sub or derived_sub or parent or ""
        subsections = list(all_subs)
        if subsection_id and subsection_id not in subsections:
            subsections.insert(0, subsection_id)

        # Inject header
        final_content = chunk_text
        if not has_existing_header(final_content):
            hdr = []
            if sf:
                hdr.append(f"SOURCE: {sf}")
            if title:
                hdr.append(f"SOURCEPAGE: {title}")
            hdr.append(f"CATEGORY: {CATEGORY}")
            if parent and parent != subsection_id:
                hdr.append(f"SECTION: {parent}")
            if subsection_id:
                hdr.append(f"## {subsection_id}")
            if hdr:
                final_content = "\n".join(hdr) + "\n\n" + final_content

        docs.append({
            "id": sanitize_id(chunk_id),
            "content": final_content,
            "embedding": [],
            "category": CATEGORY,
            "sourcepage": title or sourcefile,
            "sourcefile": sf,
            "storageUrl": url,
            "oids": ["all"],
            "groups": ["all", "36094ff3-5c6d-49ef-b385-fa37118527e3"],
            "parent_id": sanitize_id(doc_id),
            "subsection_id": subsection_id,
            "subsections": subsections,
            "updated": updated,
        })

    return docs


# ---------------------------------------------------------------------------
# Debt Claims PAP URL discovery
# ---------------------------------------------------------------------------
def discover_debt_claims_url(session: requests.Session) -> Optional[str]:
    """Find the Debt Claims PAP URL from the protocol index page."""
    logger.info("Discovering Debt Claims PAP URL from %s", PROTOCOL_INDEX_URL)
    soup = fetch_soup(session, PROTOCOL_INDEX_URL)
    if not soup:
        return None

    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        href = a["href"]
        if "debt" in text and ("claim" in text or "protocol" in text):
            full_url = urljoin(PROTOCOL_INDEX_URL, href)
            logger.info("  Found Debt Claims PAP URL: %s", full_url)
            return full_url

    logger.warning("  Could not find Debt Claims PAP link on protocol page")
    return None


# ---------------------------------------------------------------------------
# Azure clients (same as upload_court_guides_v3.py)
# ---------------------------------------------------------------------------
def get_search_client(endpoint: str, index: str):
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    return SearchClient(endpoint=endpoint, index_name=index, credential=DefaultAzureCredential())


def get_openai_client():
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    service = os.environ.get("AZURE_OPENAI_SERVICE", "")
    if not service:
        raise RuntimeError("AZURE_OPENAI_SERVICE not set")
    ep = f"https://{service}.openai.azure.com" if not service.startswith("https://") else service

    key = os.environ.get("AZURE_OPENAI_KEY", "")
    if key:
        return AzureOpenAI(api_key=key, api_version="2023-05-15", azure_endpoint=ep, max_retries=3, timeout=120.0)

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(azure_ad_token_provider=token_provider, api_version="2023-05-15", azure_endpoint=ep, max_retries=3, timeout=120.0)


# ---------------------------------------------------------------------------
# Index operations
# ---------------------------------------------------------------------------
def query_existing_by_sourcefile(client, sourcefile: str) -> list[str]:
    """Get all doc IDs in the index for a given sourcefile."""
    ids = []
    safe_sf = sourcefile.replace("'", "''")
    try:
        results = client.search(
            search_text="*",
            filter=f"sourcefile eq '{safe_sf}'",
            select=["id"],
            top=5000,
        )
        for doc in results:
            ids.append(doc["id"])
    except Exception as e:
        logger.error("  Error querying sourcefile '%s': %s", sourcefile, e)
    return ids


def find_existing_docs(client, entry: dict, new_docs: list[dict]) -> list[str]:
    """Robustly find all existing docs in the index for this action entry.
    Uses multiple strategies: sourcefile query, azure_id key lookup, and
    sourcefile derived from new docs."""
    found_ids: set[str] = set()

    # Strategy 1: Query by action list sourcefile
    ids1 = query_existing_by_sourcefile(client, entry["sourcefile"])
    found_ids.update(ids1)

    # Strategy 2: Query by derived sourcefile from new docs (may differ)
    for nd in new_docs:
        if nd["sourcefile"] != entry["sourcefile"]:
            ids2 = query_existing_by_sourcefile(client, nd["sourcefile"])
            found_ids.update(ids2)
            break

    # Strategy 3: Direct key lookup using known azure_id
    azure_id = entry.get("azure_id")
    if azure_id and azure_id not in found_ids:
        try:
            doc = client.get_document(key=azure_id, selected_fields=["id", "sourcefile"])
            found_ids.add(doc["id"])
            # Query by the ACTUAL sourcefile from the index
            actual_sf = doc.get("sourcefile", "")
            if actual_sf:
                more = query_existing_by_sourcefile(client, actual_sf)
                found_ids.update(more)
        except Exception:
            pass  # Doc doesn't exist by this key

    # Strategy 4: Look up by parent_id of new docs (handles chunk count changes)
    parent_ids_tried: set[str] = set()
    for nd in new_docs:
        pid = nd.get("parent_id", "")
        nid = nd["id"]
        # Try parent_id (the base ID without chunk suffix)
        if pid and pid not in found_ids and pid not in parent_ids_tried:
            parent_ids_tried.add(pid)
            try:
                doc = client.get_document(key=pid, selected_fields=["id", "sourcefile"])
                found_ids.add(doc["id"])
                actual_sf = doc.get("sourcefile", "")
                if actual_sf:
                    more = query_existing_by_sourcefile(client, actual_sf)
                    found_ids.update(more)
            except Exception:
                pass
        # Also try the exact new doc ID
        if nid not in found_ids:
            try:
                client.get_document(key=nid, selected_fields=["id"])
                found_ids.add(nid)
            except Exception:
                pass

    return sorted(found_ids)


def delete_docs(client, ids: list[str], dry_run: bool) -> int:
    if not ids or dry_run:
        return 0
    deleted = 0
    for i in range(0, len(ids), 1000):
        batch = [{"id": did} for did in ids[i:i + 1000]]
        try:
            client.delete_documents(documents=batch)
            deleted += len(batch)
        except Exception as e:
            logger.error("  Delete error: %s", e)
    return deleted


def generate_embeddings(docs: list[dict], dry_run: bool) -> list[dict]:
    if dry_run or not docs:
        return docs

    from openai import RateLimitError, APIConnectionError, APIError
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    client = get_openai_client()
    deployment = os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT", "text-embedding-3-large")

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
    )
    def _embed(texts):
        return client.embeddings.create(input=texts, model=deployment)

    batch_size = 50
    done = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        texts = [d["content"].replace("\n", " ")[:8000] for d in batch]
        resp = _embed(texts)
        for j, data in enumerate(resp.data):
            batch[j]["embedding"] = data.embedding
        done += len(batch)
        logger.info("  Embeddings: %d / %d", done, len(docs))
        if i + batch_size < len(docs):
            time.sleep(0.5)
    return docs


def upload_docs(client, docs: list[dict], dry_run: bool) -> int:
    if dry_run or not docs:
        return 0
    uploaded = 0
    for i in range(0, len(docs), 100):
        batch = docs[i:i + 100]
        try:
            results = client.upload_documents(documents=batch)
            for r in results:
                if r.succeeded:
                    uploaded += 1
                else:
                    logger.error("  Upload failed for %s: %s", r.key, r.error_message)
            if i + 100 < len(docs):
                time.sleep(0.3)
        except Exception as e:
            logger.error("  Upload batch error: %s", e)
    return uploaded


def compute_content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Update CPR/PD docs in index v3")
    parser.add_argument("--dry-run", action="store_true", help="Scrape and compare only, no index changes")
    parser.add_argument("--sections", nargs="+", default=["A", "B", "C", "D", "DEBT"],
                        help="Which sections to process (default: all). E.g. --sections A D DEBT")
    parser.add_argument("--verbose", action="store_true", help="Show content diffs")
    parser.add_argument("--save-json", action="store_true", help="Save scraped docs to /tmp/cpr_update_*.json")
    args = parser.parse_args()

    sections = {s.upper() for s in args.sections}
    logger.info("Sections to process: %s", ", ".join(sorted(sections)))

    load_azd_env()

    search_service = os.environ.get("AZURE_SEARCH_SERVICE", "")
    index_name = os.environ.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v3")
    if not search_service:
        logger.error("AZURE_SEARCH_SERVICE not set")
        return 1

    endpoint = f"https://{search_service}.search.windows.net"
    logger.info("Target: %s / %s", endpoint, index_name)
    if args.dry_run:
        logger.info("*** DRY RUN — no changes will be made ***")

    session = make_session()
    search_client = get_search_client(endpoint, index_name)

    # Filter action list by selected sections
    entries = [e for e in ACTION_LIST if e["section"] in sections]
    logger.info("Processing %d entries across sections %s", len(entries), ", ".join(sorted(sections)))

    # Resolve Debt Claims PAP URL if needed
    for entry in entries:
        if entry["url"] == "DISCOVER_FROM_PROTOCOL_PAGE":
            debt_url = discover_debt_claims_url(session)
            if debt_url:
                entry["url"] = debt_url
            else:
                logger.error("Cannot discover Debt Claims PAP URL — skipping")
                entries = [e for e in entries if e["url"] != "DISCOVER_FROM_PROTOCOL_PAGE"]

    # Track results
    results = {
        "scraped": 0,
        "failed_scrape": 0,
        "new_docs": 0,
        "updated_docs": 0,
        "unchanged_docs": 0,
        "deleted_old": 0,
        "uploaded": 0,
        "details": [],
    }

    all_docs_to_upload: list[dict] = []
    all_ids_to_delete: list[str] = []

    for i, entry in enumerate(entries, 1):
        sf = entry["sourcefile"]
        url = entry["url"]
        sec = entry["section"]
        logger.info("── [%d/%d] Section %s: %s", i, len(entries), sec, sf)
        logger.info("   URL: %s", url)

        # 1. Scrape
        scraped = scrape_page(session, url)
        if not scraped:
            results["failed_scrape"] += 1
            results["details"].append({"sourcefile": sf, "section": sec, "status": "SCRAPE_FAILED"})
            continue
        results["scraped"] += 1

        logger.info("   Scraped: %d chars, title='%s', updated=%s",
                     len(scraped["content"]), scraped["title"][:60], scraped["updated"])

        # 2. Build index documents
        new_docs = build_index_docs(entry, scraped)
        logger.info("   Produced %d document(s)", len(new_docs))

        # 3. Query existing docs (robust multi-strategy lookup)
        existing_ids = find_existing_docs(search_client, entry, new_docs)

        new_doc_ids = {d["id"] for d in new_docs}
        orphan_ids = [eid for eid in existing_ids if eid not in new_doc_ids]

        # 4. Compare content (for reporting)
        status = "NEW" if not existing_ids else "UPDATED"
        if existing_ids:
            # Quick content comparison using first existing doc
            try:
                old_doc = search_client.get_document(
                    key=existing_ids[0],
                    selected_fields=["id", "content"],
                )
                old_hash = compute_content_hash(old_doc.get("content", ""))
                new_hash = compute_content_hash(new_docs[0]["content"])
                if old_hash == new_hash and len(new_docs) == len(existing_ids):
                    status = "UNCHANGED"
                    if args.verbose:
                        logger.info("   Content hash match — no change detected")
            except Exception:
                pass

        detail = {
            "sourcefile": sf,
            "section": sec,
            "status": status,
            "new_chunks": len(new_docs),
            "old_chunks": len(existing_ids),
            "orphans_to_delete": len(orphan_ids),
        }
        results["details"].append(detail)

        if status == "NEW":
            results["new_docs"] += 1
        elif status == "UPDATED":
            results["updated_docs"] += 1
        else:
            results["unchanged_docs"] += 1

        logger.info("   Status: %s | old=%d chunks, new=%d chunks, orphans=%d",
                     status, len(existing_ids), len(new_docs), len(orphan_ids))

        if args.verbose and status == "UPDATED" and existing_ids:
            try:
                old_doc = search_client.get_document(key=existing_ids[0], selected_fields=["content"])
                old_lines = len((old_doc.get("content", "") or "").splitlines())
                new_lines = len(new_docs[0]["content"].splitlines())
                logger.info("   Content lines: old=%d, new=%d", old_lines, new_lines)
            except Exception:
                pass

        # Collect for batch processing
        if status != "UNCHANGED":
            all_docs_to_upload.extend(new_docs)
            all_ids_to_delete.extend(orphan_ids)
            # Also delete all existing docs (we'll re-upload fresh versions)
            for eid in existing_ids:
                if eid not in orphan_ids:
                    all_ids_to_delete.append(eid)

        # Brief delay between scrapes
        time.sleep(0.2)

    # Save JSON if requested
    if args.save_json and all_docs_to_upload:
        save_path = "/tmp/cpr_update_docs.json"
        # Strip embeddings for readable output
        save_docs = [{k: v for k, v in d.items() if k != "embedding"} for d in all_docs_to_upload]
        with open(save_path, "w") as f:
            json.dump(save_docs, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d docs to %s", len(save_docs), save_path)

    # ── Summary before action ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("  SCRAPE SUMMARY")
    logger.info("=" * 70)
    logger.info("  Scraped:     %d", results["scraped"])
    logger.info("  Failed:      %d", results["failed_scrape"])
    logger.info("  New:         %d", results["new_docs"])
    logger.info("  Updated:     %d", results["updated_docs"])
    logger.info("  Unchanged:   %d", results["unchanged_docs"])
    logger.info("  To delete:   %d old docs", len(all_ids_to_delete))
    logger.info("  To upload:   %d docs", len(all_docs_to_upload))
    logger.info("=" * 70)

    # Detailed breakdown
    for d in results["details"]:
        flag = "✓" if d["status"] in ("UPDATED", "NEW") else ("=" if d["status"] == "UNCHANGED" else "✗")
        logger.info("  %s [%s] %s: %s (old=%s, new=%s)",
                     flag, d["section"], d.get("sourcefile", "?"), d["status"],
                     d.get("old_chunks", "?"), d.get("new_chunks", "?"))

    if args.dry_run:
        logger.info("")
        logger.info("*** DRY RUN complete — no changes made ***")
        return 0

    if not all_docs_to_upload:
        logger.info("Nothing to update.")
        return 0

    # ── Execute changes ──
    # Step 1: Delete old docs
    logger.info("")
    logger.info("── Deleting %d old documents...", len(all_ids_to_delete))
    deleted = delete_docs(search_client, all_ids_to_delete, dry_run=False)
    results["deleted_old"] = deleted
    logger.info("  Deleted: %d", deleted)
    if deleted:
        time.sleep(2)

    # Step 2: Generate embeddings
    logger.info("── Generating embeddings for %d documents...", len(all_docs_to_upload))
    all_docs_to_upload = generate_embeddings(all_docs_to_upload, dry_run=False)

    # Step 3: Upload
    logger.info("── Uploading %d documents...", len(all_docs_to_upload))
    uploaded = upload_docs(search_client, all_docs_to_upload, dry_run=False)
    results["uploaded"] = uploaded

    logger.info("")
    logger.info("=" * 70)
    logger.info("  UPDATE COMPLETE")
    logger.info("=" * 70)
    logger.info("  Deleted:   %d", deleted)
    logger.info("  Uploaded:  %d", uploaded)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
