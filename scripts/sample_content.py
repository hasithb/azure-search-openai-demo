#!/usr/bin/env python3
"""Sample content from key source documents for ground truth creation."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "data" / "legal-scraper" / "processed" / "Upload"

files_to_sample = [
    "Part_2___Application_And_Interpretation_Of_The_Rules.json",
    "Part_5___Court_Documents.json",
    "Part_6___Service_Of_Documents_chunk_000.json",
    "Part_9___Responding_To_Particulars_Of_Claim___General.json",
    "Part_10___Acknowledgment_Of_Service.json",
    "Part_14___Admissions.json",
    "Part_25___Interim_Remedies_And_Security_For_Costs_chunk_000.json",
    "Part_29___The_Multi-Track.json",
    "Part_38___Discontinuance.json",
    "Part_40___Judgments__Orders__Sale_Of_Land_Etc__chunk_000.json",
    "Part_54___Judicial_Review_And_Statutory_Review_chunk_000.json",
    "Part_55___Possession_Claims_chunk_000.json",
    "Part_70___General_Rules_About_Enforcement_Of_Judgments_And_Orders.json",
    "Part_72___Third_Party_Debt_Orders.json",
    "Chancery-Guide-2024-web_processed.json",
    "Pre-Action_Protocol_for_Personal_Injury_Claims_chunk_000.json",
    "Pre-Action_Protocol_for_the_Construction_and_Engineering_Disputes.json",
    "Pre-Action_Protocol_for_Judicial_Review.json",
    "Part_11___Disputing_The_Court_S_Jurisdiction.json",
    "Part_15___Defence_And_Reply.json",
    "Part_17___Amendments_To_Statements_Of_Case.json",
    "Part_21___Children_And_Protected_Parties_chunk_000.json",
    "Part_27___The_Small_Claims_Track.json",
    "Part_28___The_Fast_Track_And_The_Intermediate_Track.json",
    "Part_30___Transfer.json",
    "Part_37___Miscellaneous_Provisions_About_Payments_Into_Court.json",
    "Part_39___Miscellaneous_Provisions_Relating_To_Hearings.json",
    "Part_41___Damages.json",
    "Part_42___Change_Of_Solicitor.json",
    "Part_45___Fixed_Costs_chunk_000.json",
    "Part_47___Procedure_For_Assessment_Of_Costs_And_Default_Provisions_chunk_000.json",
    "Part_57___Probate__Inheritance__Presumption_Of_Death_And_Guardianship_Of_Missing_Persons_chunk_000.json",
    "Part_64___Estates__Trusts_And_Charities.json",
    "Part_66___Crown_Proceedings.json",
    "Part_67___Proceedings_Relating_To_Solicitors.json",
    "Part_68___Proceedings_under_the_European_Union__Withdrawal__Act_2018.json",
    "Part_69___Court_S_Power_To_Appoint_A_Receiver.json",
]

for fname in files_to_sample:
    f = root / fname
    if f.exists():
        data = json.loads(f.read_text())
        if isinstance(data, list):
            doc = data[0] if data else {}
        else:
            doc = data
        content = doc.get("content", "")[:600]
        sp = doc.get("sourcepage", "")
        sf = doc.get("sourcefile", "")
        cat = doc.get("category", "")
        print(f"=== {fname} ===")
        print(f"  sourcepage: {sp}")
        print(f"  sourcefile: {sf}")
        print(f"  category: {cat}")
        print(f"  content: {content}...")
        print()
    else:
        print(f"=== {fname} === NOT FOUND")
        print()
