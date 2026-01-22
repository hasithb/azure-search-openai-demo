sample = """1.1 “Control” in the context of disclosure includes documents: (a) which are or were in a party’s physical possession; (b) in respect of which a party has or has had a right to possession; or (c) in respect of which a party has or has had a right to inspect or take copies.
1.2 “Copy” means a facsimile of a document either in the same format as the document being copied or in a similar format that is readable by the recipient, and in all cases having identical content.
1.3 “Data Sampling” means the process of checking data by identifying and checking representative individual documents.
1.4 “Disclose” comprises a party stating that a document that is or was in its control has been identified or forms part of an identified class of documents and either producing a copy, or stating why a copy will not be produced.
1.5 “Disclosure Certificate” means a certificate that is substantially in the form set out in Appendix 3 and signed in accordance with the Practice Direction.
1.6 “Disclosure Review Document” means as the case may be the Disclosure Review Document at Appendix 2, or in the case of Less Complex Claims the Disclosure Review Document at Appendix 6, which is to be completed by the parties pursuant to the Practice Direction, in respect of any application for Extended Disclosure.
## 1.7 “Electronic Image” means an electronic representation of a paper document.
1.8 “Keyword Search” means a software-aided search for words across the text of an electronic document.
[PRACTICE DIRECTION 57AD – DISCLOSURE IN THE BUSINESS AND PROPERTY COURTS > 1.1 This Practice Direction provides for disclosure in the Business and Property Courts.]
1.9 “Less Complex Claim” means a claim which the parties have agreed or the Court has ordered is one that meets the criteria for the Less Complex Claims regime as set out in Appendix 5 of this Practice Direction.
"""


def clean_supporting_content_for_display(s: str) -> str:
    lines = s.splitlines()
    cleaned = []
    for line in lines:
        updated = line
        updated = __import__("re").sub(r"^##\s*", "", updated)
        updated = __import__("re").sub(r"\[[^\]]*(PRACTICE\s*DIRECTION|PD\s*\d+|PART\s+\d+|SECTION\s+\d+|APPENDIX|>)[^\]]*\]\s*", "", updated, flags=__import__("re").I)
        cleaned.append(updated)

    filtered = []
    for line in cleaned:
        trimmed = line.strip()
        if not (trimmed.startswith("[") and trimmed.endswith("]")):
            filtered.append(line)
            continue
        if __import__("re").match(r"^\[\d+\]$", trimmed):
            filtered.append(line)
            continue
        is_metadata = __import__("re").search(r"\b(PRACTICE\s*DIRECTION|PD\s*\d+|PART\s+\d+|SECTION\s+\d+|APPENDIX|>)", trimmed, __import__("re").I)
        if not is_metadata:
            filtered.append(line)

    with_spacing = []
    import re
    for line in filtered:
        trimmed = line.strip()
        is_numbered = re.match(r"^\d+(?:\.\d+)?\b", trimmed) is not None
        prev = with_spacing[-1] if with_spacing else ""
        if is_numbered and with_spacing and prev.strip() != "":
            with_spacing.append("")
        with_spacing.append(line)

    return "\n".join(with_spacing)


print(clean_supporting_content_for_display(sample))
