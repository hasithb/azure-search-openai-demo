#!/usr/bin/env python3
"""
Manage Azure AI Search synonym maps for legal terminology.

This script creates, updates, or deletes synonym maps that help the search engine
understand legal terminology variations. For example, when a user searches for
"pre-action disclosure", the search will also match documents containing
"disclosure before proceedings have started" (the official CPR terminology).

Usage:
    python scripts/manage_synonym_map.py create    # Create or update synonym map
    python scripts/manage_synonym_map.py apply     # Apply synonym map to index fields
    python scripts/manage_synonym_map.py delete    # Delete synonym map
    python scripts/manage_synonym_map.py list      # List all synonym maps
    python scripts/manage_synonym_map.py test      # Test synonym expansion

Environment Variables (loaded from azd env):
    AZURE_SEARCH_SERVICE: Name of the Azure AI Search service
    AZURE_SEARCH_INDEX: Name of the search index

For more information, see:
    https://learn.microsoft.com/en-us/azure/search/search-synonyms
"""

import argparse
import asyncio
import json
import logging
import os
import sys

from azure.identity import AzureDeveloperCliCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SynonymMap

# Add scripts directory to path for load_azd_env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_azd_env import load_azd_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Legal terminology synonym map name
SYNONYM_MAP_NAME = "legal-cpr-synonyms"

# Legal terminology synonyms in Solr format
# Format: Either comma-separated equivalents OR "term1, term2 => canonical_term"
# See: https://cwiki.apache.org/confluence/display/solr/Filter+Descriptions#FilterDescriptions-SynonymGraphFilter
#
# IMPORTANT: This synonym map handles TERMINOLOGY variations, not typos.
# For typos/fuzzy matching, use queryType=full with fuzzy operators (~1, ~2)
# in the search queries. See docs/legal_synonyms.md for details.

LEGAL_SYNONYMS = """
# ============================================================================
# TERMINOLOGY VARIATIONS (colloquial vs official CPR wording)
# ============================================================================
pre-action disclosure, pre action disclosure, disclosure before proceedings have started, CPR 31.16
third party disclosure, third-party disclosure, non-party disclosure, disclosure by a person who is not a party, CPR 31.17
freezing order, freezing injunction, mareva order, mareva injunction
search order, search and seizure order, anton piller order
summary judgment, judgment without trial, Part 24 summary judgment
default judgment, judgment in default, Part 12 default judgment
interim payment, Part 25 interim payment
security for costs, security for costs application, costs security
wasted costs, wasted costs order, costs wasted
split trial, separate trial of issues
witness statement, witness evidence, statement witness
expert witness, expert evidence, Part 35 expert
case management, case management conference, CMC
pre-trial review, PTR, pretrial review
unless order, conditional order
stay of proceedings, stay of action
discontinuance, notice of discontinuance
counterclaim, counter-claim
set-off, set off, setoff
limitation, limitation period, statute of limitations
costs budget, precedent H, costs budgeting
costs management, costs management order
proportionality, proportionate costs
indemnity basis, indemnity costs
standard basis, standard costs
costs assessment, detailed assessment
summary assessment, summary costs assessment
litigation friend, next friend
service, service of documents
acknowledgment of service, acknowledgement of service, service acknowledgment
particulars of claim, statement of claim, claim particulars
reply, reply to defence
schedule of loss, statement of special damages
standard disclosure, limited disclosure
specific disclosure, specific discovery
inspection, inspection of documents
interrogatories, requests for information, Part 18 request
admission, Part 14 admission
striking out, strike out, strike-out, struck out
allocation, allocation questionnaire, directions questionnaire, DQ
fast track, fast-track, fasttrack
small claims, small claims track
group litigation, group litigation order, GLO
representative action, representative proceedings
permission, leave, permission to appeal
permission to appeal, leave to appeal
skeleton argument, skeleton arguments, skeleton submissions
trial bundle, court bundle, hearing bundle, agreed bundle
authorities bundle, bundle of authorities
chronology, timeline of events

# ============================================================================
# PHASE 2 ADDITIONS: INTERIM REMEDIES & PROTECTIVE MEASURES (CPR Part 25)
# ============================================================================
prohibitory injunction, prohibitive injunction
mandatory injunction, positive injunction, affirmative injunction
interim injunction with notice, on-notice injunction
interim injunction without notice, ex parte injunction, without notice application
non-disclosure order, privacy injunction, secrecy order
super-injunction, non-publication order
anonymity order, anonymised injunction
self-identification order, identification order
confidentiality injunction, breach of confidence order
restraining order, restraint order, prohibiting order

# ============================================================================
# PHASE 2 ADDITIONS: GROUP LITIGATION & PARTY MANAGEMENT (CPR Parts 19-20)
# ============================================================================
GLO issues, group issues, common issues
test claim, sample claim, lead claim
Part 20 claim, additional claim, contribution claim, indemnity claim
common interest, same interest

# ============================================================================
# PHASE 2 ADDITIONS: COSTS MANAGEMENT & ASSESSMENT (CPR Parts 44-47)
# ============================================================================
Precedent H, costs schedule, costs estimate
approved budget, agreed budget, approved phase budget
costs management order, CMO, costs management direction
detailed assessment, costs taxing, taxation of costs
provisional assessment, provisional costs assessment
receiving party, recovering party, successful party
indemnity basis, indemnity standard, full indemnity
wasted costs, costs wasted, wasted costs order
fee remission, help with fees, court fee waiver, fee exemption
QOCS, qualified one-way costs shifting, one way costs shifting, costs protection
share documents, exchange documents, document sharing, disclosure

# ============================================================================
# PHASE 3 ADDITIONS: TRIAL PROCEDURE & EVIDENCE (CPR Parts 29-35)
# ============================================================================
trial readiness certificate, trial statement, trial preparation
without prejudice, off record, confidential discussion, settlement discussion
pre-trial review, pretrial conference, pre-trial conference
contempt of court, breach of order, disobedience of order

# ============================================================================
# PHASE 3 ADDITIONS: PROCEDURAL DEFECTS & REMEDIES (CPR Parts 3-4, 11-13)
# ============================================================================
abuse of process, vexatious claim, vexatious proceedings
jurisdiction, jurisdiction of court, subject matter jurisdiction
relief from sanction, relief from sanctions, relief from default, waiver of requirement, Denton relief
unless order, conditional strike-out order
totally without merit, TWM application, manifestly unfounded
amendment application, permission to amend, leave to amend

# ============================================================================
# PHASE 3 ADDITIONS: SPECIALIZED CLAIMS & LOW-VALUE PROCEDURES (CPR 45, 49, 68-76)
# ============================================================================
fixed costs, recoverable costs, prescribed costs, fixed recoverable costs, FRC
low value claim, low value personal injury, low value RTA claim
whiplash claim, soft tissue claim
road traffic accident, RTA claim, motor accident claim
official injury claims portal, OIC portal, injury claims portal
judicial review, JR application, review hearing
defamation claim, libel claim, slander claim, publication claim
qualified privilege, absolute privilege, privilege defence
media and communications list, MCL, media claim
data protection claim, GDPR claim, data subject claim

# ============================================================================
# HISTORICAL TERMINOLOGY (older terms still used by practitioners)
# ============================================================================
disclosure, discovery
claimant, plaintiff
defendant, respondent
defence, defense
interim injunction, interlocutory injunction
judicial review, JR

# ============================================================================
# ABBREVIATIONS (common legal abbreviations)
# ============================================================================
Civil Procedure Rules, CPR, C.P.R.
Practice Direction, PD, P.D.
rule, r.
r. 31.6, rule 31.6
alternative dispute resolution, ADR

# ============================================================================
# BRITISH/AMERICAN SPELLING VARIANTS
# ============================================================================
judgement, judgment
honour, honor
favour, favor
authorised, authorized
defence, defense

# ============================================================================
# PHASE 4 ADDITIONS: COURT NAMES & ABBREVIATIONS (from index categories/sourcepages)
# ============================================================================
Technology and Construction Court, TCC, construction court, TCC court
King's Bench Division, KBD, Queen's Bench Division, QBD
Chancery Division, Ch D, chancery court
Business and Property Courts, BPC, business property courts
Insolvency and Companies List, ICL, insolvency list, companies list
Financial List, financial disputes list
Circuit Commercial Court, circuit commercial
Commercial Court, commercial list
Patents Court, patents list, intellectual property court
Admiralty Court, admiralty, admiralty claims

# ============================================================================
# PHASE 4 ADDITIONS: ADMIRALTY & SHIPPING (Commercial Court Guide, Section N)
# ============================================================================
claim in rem, in rem claim, in rem proceedings, action in rem
collision claim, shipping collision, maritime collision
limitation claim, limitation of liability, tonnage limitation
arrest of vessel, ship arrest, arrested vessel, arrested property
limitation fund, limitation decree
maritime lien, maritime claim
salvage claim, salvage proceedings

# ============================================================================
# PHASE 4 ADDITIONS: ARBITRATION (Commercial Court Guide, Section O)
# ============================================================================
arbitration claim, arbitration proceedings, arbitral proceedings
arbitral award, arbitration award
challenging the award, appeal against award, set aside award, challenge to award
enforcement of award, enforce arbitral award, enforce arbitration award
arbitration claim form, Part 62 claim

# ============================================================================
# PHASE 4 ADDITIONS: NDR & MEDIATION (Commercial Court Guide Section G, Chancery Ch.10)
# ============================================================================
negotiated dispute resolution, NDR
early neutral evaluation, ENE, neutral evaluation
mediation, court mediation, ADR mediation, dispute mediation
Chancery Financial Dispute Resolution, Ch FDR, financial dispute resolution, FDR

# ============================================================================
# PHASE 4 ADDITIONS: ENFORCEMENT (CPR Parts 70-75, 83, 85, 89)
# ============================================================================
third party debt order, garnishee order, attachment of debt
charging order, charge on property, charge over asset
attachment of earnings, wage deduction, earnings arrest
writ of control, writ of execution, execution writ
enforcement agent, bailiff, certificated enforcement agent
controlled goods agreement, walking possession
judgment debtor, debtor, enforcement debtor
warrant of control, warrant of execution

# ============================================================================
# PHASE 4 ADDITIONS: POSSESSION & PROPERTY (CPR Part 55-56)
# ============================================================================
possession claim, eviction claim, possession proceedings
possession order, eviction order
trespasser, squatter, unauthorized occupant
landlord and tenant, tenancy dispute, landlord tenant dispute
renting homes, rental dispute, tenancy claim

# ============================================================================
# PHASE 4 ADDITIONS: PROBATE, ESTATES & TRUSTS (CPR Part 57, 64)
# ============================================================================
probate claim, will dispute, will contest, contested will
presumption of death, missing person declaration
inheritance claim, Inheritance Act claim, family provision claim
estate administration, administration of estate
trust claim, breach of trust
charity proceedings, Charity Commission proceedings

# ============================================================================
# PHASE 4 ADDITIONS: ELECTRONIC FILING & DISCLOSURE (PD 31B, PD 57AD)
# ============================================================================
electronic disclosure, e-disclosure, digital disclosure, PD 31B
electronic filing, e-filing, CE-File, online filing
extended disclosure, PD 57AD disclosure, disclosure in BPC
initial disclosure, voluntary disclosure, early disclosure
disclosure review document, DRD

# ============================================================================
# PHASE 4 ADDITIONS: CHANCERY-SPECIFIC ORDERS & PROCEDURES
# ============================================================================
Norwich Pharmacal order, Norwich Pharmacal application, third party disclosure order
Tomlin order, consent order with schedule, structured consent order
Beddoe order, Beddoe application, trustee directions
vesting order, transfer order, property vesting
civil restraint order, CRO, extended civil restraint order, ECRO

# ============================================================================
# PHASE 4 ADDITIONS: SPECIFIC PROCEDURES FROM INDEX
# ============================================================================
anti-social behaviour, ASBO, harassment claim, Part 65
closed material procedure, CMP, secret evidence, Part 82
terrorism prevention, TPIM, Part 80
damages claims pilot, PD 51ZB, damages pilot

# ============================================================================
# PHASE 5 ADDITIONS: EXHAUSTIVE DOCUMENT ANALYSIS (850 docs, 210 sources)
# Identified by cross-referencing ~120 regex patterns + ~300 concept phrases
# against all index content, then gap-analyzing against existing 170 rules.
# ============================================================================

# --- Court Hierarchy & Judicial Officers ---
Court of Appeal, CA, appellate court, appeal court
Supreme Court, UKSC, highest court
Family Division, family court, family proceedings
district judge, DJ, deputy district judge, DDJ
litigant in person, LIP, self-represented, unrepresented, self represented
McKenzie Friend, lay assistant

# --- Hearing Types & Formats ---
without a hearing, on the papers, paper disposal, paper application
oral hearing, attended hearing, in court hearing
remote hearing, virtual hearing, online hearing
telephone hearing, phone hearing, telephone conference
video hearing, video link hearing, Teams hearing
out of hours, out-of-hours, urgent out of hours

# --- Pre-Action ---
letter of claim, letter before claim, letter before action, pre-action letter
pre-action protocol, pre action protocol, PAP, pre-action conduct, pre action conduct

# --- Costs: Funding Arrangements (CFA/DBA/ATE/BTE/Third Party) ---
conditional fee agreement, CFA, conditional fee, no win no fee
success fee, uplift, percentage increase
damages-based agreement, DBA, damages based agreement
litigation funding, third party funding, third-party funding, litigation funder
after the event insurance, ATE insurance, ATE
before the event insurance, BTE insurance, BTE
protective costs order, PCO, costs protection order

# --- Costs: Assessment, Management & Budgeting ---
costs capping, costs cap, costs capping order, capped costs
bill of costs, bill for detailed assessment
points of dispute, objections to bill
costs and case management conference, CCMC
statement of costs, schedule of costs
costs officer, costs judge, authorised court officer
Precedent R, Precedent S, costs precedent
costs thrown away, abortive costs

# --- Trial, Listing & Preparation ---
time estimate, time estimates, estimated length of hearing
draft order, drawn order, proposed order
sealed order, perfected order
minute of order, drawn up order
reading time, pre-reading, reading list, judicial reading
trial window, trial listing window
Flexible Trials Scheme, FTS, flexible trial
Shorter Trials Scheme, STS, shorter trial
Capped Costs List, CCL, capped costs pilot
Scott Schedule, schedule of defects

# --- Appeals ---
notice of appeal, appellant's notice, appeal notice
respondent's notice, cross-appeal notice
grounds of appeal, grounds for appeal, appeal grounds
appeal bundle, appeal hearing bundle
first instance, court below, lower court

# --- Insolvency & Bankruptcy ---
winding up, winding-up, compulsory liquidation, wind up
bankruptcy, personal insolvency
receivership, receiver, court-appointed receiver

# --- Equity & Chancery ---
unfair prejudice, section 994, unfair prejudice petition
derivative claim, derivative action, shareholder derivative
breach of fiduciary duty, fiduciary breach
constructive trust, resulting trust
rectification, rectification of register
rescission, rescind
specific performance, order for specific performance
undue influence, presumed undue influence
account of profits, disgorgement of profits
passing off, passing-off
deed poll, change of name, enrolled deed, enrolment
proprietary estoppel, estoppel claim

# --- Probate & Estates ---
grant of probate, probate grant
letters of administration, administration grant
intestacy, intestate estate, dying intestate
personal representative, executor

# --- Evidence & Witnesses ---
cross-examination, cross examination, cross-examine
witness summons, subpoena, witness attendance
expert report, expert's report
single joint expert, SJE, joint expert
full and frank disclosure, duty of candour, complete disclosure

# --- International Service ---
service out of the jurisdiction, service abroad, overseas service
Hague Convention, Hague Service Convention

# --- Settlement & Offers ---
Part 36 offer, offer to settle, Calderbank offer
agreed order, consent order, order by consent

# --- Pleadings, Statements & Principles ---
statement of case, statements of case, pleading, pleadings
statement of truth, verified statement
overriding objective, dealing with cases justly

# --- Compliance & Sanctions ---
non-compliance, non compliance, failure to comply
committal, committal order, committal for contempt
penal notice, penal endorsement

# --- Submissions ---
written submissions, written arguments
closing submissions, closing arguments, final submissions

# --- Stays & Adjournments ---
adjourn, adjournment, postpone, postponement
stay of execution, execution stay

# --- Tracks & Schemes ---
multi-track, multi track, multitrack
intermediate track, intermediate claims

# --- Specialist Courts & Lists ---
Competition List, competition claims
Intellectual Property List, IP list, IPEC, Intellectual Property Enterprise Court
Revenue List, revenue claims, tax claims

# --- Claim Procedures ---
Part 8 claim, Part 8 procedure, alternative procedure claim
Part 7 claim, Part 7 procedure, ordinary procedure claim

# --- Other Important Terms ---
legal aid, public funding, publicly funded
urgent application, emergency application, urgent hearing
warrant of possession, eviction warrant
reporting restriction, reporting restrictions order
special measures, vulnerable witness measures
Aarhus Convention, Aarhus costs, environmental review costs
adjudication enforcement, enforcement of adjudicator's decision

# ============================================================================
# NOTE: TYPO/MISSPELLING HANDLING
# ============================================================================
# Typos are now handled by fuzzy search (enabled in backend/approaches/approach.py)
# which automatically applies ~1 edit distance operators to search terms.
# This is more efficient than maintaining explicit typo synonym rules.
# Removed entries: discloure, disclousure, diclosure, injuction, injuntion, etc.
""".strip()


def get_search_clients():
    """Initialize Azure Search clients with credentials."""
    load_azd_env()
    
    search_service = os.environ.get("AZURE_SEARCH_SERVICE")
    search_index = os.environ.get("AZURE_SEARCH_INDEX")
    
    if not search_service:
        raise ValueError("AZURE_SEARCH_SERVICE environment variable is not set. Run 'azd env refresh'.")
    
    endpoint = f"https://{search_service}.search.windows.net"
    credential = AzureDeveloperCliCredential(tenant_id=os.environ.get("AZURE_TENANT_ID"))
    
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    search_client = None
    if search_index:
        search_client = SearchClient(endpoint=endpoint, index_name=search_index, credential=credential)
    
    return index_client, search_client, search_index


def create_synonym_map(index_client: SearchIndexClient):
    """Create or update the legal terminology synonym map."""
    # Parse rules from the multiline string
    # Filter out comments (lines starting with #) and empty lines
    rules = [
        line.strip() 
        for line in LEGAL_SYNONYMS.split('\n') 
        if line.strip() and not line.strip().startswith('#')
    ]
    
    synonym_map = SynonymMap(
        name=SYNONYM_MAP_NAME,
        synonyms=rules  # Must be a list of strings, not a single string
    )
    
    try:
        # Try to get existing map
        existing = index_client.get_synonym_map(SYNONYM_MAP_NAME)
        logger.info(f"Updating existing synonym map: {SYNONYM_MAP_NAME}")
        result = index_client.create_or_update_synonym_map(synonym_map)
    except Exception:
        logger.info(f"Creating new synonym map: {SYNONYM_MAP_NAME}")
        result = index_client.create_or_update_synonym_map(synonym_map)
    
    logger.info(f"✅ Synonym map '{result.name}' created/updated with {len(rules)} rules")
    
    return result


def apply_synonym_map_to_index(index_client: SearchIndexClient, index_name: str):
    """Apply the synonym map to searchable text fields in the index."""
    if not index_name:
        raise ValueError("AZURE_SEARCH_INDEX environment variable is not set.")
    
    logger.info(f"Fetching index definition: {index_name}")
    index = index_client.get_index(index_name)
    
    # Find searchable string fields to apply synonyms to
    fields_updated = []
    for field in index.fields:
        # Apply to main content fields (but not vector fields)
        if (field.type in ("Edm.String", "Collection(Edm.String)") and 
            field.searchable and 
            field.name in ("content", "title", "description", "sourcepage", "chunk")):
            
            if field.synonym_map_names != [SYNONYM_MAP_NAME]:
                field.synonym_map_names = [SYNONYM_MAP_NAME]
                fields_updated.append(field.name)
    
    if fields_updated:
        logger.info(f"Applying synonym map to fields: {', '.join(fields_updated)}")
        index_client.create_or_update_index(index)
        logger.info(f"✅ Synonym map applied to {len(fields_updated)} fields")
    else:
        logger.info("ℹ️ No eligible fields found or synonym map already applied")
    
    return fields_updated


def delete_synonym_map(index_client: SearchIndexClient):
    """Delete the legal terminology synonym map."""
    try:
        index_client.delete_synonym_map(SYNONYM_MAP_NAME)
        logger.info(f"✅ Synonym map '{SYNONYM_MAP_NAME}' deleted")
    except Exception as e:
        logger.warning(f"Could not delete synonym map: {e}")


def list_synonym_maps(index_client: SearchIndexClient):
    """List all synonym maps in the search service."""
    maps = list(index_client.get_synonym_maps())
    if maps:
        logger.info(f"Found {len(maps)} synonym map(s):")
        for sm in maps:
            rule_count = len([line for line in sm.synonyms.split('\n') if line.strip()])
            logger.info(f"  - {sm.name} ({rule_count} rules)")
            # Show first few rules as preview
            rules = [line.strip() for line in sm.synonyms.split('\n') if line.strip()][:3]
            for rule in rules:
                logger.info(f"      {rule}")
            if len(rules) > 3:
                logger.info(f"      ... and {rule_count - 3} more rules")
    else:
        logger.info("No synonym maps found")
    return maps


def test_synonym_search(search_client: SearchClient):
    """Test synonym expansion with sample queries."""
    if not search_client:
        logger.warning("No search client available - cannot test")
        return
    
    test_queries = [
        "pre-action disclosure",
        "freezing order",
        "mareva injunction",
        "third party disclosure",
        "anton piller order",
    ]
    
    logger.info("Testing synonym expansion with sample queries...")
    for query in test_queries:
        results = search_client.search(query, top=3)
        hits = list(results)
        logger.info(f"  Query '{query}': {len(hits)} results")
        for hit in hits[:2]:
            title = hit.get("sourcepage", hit.get("title", "Unknown"))
            score = hit.get("@search.score", 0)
            logger.info(f"    - {title} (score: {score:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Azure AI Search synonym maps for legal terminology"
    )
    parser.add_argument(
        "action",
        choices=["create", "apply", "delete", "list", "test", "all"],
        help="Action to perform: create (synonym map), apply (to index), delete, list, test, or all (create + apply)"
    )
    args = parser.parse_args()
    
    try:
        index_client, search_client, index_name = get_search_clients()
        
        if args.action == "create":
            create_synonym_map(index_client)
        elif args.action == "apply":
            apply_synonym_map_to_index(index_client, index_name)
        elif args.action == "delete":
            delete_synonym_map(index_client)
        elif args.action == "list":
            list_synonym_maps(index_client)
        elif args.action == "test":
            test_synonym_search(search_client)
        elif args.action == "all":
            create_synonym_map(index_client)
            apply_synonym_map_to_index(index_client, index_name)
            logger.info("\n✅ All done! Synonym map created and applied to index.")
            logger.info("   Users can now search using colloquial legal terms.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
