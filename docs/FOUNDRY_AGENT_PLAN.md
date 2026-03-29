# Legal Agent Framework — CPR, Court Guides & Pre-Action Protocols

> **Status**: Planning  
> **Created**: February 2026  
> **Updated**: February 28, 2026 — Narrowed scope to CPR/PDs, Court Guides, Pre-Action; non-OpenAI model options  
> **Codebase**: Separate from main RAG app (`legal-agent-framework/`)  
> **Framework**: Microsoft Agent Framework + Azure AI Foundry Hosted Agents + Workflows  
> **Index**: `legal-court-rag-index-v3` on `cpr-rag.search.windows.net`  
> **Delivery**: Microsoft Teams + M365 Copilot (via Foundry Agent Applications)  
> **Multi-Model**: See §3.5 — OpenAI + non-OpenAI (DeepSeek-R1, Grok-4, Llama-4) options evaluated  
> **Scope**: Civil Procedure Rules · Practice Directions · Court Guides · Pre-Action Protocols

---

## 1. Vision & Goals

Build a **focused legal assistant** that helps UK law firms navigate Civil Procedure Rules, Practice Directions, Court Guides, and Pre-Action Protocols — accessible directly in Microsoft Teams and M365 Copilot. A conductor agent routes to specialist agents, with additional specialists added in future phases.

### Scope (This Phase)

| Content Area | Documents in Index | Category |
|-------------|-------------------|----------|
| **Civil Procedure Rules (CPR)** | 310 docs | `Civil Procedure Rules and Practice Directions` |
| **Practice Directions (PDs)** | (included in CPR count) | `Civil Procedure Rules and Practice Directions` |
| **Pre-Action Protocols** | (included in CPR count) | `Civil Procedure Rules and Practice Directions` |
| **Chancery Division Guide** | 272 docs | `Chancery Division` |
| **Commercial Court Guide** | 138 docs | `Commercial Court` |
| **Technology & Construction Court** | 63 docs | `Technology and Construction Court` |
| **King's Bench Division Guide** | 39 docs | `King's Bench Division` |
| **Patents Court Guide** | 28 docs | `Patents Court` |
| **Total** | **850 docs** | |

### End-State Architecture (Current Scope)

```
Law Firm Users (in Teams / M365 Copilot)
         │
         ▼
┌─────────────────────────────────────┐
│   @LegalAssistant (Conductor)       │  ← Published Agent Application
│   Routes to specialist agents       │
├─────────────────────────────────────┤
│ @CPRResearch   │ @CourtGuide       │  ← Phase 1-2 specialists
└─────────────────────────────────────┘
```

### What Each Agent Does

| Agent | Purpose | Key Capability |
|-------|---------|----------------|
| **Conductor** | Routes user questions to the right specialist | Understands legal intent, asks clarifying questions, synthesises multi-agent responses |
| **CPR Research** | Searches CPR, Practice Directions, and Pre-Action Protocols iteratively | Think → Search → Refine → Cite. Up to 5 search iterations per question |
| **Court Guide** | Court-division-specific procedures | Filters by Chancery, Commercial, TCC, KBD, Patents Court |

### Future Agents (Not In Current Scope)

| Agent | Purpose | When |
|-------|---------|------|
| **Costs Advisor** | Part 36 offers, costs consequences, budgeting | After core agents proven |
| **Deadline Tracker** | Time limits and key date calculations | After core agents proven |
| **Drafter** | Helps draft court documents | After multi-agent framework established |
| **Case Analyser** | Analyses uploaded case documents | After multi-agent framework established |

### Why Multiple Agents?

| Concern | Decision |
|---------|----------|
| CPR/PD research and Court Guide lookups have different retrieval strategies | Separate specialists with focused prompts and filters are more accurate |
| Conductor pattern handles ambiguity | Users ask one agent; conductor decides if it's CPR or Court Guide |
| Foundry Workflows support sequential and group chat patterns | Multi-agent orchestration is first-class in Foundry |
| Teams publishing gives the agent its own identity and RBAC | Granular access control |
| Different content types may benefit from different models | Flexibility to assign optimal model per specialist |

### Why a Separate Codebase?

| Concern | Decision |
|---------|----------|
| The main RAG app is a chat UI with single-shot retrieval | Multi-agent framework needs iterative reasoning loops |
| Foundry hosted agents run as containers on port 8088 | Different runtime from the Quart backend |
| Different deployment lifecycle (ACR → Foundry Agent Service → Teams) | Independent deploy + publish pipeline |
| Published as Agent Applications to Teams/M365 Copilot | Foundry's built-in channel publishing |
| Main app continues to work as-is | No risk to existing production |

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     User Channels                                │
│   Microsoft Teams │ M365 Copilot │ Foundry Playground │ REST API │
└────────────┬─────────────────────────────────────────────────────┘
             │  Activity Protocol (Teams/M365)
             │  Responses API (REST/Playground)
             ▼
┌──────────────────────────────────────────────────────────────────┐
│          Agent Application (Published, Stable Endpoint)          │
│  Entra Identity, RBAC, and routing policy                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                CONDUCTOR AGENT                              │ │
│  │         "Legal Assistant" — user-facing in Teams            │ │
│  │                                                             │ │
│  │  Model: Orchestrator (see §3.5 for model options)           │ │
│  │  Role:  Understands intent → routes to specialist agents    │ │
│  │         Asks clarifications → synthesises final answers     │ │
│  │                                                             │ │
│  │  ┌──────────────────┐ ┌──────────────────┐                 │ │
│  │  │   CPR Research   │ │   Court Guide    │                 │ │
│  │  │   Agent          │ │   Agent          │                 │ │
│  │  │                  │ │                  │                 │ │
│  │  │ CPR Parts/Rules  │ │ Chancery         │                 │ │
│  │  │ Practice Dirs    │ │ Commercial       │                 │ │
│  │  │ Pre-Action       │ │ TCC              │                 │ │
│  │  │ Protocols        │ │ KBD, Patents     │                 │ │
│  │  └────┬─────────────┘ └────┬─────────────┘                 │ │
│  │       │                     │                               │ │
│  │  ┌────┴─────────────────────┴────────────────────────┐      │ │
│  │  │              Shared Tool Layer                     │      │ │
│  │  │  search_cpr_rules()    search_court_guides()       │      │ │
│  │  │  get_subsections()     get_document_by_id()        │      │ │
│  │  │  deep_legal_analysis() ask_clarification()         │      │ │
│  │  └────────────────────────────────────────────────────┘      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Costs    │ │ Deadline │ │ Drafter  │ │ Case     │  ← Future  │
│  │ Advisor  │ │ Tracker  │ │          │ │ Analyser │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
└──────────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Azure Services                               │
│                                                                  │
│  Azure AI Search            Models (East US 2)                   │
│  cpr-rag.search.            ┌──────────────────────────────┐     │
│  windows.net                │ Azure OpenAI deployments:    │     │
│  ├── legal-court-rag-       │   searchagent (gpt-4.1-mini) │     │
│  │   index-v3 (850 docs)    │   gpt-5-mini (fallback)      │     │
│  │   ├── 310 CPR/PDs        │   text-embedding-3-large     │     │
│  │   └── 540 Court Guides   │                              │     │
│  └── Semantic ranker        │ Non-OpenAI (agent-supported):│     │
│                             │   DeepSeek-R1-0528           │     │
│  ACR                        │   Grok-4                     │     │
│  cprragacrot6tupm5qi5wy     │   Llama-4-Maverick           │     │
│  .azurecr.io                │   MAI-DS-R1                  │     │
│                             └──────────────────────────────┘     │
│  Foundry Project           (To Create)                           │
│  in rg-cpr-rag, East US 2                                       │
│  + Capability Host + Agent Application                           │
└──────────────────────────────────────────────────────────────────┘
```

### Teams / M365 Copilot Publishing Flow

```
Agent Version (in Foundry Project)
     │
     ▼  Publish Agent
Agent Application (Azure resource)
  ├── Own Entra Agent Identity
  ├── Own RBAC scope (Azure AI User to invoke)
  ├── Stable endpoint URL (survives version updates)
  │
  ├── Protocol: Responses API (REST/Playground)
  └── Protocol: Activity Protocol (Teams/M365)
          │
          ▼  Publish to M365/Teams
Microsoft Teams Channel
  └── @LegalAssistant available in Teams chat
  └── Law firm users interact naturally
```

---

## 3. Project Structure

```
legal-agent-framework/
├── README.md                    # Setup & usage guide
├── Dockerfile                   # Container image (linux/amd64)
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for local development
├── .env                         # Local env vars (gitignored)
│
├── agents/                      # Agent definitions
│   ├── __init__.py
│   ├── conductor.py             # Conductor agent — routes to specialists
│   ├── cpr_research.py          # CPR, PDs & Pre-Action Protocol specialist
│   └── court_guide.py           # Court Guide specialist (5 divisions)
│
├── tools/                       # Shared agent tool functions
│   ├── __init__.py
│   ├── search_tools.py          # search_cpr_rules(), search_court_guides()
│   ├── subsection_tools.py      # get_subsections(), get_document_by_id()
│   ├── deep_analysis.py         # deep_legal_analysis() → reasoning model
│   └── clarification_tools.py   # ask_clarification() for ambiguous queries
│
├── models/                      # Multi-model configuration
│   ├── __init__.py
│   ├── config.py                # Model deployment configs + env vars
│   └── clients.py               # Pre-configured model clients
│
├── prompts/                     # System prompts per agent
│   ├── conductor.py             # Routing + synthesis instructions
│   ├── cpr_research.py          # CPR/PD/Pre-Action search + citation instructions
│   └── court_guide.py           # Court-division-aware instructions
│
├── workflows/                   # Foundry Workflow definitions (YAML)
│   └── legal_assistant.yaml     # Main workflow: conductor → specialists
│
├── main.py                      # Entry point — creates agents, starts server
│
├── tests/                       # Unit & integration tests
│   ├── __init__.py
│   ├── test_conductor.py        # Routing accuracy tests
│   ├── test_search_tools.py     # Search tool tests
│   └── test_agent_local.py      # Integration tests against localhost:8088
│
├── scripts/                     # Utility scripts
│   ├── deploy.py                # Build + push + create version
│   ├── publish.py               # Publish Agent Applications + Teams
│   └── test_local.sh            # curl commands for local testing
│
└── evals/                       # Evaluation using Foundry evaluators
    ├── ground_truth.jsonl        # Symlink to ../evals/ground_truth_cpr.jsonl
    └── run_evaluation.py         # Agent evaluation harness
```

---

## 3.5. Model Routing Strategy — OpenAI and Non-OpenAI Options

### Non-OpenAI Model Support in Foundry Agent Service

Foundry Agent Service **officially supports non-OpenAI models** for agents. The [model-region support documentation](https://learn.microsoft.com/en-us/azure/foundry-classic/agents/concepts/model-region-support#non-openai-models) lists these as "models sold directly by Azure":

| Model | Strength | Use Case |
|-------|----------|----------|
| **DeepSeek-R1-0528** | Advanced long-form, multi-step reasoning | Deep legal cross-referencing |
| **MAI-DS-R1** | Deterministic, precision-focused reasoning | Exact rule interpretation |
| **Grok-4** | Frontier-scale reasoning, complex multi-step | Orchestrator alternative |
| **Grok-4-fast-reasoning** | Accelerated agentic reasoning | Workflow automation |
| **Grok-4-fast-non-reasoning** | High-throughput, low-latency generation | Fast search response |
| **Llama-4-Maverick-17B-128E-Instruct-FP8** | FP8-optimised, cost-efficient inference | Fast retrieval specialist |
| **Llama-3.3-70B-Instruct** | Enterprise Q&A, decision support | General research |
| **DeepSeek-V3.1** | Enhanced multimodal reasoning + grounded retrieval | Search + reasoning |
| **gpt-oss-120b** | Open-ecosystem, transparency + reproducibility | Audit-friendly |

> **Note**: Claude (Anthropic) is **NOT** listed as agent-supported — it's available in East US 2 as a serverless model but not officially approved for Foundry Agent Service. It can still be called from `@ai_function` tools via direct API.

### How Non-OpenAI Models Work in Hosted Agents

For **hosted agents** (containerised code), there are two approaches:

1. **`AzureAIAgentClient` with Foundry direct models**: The Foundry SDK serves non-OpenAI models through the OpenAI-compatible `/openai` route on the project endpoint. If a model is listed as agent-supported, `AzureAIAgentClient` can use it as the deployment name. Deploy the model through the Foundry portal model catalog (filter by "Agent supported").

2. **Custom code** (maximum flexibility): Since hosted agents are containers, your code can call any model endpoint directly — Azure OpenAI, serverless (MaaS), or even external APIs. The hosting adapter (`from_agent_framework()`) only wraps the HTTP protocol; it doesn't care what model you use internally.

### Recommended Model Options

Three options evaluated for the CPR/Court Guide agent:

#### Option A: Pure OpenAI (Simplest)

| Agent | Model | Deployment | TPM | Notes |
|-------|-------|------------|-----|-------|
| Conductor | gpt-5.2 | `legal-orchestrator` (NEW) | 10,000 | Best intent classification |
| CPR Research | gpt-4.1-mini | `searchagent` (EXISTS) | 12,000 | Fast iterative search |
| Court Guide | gpt-4.1-mini | `searchagent` (EXISTS) | 12,000 | Shared with CPR Research |
| Deep reasoning | o3 | `legal-reasoning` (NEW) | 5,000 | Complex cross-referencing (`reasoning_effort` = low/medium/high) |

**Pros**: Simplest to implement; `AzureAIAgentClient` works out of the box; well-documented.  
**Cons**: Locked to OpenAI models only; gpt-5.2 pricing.

#### Option B: Hybrid — OpenAI Orchestrator + Non-OpenAI Reasoning (Recommended)

| Agent | Model | Type | Notes |
|-------|-------|------|-------|
| Conductor | gpt-4.1-mini | Azure OpenAI | Cost-effective routing (doesn't need frontier model) |
| CPR Research | gpt-4.1-mini | Azure OpenAI | Fast iterative search (12K TPM existing) |
| Court Guide | gpt-4.1-mini | Azure OpenAI | Shared, division-filtered search |
| Deep reasoning | **DeepSeek-R1-0528** | Foundry direct | Superior multi-step reasoning for legal cross-referencing |
| Fallback reasoning | **MAI-DS-R1** | Foundry direct | Deterministic precision for exact rule interpretation |

**Pros**: Best reasoning quality via DeepSeek-R1; lower cost than gpt-5.2 for orchestrator (gpt-4.1-mini handles routing well); existing deployment reused.  
**Cons**: Two model types to manage; DeepSeek-R1 latency can be high for deep reasoning chains.

#### Option C: Non-OpenAI Primary (Most Flexible)

| Agent | Model | Type | Notes |
|-------|-------|------|-------|
| Conductor | **Grok-4** | Foundry direct | Frontier reasoning, excellent at routing |
| CPR Research | **Llama-4-Maverick** | Foundry direct | FP8-optimised, very fast inference |
| Court Guide | **Llama-4-Maverick** | Foundry direct | Shared, cost-efficient |
| Deep reasoning | **DeepSeek-R1-0528** | Foundry direct | Best open reasoning model |

**Pros**: No OpenAI dependency; potentially lower cost; Grok-4 is frontier-quality.  
**Cons**: Less battle-tested for agent orchestration; may need more prompt engineering; Grok-4 pricing TBD.

### Decision: Start with Option A, Evaluate Option B

**Rationale**: Start with the simplest working architecture (Option A — pure OpenAI). o3 is chosen over o4-mini because legal cross-referencing demands higher-quality structured reasoning — o3 at `medium` effort outperforms o4-mini at `high` effort on multi-step rule chains, while keeping latency under 20s. The `reasoning_effort` parameter (low/medium/high) gives adaptive latency without needing multiple models. Once the agents are functional, benchmark DeepSeek-R1-0528 against o3 for the deep reasoning tool. The conductor can stay on gpt-4.1-mini (it only does routing, not reasoning).

### Model Inventory (East US 2)

#### Currently Deployed

| Deployment Name | Model | SKU | TPM | Role |
|-----------------|-------|-----|-----|------|
| `searchagent` | gpt-4.1-mini | GlobalStandard | 12,000 | Fast search (shared by CPR Research + Court Guide) |
| `gpt-5-mini` | gpt-5-mini | GlobalStandard | 1,517 | Fallback orchestrator |
| `gpt-5-nano` | gpt-5-nano | GlobalStandard | 1,500 | Existing RAG app (no change) |
| `text-embedding-3-large` | text-embedding-3-large | GlobalStandard | 2,000 | Embeddings |

#### To Deploy (Phase 1)

| Deployment Name | Model | Version | SKU | Requested TPM | Role |
|-----------------|-------|---------|-----|---------------|------|
| `legal-orchestrator` | **gpt-5.2** | 2025-12-11 | GlobalStandard | 10,000 | **Conductor agent** — intent routing + synthesis |
| `legal-reasoning` | **o3** | 2025-04-16 | GlobalStandard | 5,000 | **Deep reasoning** — complex cross-referencing (adaptive `reasoning_effort`) |

#### To Evaluate (After Phase 2)

| Model | Type | Purpose |
|-------|------|---------|
| **DeepSeek-R1-0528** | Foundry direct | Compare against o3 for deep legal reasoning |
| **MAI-DS-R1** | Foundry direct | Evaluate for deterministic rule interpretation |
| **Grok-4** | Foundry direct | Evaluate as conductor alternative (vs gpt-5.2) |

### Agent ↔ Model Mapping (Phase 1 — Option A)

| Agent | Orchestrator Model | Why | Escalation Model |
|-------|--------------------|-----|-------------------|
| **Conductor** | gpt-5.2 (10K TPM) | Best intent classification + synthesis | — |
| **CPR Research** | gpt-4.1-mini (12K TPM) | Fast iterative search — 80% of questions need this | o3 (cross-referencing) |
| **Court Guide** | gpt-4.1-mini (12K TPM) | Division-filtered search, straightforward | o3 (rare conflicts) |

### Three-Tier Architecture (Shared Across Agents)

```
TIER 1: ORCHESTRATION (gpt-5.2, 10K TPM OR future: Grok-4)
  ├── Conductor — routes to specialist agents
  ├── Synthesises final answers
  └── Can be replaced by non-OpenAI model after benchmarking

TIER 2: FAST SEARCH (gpt-4.1-mini, 12K TPM OR future: Llama-4-Maverick)
  ├── CPR Research — iterative search (1-5 iterations)
  ├── Court Guide — division-filtered search
  └── Highest throughput tier (12K TPM)

TIER 3: DEEP REASONING (o3, 5K TPM OR future: DeepSeek-R1-0528)
  ├── Complex cross-referencing (3+ CPR Parts)
  ├── Conflicting provisions analysis
  ├── reasoning_effort: low (~5-8s) | medium (~15-20s) | high (~25-40s)
  ├── o3 chosen over o4-mini: better structured multi-step reasoning for legal chains
  ├── o3 chosen over o3-pro: adequate quality at 3-4x lower latency
  └── Called as @ai_function from any specialist agent
```

### When Specialists Escalate to Deep Reasoning

Any specialist agent calls `deep_legal_analysis()` when it detects:

1. **Cross-referencing complexity** — 3+ CPR Parts interacting (e.g., Part 36 + Part 44 + Part 3)
2. **Conflicting provisions** — Court Guide vs CPR vs Practice Direction
3. **Ambiguous authority** — Which rule takes priority
4. **Pre-Action Protocol interplay** — Protocol requirements vs CPR overrides

### Deploying the New Models

```bash
# Deploy gpt-5.2 as conductor/orchestrator (GlobalStandard, 10K TPM)
az cognitiveservices account deployment create \
  --name cog-gz2m4s637t5me-us2 \
  --resource-group rg-cpr-rag \
  --deployment-name legal-orchestrator \
  --model-name gpt-5.2 \
  --model-version 2025-12-11 \
  --model-format OpenAI \
  --sku-name GlobalStandard \
  --sku-capacity 10000

# Deploy o3 for deep reasoning (GlobalStandard, 5K TPM)
# o3 chosen over o4-mini: superior multi-step legal reasoning at manageable latency
# o3 chosen over o3-pro: adequate quality, 3-4x faster (15-20s vs 60-90s)
az cognitiveservices account deployment create \
  --name cog-gz2m4s637t5me-us2 \
  --resource-group rg-cpr-rag \
  --deployment-name legal-reasoning \
  --model-name o3 \
  --model-version 2025-04-16 \
  --model-format OpenAI \
  --sku-name GlobalStandard \
  --sku-capacity 5000
```

### Non-OpenAI Evaluation Plan (After Phase 2)

Once the core agents are working with OpenAI models, benchmark these alternatives:

| Evaluation | Models Compared | Metric | Approach |
|-----------|----------------|--------|----------|
| Deep reasoning | o3 vs **DeepSeek-R1-0528** | Legal accuracy on cross-referencing questions | Ground truth eval (62 questions) |
| Orchestration | gpt-5.2 vs **Grok-4** | Routing accuracy + synthesis quality | 20-question routing test |
| Fast search | gpt-4.1-mini vs **Llama-4-Maverick** | Search result quality + latency | Latency benchmarks |
| Determinism | o3 vs **MAI-DS-R1** | Consistency across repeated queries | 10x repeat test |

---

## 4. Implementation Phases

### Phase 1: MVP — CPR Research Agent (Days 1–3)

**Goal**: One specialist agent runs locally, answers CPR/PD/Pre-Action Protocol questions using Azure AI Search

| Step | Task | Details |
|------|------|---------|
| 1.1 | **Deploy gpt-5.2 and o3** | Run `az cognitiveservices account deployment create` commands from Section 3.5 |
| 1.2 | **Create Foundry project** | Use Foundry portal in `rg-cpr-rag` (East US 2). Record the project endpoint |
| 1.3 | **Scaffold `legal-agent-framework/`** | Create directory structure from Section 3 |
| 1.4 | **Install dependencies** | `azure-ai-agentserver-agentframework==1.0.0b12`, `azure-search-documents`, `azure-identity`, `openai`, `python-dotenv` |
| 1.5 | **Implement `models/config.py`** | Multi-model config with env vars for all deployment names |
| 1.6 | **Implement `tools/search_tools.py`** | `search_cpr_rules()`, `search_court_guides()` as shared `@ai_function` tools |
| 1.7 | **Implement `agents/cpr_research.py`** | CPR Research specialist (CPR Parts, Practice Directions, Pre-Action Protocols) |
| 1.8 | **Write `prompts/cpr_research.py`** | Legal system prompt: iterative search, UK terminology, citation rules, Pre-Action Protocol awareness |
| 1.9 | **Implement `main.py`** | Wire up single agent with `AzureAIAgentClient` + `from_agent_framework()` |
| 1.10 | **Test locally** | `python main.py` → `curl localhost:8088/responses` with CPR + Pre-Action questions |

### Phase 2: Conductor + Court Guide (Days 4–6)

**Goal**: Conductor agent routes to CPR Research + Court Guide specialists

| Step | Task | Details |
|------|------|---------|
| 2.1 | **Implement `agents/court_guide.py`** | Court Guide specialist with division filtering (5 divisions) |
| 2.2 | **Implement `agents/conductor.py`** | Conductor agent — receives all questions, classifies intent, routes to specialist |
| 2.3 | **Write `prompts/conductor.py`** | Routing instructions: CPR/PD/Pre-Action → CPR Research; division-specific → Court Guide |
| 2.4 | **Implement agent-to-agent routing** | Conductor calls specialists via `@ai_function` tools |
| 2.5 | **Add clarification logic** | Conductor asks "Which court division?" when ambiguous |
| 2.6 | **Add `tools/deep_analysis.py`** | Deep reasoning tool, callable from any specialist |
| 2.7 | **Test routing accuracy** | 20 test questions covering CPR, PDs, Pre-Action Protocols, and Court Guides |

### Phase 3: Containerise & Deploy (Days 7–9)

**Goal**: All agents deployed to Foundry Agent Service

| Step | Task | Details |
|------|------|---------|
| 3.1 | **Create Dockerfile** | Based on `python:3.12-slim`, copy all agents/tools, expose 8088 |
| 3.2 | **Create capability host** | `az rest --method put` to create account-level capability host |
| 3.3 | **Configure ACR permissions** | Grant project MI `Container Registry Repository Reader` on ACR |
| 3.4 | **Build & push image** | `docker build --platform linux/amd64` → push to ACR |
| 3.5 | **Create hosted agent version** | `AIProjectClient.agents.create_version()` with `HostedAgentDefinition` |
| 3.6 | **Deploy agents** | Start deployment with min 0 / max 2 replicas |
| 3.7 | **Test via Foundry playground** | Open Foundry portal → Agent Builder → Test each specialist |

### Phase 4: Evaluation & Non-OpenAI Benchmarking (Days 10–13)

**Goal**: Measure quality, benchmark non-OpenAI alternatives, tune prompts

| Step | Task | Details |
|------|------|---------|
| 4.1 | **Evaluate routing accuracy** | % of questions routed to correct specialist by Conductor |
| 4.2 | **Run Foundry evaluators** | `IntentResolutionEvaluator`, `TaskAdherenceEvaluator`, `ToolCallAccuracyEvaluator` |
| 4.3 | **Run custom legal metrics** | Precedent matching, legal terminology, statute citation |
| 4.4 | **Deploy DeepSeek-R1-0528** | Deploy via Foundry model catalog, evaluate vs o3 on complex questions |
| 4.5 | **Benchmark Grok-4** | Evaluate as conductor/orchestrator alternative (routing accuracy) |
| 4.6 | **Decide model composition** | Adopt Option B (hybrid) if non-OpenAI reasoning outperforms |
| 4.7 | **Tune specialist prompts** | Refine each specialist's instructions based on evaluation failures |

### Phase 5: Publish to Teams & M365 Copilot (Days 14–17)

**Goal**: Law firm users access the agent in Microsoft Teams

| Step | Task | Details |
|------|------|---------|
| 5.1 | **Create Agent Application** | Publish Conductor agent as Agent Application (Azure resource) |
| 5.2 | **Configure Activity Protocol** | Set protocol to Activity Protocol for Teams delivery |
| 5.3 | **Register Entra identity** | Agent Application gets its own Entra identity + RBAC |
| 5.4 | **Publish to Teams** | Agent available as `@LegalAssistant` in Teams chat |
| 5.5 | **User acceptance testing** | Law firm users test with real CPR/Court Guide/Pre-Action questions |
| 5.6 | **Monitor & iterate** | Track usage, routing accuracy, user satisfaction |

### Future Phases

| Phase | Description |
|-------|-------------|
| **Phase 6: Costs Advisor Agent** | Part 36 offers, costs consequences, budgeting (cross-references Part 36, Part 44, Part 45) |
| **Phase 7: Deadline Tracker Agent** | Time limits and key date calculations with CPR Part 2 day-counting rules |
| **Phase 8: Drafter Agent** | Template-driven court document drafting (claim forms, witness statements, skeleton arguments) |
| **Phase 9: Case Analyser Agent** | Upload case documents, identify issues, relevant CPR rules, procedural next steps |
| **Phase 10: Additional Indexes** | Add legislation.gov.uk index, case law index as new search tools |
| **Phase 11: Bing Web Search** | Add `BingGroundingToolDefinition` for live web search beyond the index |

---

## 5. Key Implementation Details

### 5.1 Main Entry Point (`main.py`)

```python
"""
Legal Agent Framework — CPR, Court Guides & Pre-Action Protocols.
Serves the Conductor agent via Foundry Hosted Agent protocol.

The Conductor routes questions to specialist agents:
  - CPR Research (gpt-4.1-mini) — CPR rules, PDs, pre-action protocols
  - Court Guide (gpt-4.1-mini) — division-specific procedures
"""
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv(override=False)  # Foundry env vars take precedence at runtime

from agent_framework.azure import AzureAIAgentClient
from azure.ai.agentserver.agentframework import from_agent_framework
from azure.identity.aio import DefaultAzureCredential

from models.config import PROJECT_ENDPOINT, ORCHESTRATOR_DEPLOYMENT

# Conductor agent — primary user-facing agent
from agents.conductor import create_conductor_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    async with (
        DefaultAzureCredential() as credential,
        AzureAIAgentClient(
            project_endpoint=PROJECT_ENDPOINT,
            model_deployment_name=ORCHESTRATOR_DEPLOYMENT,  # gpt-5.2
            credential=credential,
        ) as client,
    ):
        agent = create_conductor_agent(client)

        logger.info(
            "Legal Agent Framework running on http://localhost:8088\n"
            f"  Conductor model: {ORCHESTRATOR_DEPLOYMENT}\n"
            f"  Specialists: CPR Research, Court Guide"
        )
        server = from_agent_framework(agent)
        await server.run_async()


if __name__ == "__main__":
    asyncio.run(main())
```

### 5.2 Conductor Agent (`agents/conductor.py`)

```python
"""
Conductor Agent — routes user questions to specialist agents.
Uses gpt-5.2 for intent classification and response synthesis.
"""
from agent_framework import ai_function
from typing import Annotated

from prompts.conductor import CONDUCTOR_INSTRUCTIONS

# Import specialist agent functions (these wrap inner agents)
from agents.cpr_research import handle_cpr_question
from agents.court_guide import handle_court_guide_question


def create_conductor_agent(client):
    """Create the Conductor agent with routing tools."""
    return client.create_agent(
        name="LegalAssistant",
        instructions=CONDUCTOR_INSTRUCTIONS,
        tools=[
            handle_cpr_question,
            handle_court_guide_question,
        ],
    )
```

### 5.3 CPR Research Specialist (`agents/cpr_research.py`)

```python
"""
CPR Research Agent — iterative search specialist.
Uses gpt-4.1-mini (12K TPM) for fast iterative search.
Escalates to o3 for complex cross-referencing.
"""
import json
from typing import Annotated
from agent_framework import ai_function

from tools.search_tools import search_cpr_rules, search_court_guides
from tools.subsection_tools import get_subsections, get_document_by_id
from tools.deep_analysis import deep_legal_analysis
from prompts.cpr_research import CPR_RESEARCH_INSTRUCTIONS


@ai_function
def handle_cpr_question(
    question: Annotated[str, "The user's CPR-related question"],
    context: Annotated[str, "Any clarification context from the conversation"] = "",
) -> str:
    """
    Route a Civil Procedure Rules question to the CPR Research specialist.
    Use this when the user asks about:
    - CPR Parts, Rules, or Practice Directions
    - Pre-Action Protocols (e.g., professional negligence, personal injury)
    - Procedural time limits and filing requirements
    - Service of documents, statements of case, evidence rules
    - General procedural questions not specific to a court division
    """
    # In production, this would invoke a sub-agent with its own
    # AzureAIAgentClient bound to gpt-4.1-mini (searchagent deployment).
    # For the MVP, it delegates to the search tools directly.
    from models.clients import get_search_agent_client

    client = get_search_agent_client()
    result = client.chat.completions.create(
        model="searchagent",  # gpt-4.1-mini
        messages=[
            {"role": "system", "content": CPR_RESEARCH_INSTRUCTIONS},
            {"role": "user", "content": f"Question: {question}\nContext: {context}"},
        ],
        tools=[
            # Search tools available to this specialist
            {"type": "function", "function": search_cpr_rules.schema},
            {"type": "function", "function": get_subsections.schema},
            {"type": "function", "function": get_document_by_id.schema},
            {"type": "function", "function": deep_legal_analysis.schema},
        ],
        max_tokens=4096,
    )

    return result.choices[0].message.content or "Unable to find relevant CPR rules."
```

### 5.4 Search Tools (`tools/search_tools.py`)

```python
"""Azure AI Search tool functions — shared across all specialist agents."""
import json
from typing import Annotated
from agent_framework import ai_function
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from models.config import SEARCH_SERVICE, SEARCH_INDEX


def _get_search_client():
    return SearchClient(
        endpoint=f"https://{SEARCH_SERVICE}.search.windows.net",
        index_name=SEARCH_INDEX,
        credential=DefaultAzureCredential(),
    )


@ai_function
def search_cpr_rules(
    query: Annotated[str, "Search query for Civil Procedure Rules"],
    top_k: Annotated[int, "Number of results to return (default 5)"] = 5,
) -> str:
    """
    Search the CPR index for rules, practice directions, and procedural guidance.
    Use for: procedural rules, time limits, filing requirements, costs, CPR Part/Rule references.
    """
    client = _get_search_client()
    results = client.search(
        search_text=query,
        filter="category ne 'Chancery Division' and category ne 'Commercial Court' "
               "and category ne 'Technology and Construction Court' "
               "and category ne 'King''s Bench Division' and category ne 'Patents Court'",
        query_type="semantic",
        semantic_configuration_name="default",
        top=top_k,
        select=["id", "content", "sourcepage", "sourcefile", "category", "subsection_id"],
    )

    docs = []
    for r in results:
        docs.append({
            "id": r["id"],
            "source": r["sourcepage"],
            "category": r.get("category", ""),
            "content": r["content"][:1500],
            "relevance": r["@search.score"],
        })

    return json.dumps(docs, indent=2) if docs else "No CPR rules found. Try rephrasing."


@ai_function
def search_court_guides(
    query: Annotated[str, "Search query for Court Guide content"],
    court_division: Annotated[str, "Court division filter: 'Chancery Division', 'Commercial Court', 'Technology and Construction Court', 'King''s Bench Division', or 'Patents Court'. Leave empty for all."] = "",
    top_k: Annotated[int, "Number of results to return (default 5)"] = 5,
) -> str:
    """
    Search Court Guides for division-specific procedures, listing requirements,
    and practice rules. Use when CPR rules reference court-specific practice.
    """
    client = _get_search_client()

    if court_division:
        filter_str = f"category eq '{court_division}'"
    else:
        filter_str = ("category eq 'Chancery Division' or category eq 'Commercial Court' "
                      "or category eq 'Technology and Construction Court' "
                      "or category eq 'King''s Bench Division' or category eq 'Patents Court'")

    results = client.search(
        search_text=query,
        filter=filter_str,
        query_type="semantic",
        semantic_configuration_name="default",
        top=top_k,
        select=["id", "content", "sourcepage", "sourcefile", "category", "subsection_id"],
    )

    docs = []
    for r in results:
        docs.append({
            "id": r["id"],
            "source": r["sourcepage"],
            "category": r.get("category", ""),
            "content": r["content"][:1500],
            "relevance": r["@search.score"],
        })

    return json.dumps(docs, indent=2) if docs else "No Court Guide entries found."
```

### 5.5 Conductor System Prompt (`prompts/conductor.py`)

```python
CONDUCTOR_INSTRUCTIONS = """You are the Legal Assistant conductor for a UK law firm. You route questions to specialist agents and synthesise their responses.

## Your Role

You are NOT a legal research agent yourself. You are a router and synthesiser:
1. **Classify** the user's question by legal domain
2. **Route** to the correct specialist agent via tool calls
3. **Synthesise** the specialist's response into a clear answer
4. **Ask clarification** when the question is ambiguous

## Routing Rules

| Question Type | Route To | Examples |
|--------------|----------|----------|
| CPR rules, practice directions, general procedure | `handle_cpr_question` | "What is the time limit for filing a defence?" |
| Pre-Action Protocols | `handle_cpr_question` | "What steps must I take before issuing a professional negligence claim?" |
| Court-division-specific procedures | `handle_court_guide_question` | "How do I list a hearing in the Chancery Division?" |
| Cross-cutting (CPR + division practice) | Call both specialists | "What are the listing requirements for Part 8 claims in Commercial Court?" |

## Clarification Rules

Ask ONE clarifying question when:
- The user doesn't specify a court division and it matters
- The question could apply to multiple CPR Parts
- The claim value or track isn't specified but affects the answer
- The terminology is ambiguous (e.g., "application" could mean many things)

## Synthesis Rules

When you receive a specialist's response:
- Present it clearly with proper formatting
- Keep all CPR Part/Rule citations intact
- Keep all source references intact
- If multiple specialists contribute, combine into one coherent answer
- Note when Pre-Action Protocol requirements interact with CPR Rules
- Use UK legal terminology throughout
"""
```

### 5.6 Deep Legal Analysis Tool (`tools/deep_analysis.py`)

```python
"""Deep legal analysis — routes complex questions to o3 reasoning model.

o3 chosen over o4-mini: superior multi-step structured reasoning for legal rule chains.
o3 chosen over o3-pro: adequate quality at 3-4x lower latency (~15-20s vs 60-90s).
The reasoning_effort parameter gives adaptive latency:
  - low (~5-8s): 2-rule interactions, simple cross-referencing
  - medium (~15-20s): most legal analysis, CPR + PD interplay
  - high (~25-40s): 3+ Part cascading analysis, conflicting provisions
"""
import json
from typing import Annotated
from agent_framework import ai_function
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from models.config import AZURE_OPENAI_ENDPOINT, REASONING_DEPLOYMENT, REASONING_EFFORT

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(
    _credential, "https://cognitiveservices.azure.com/.default"
)


@ai_function
def deep_legal_analysis(
    question: Annotated[str, "The complex legal question requiring deep reasoning"],
    search_context: Annotated[str, "Relevant search results gathered so far"],
    reasoning_effort: Annotated[str, "Reasoning depth: 'low', 'medium', or 'high'"] = "high",
) -> str:
    """
    Deep analysis for complex multi-rule cross-referencing. Uses o3 reasoning model.
    Only use for: 3+ CPR Parts interacting, conflicting provisions, cascading deadlines,
    comparative analysis across divisions. Simple lookups should use search tools directly.

    reasoning_effort guide:
    - 'low': 2-rule interaction (fast, ~5-8s)
    - 'medium': CPR + PD interplay, standard cross-referencing (~15-20s)
    - 'high': 3+ Parts cascading, conflicting provisions, Pre-Action Protocol interplay (~25-40s)
    """
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=_token_provider,
        api_version="2025-04-01-preview",
    )

    effort = reasoning_effort if reasoning_effort in ("low", "medium", "high") else REASONING_EFFORT

    response = client.chat.completions.create(
        model=REASONING_DEPLOYMENT,
        reasoning_effort=effort,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a UK CPR expert performing deep legal analysis. "
                    "Reason through the interplay between rules, practice directions, and court guides. "
                    "Cite specific CPR Part/Rule numbers. Use UK legal terminology."
                ),
            },
            {
                "role": "user",
                "content": f"## Question\n{question}\n\n## Search Context\n{search_context}",
            },
        ],
        max_completion_tokens=4096,
    )

    return response.choices[0].message.content or "Unable to complete deep analysis."
```

### 5.7 Multi-Model Configuration (`models/config.py`)

```python
"""Configuration for the Legal Agent Framework — multi-model, multi-agent."""
import os

# Foundry project
PROJECT_ENDPOINT = os.getenv(
    "PROJECT_ENDPOINT",
    os.getenv("AZURE_AI_PROJECT_ENDPOINT", ""),
)

# Azure OpenAI endpoint (for direct API calls to reasoning/search models)
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://cog-gz2m4s637t5me-us2.openai.azure.com/",
)

# --- Agent model assignments ---
# Conductor → gpt-5.2
ORCHESTRATOR_DEPLOYMENT = os.getenv("ORCHESTRATOR_DEPLOYMENT", "legal-orchestrator")

# CPR Research + Court Guide → gpt-4.1-mini
SEARCH_DEPLOYMENT = os.getenv("SEARCH_DEPLOYMENT", "searchagent")

# Deep reasoning (shared across all agents) → o3
# o3 > o4-mini for structured multi-step legal reasoning
# o3 < o3-pro but 3-4x faster (adequate quality for real-time chat)
REASONING_DEPLOYMENT = os.getenv("REASONING_DEPLOYMENT", "legal-reasoning")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")  # low (~5-8s) | medium (~15-20s) | high (~25-40s)

# Non-OpenAI alternative (Phase 4 evaluation) — uncomment after benchmarking
# REASONING_DEPLOYMENT = os.getenv("REASONING_DEPLOYMENT", "DeepSeek-R1-0528")

# Fallback orchestrator
FALLBACK_DEPLOYMENT = os.getenv("FALLBACK_DEPLOYMENT", "gpt-5-mini")

# Azure AI Search
SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE", "cpr-rag")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "legal-court-rag-index-v3")
```

### 5.8 Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY . user_agent/
WORKDIR /app/user_agent

RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

EXPOSE 8088

CMD ["python", "main.py"]
```

### 5.9 Dependencies (`requirements.txt`)

```
azure-ai-agentserver-agentframework==1.0.0b12
azure-search-documents>=11.6.0
azure-identity>=1.19.0
openai>=1.60.0
python-dotenv>=1.0.0

# Azure Monitor / OpenTelemetry (for hosted agent tracing)
azure-monitor-opentelemetry-exporter>=1.0.0b46
opentelemetry-sdk>=1.39.0
opentelemetry-api>=1.39.0
```

### 5.10 Foundry Workflow Definition (`workflows/legal_assistant.yaml`)

```yaml
# Foundry Workflow: Conductor → Specialists (Sequential pattern)
# Created via Foundry portal Agent Builder or REST API
name: legal-assistant-workflow
description: >
  Multi-agent legal assistant workflow. The Conductor receives user
  questions, classifies intent, routes to specialist agents,
  and synthesises responses for Teams delivery.
agents:
  - name: conductor
    role: orchestrator
    model: legal-orchestrator  # gpt-5.2
    tools:
      - handle_cpr_question
      - handle_court_guide_question
  - name: cpr-research
    role: specialist
    model: searchagent  # gpt-4.1-mini
    tools:
      - search_cpr_rules
      - get_subsections
      - get_document_by_id
      - deep_legal_analysis
  - name: court-guide
    role: specialist
    model: searchagent  # gpt-4.1-mini
    tools:
      - search_court_guides
      - deep_legal_analysis
pattern: sequential  # Conductor invokes specialists as needed
publishing:
  protocol: activity  # Activity Protocol for Teams delivery
  channels:
    - teams
    - m365-copilot
```

---

## 6. Prerequisites & Setup

### 6.1 Azure Resources Needed

| Resource | Status | Action Required |
|----------|--------|-----------------|
| Azure AI Search (`cpr-rag`) | ✅ Exists | No action |
| Index (`legal-court-rag-index-v3`) | ✅ Exists (852 docs) | No action |
| Azure OpenAI (`cog-gz2m4s637t5me-us2`) | ✅ Exists | No action |
| `searchagent` (gpt-4.1-mini) deployment | ✅ Exists (12K TPM) | Shared by CPR Research, Court Guide |
| `gpt-5-mini` deployment | ✅ Exists (1,517 TPM) | Fallback orchestrator |
| `text-embedding-3-large` deployment | ✅ Exists | No action |
| ACR (`cprragacrot6tupm5qi5wy`) | ✅ Exists | Grant project MI pull access |
| **Foundry Project** | ❌ Missing | **CREATE** in `rg-cpr-rag`, East US 2 |
| **Capability Host** | ❌ Missing | **CREATE** on Foundry account |
| **`legal-orchestrator` (gpt-5.2)** | ❌ Missing | **DEPLOY** — 10K TPM GlobalStandard |
| **`legal-reasoning` (o3)** | ❌ Missing | **DEPLOY** — 5K TPM GlobalStandard |

### 6.2 Create Foundry Project

**Option A — Foundry Portal (Recommended)**:
1. Go to [ai.azure.com](https://ai.azure.com)
2. Create a new project in existing Azure AI Services resource (`cog-gz2m4s637t5me-us2`)
3. Region: East US 2
4. Note the project endpoint (format: `https://<resource>.services.ai.azure.com/api/projects/<project>`)

**Option B — Azure Developer CLI**:
```bash
azd init -t https://github.com/Azure-Samples/azd-ai-starter-basic
azd up
```

### 6.3 Create Capability Host

```bash
az rest --method put \
    --url "https://management.azure.com/subscriptions/0d1ec78c-510f-4a29-b851-be9a980219cb/resourceGroups/rg-cpr-rag/providers/Microsoft.CognitiveServices/accounts/cog-gz2m4s637t5me-us2/capabilityHosts/accountcaphost?api-version=2025-10-01-preview" \
    --headers "content-type=application/json" \
    --body '{
        "properties": {
            "capabilityHostKind": "Agents",
            "enablePublicHostingEnvironment": true
        }
    }'
```

### 6.4 RBAC Requirements

| Role | Scope | Who |
|------|-------|-----|
| Azure AI Owner | Foundry project | Your user account |
| Contributor | Azure subscription | Your user account |
| Container Registry Repository Reader | ACR | Foundry project managed identity |
| Cognitive Services OpenAI User | OpenAI resource | Foundry project managed identity |
| Search Index Data Reader | Azure AI Search | Foundry project managed identity |
| **Azure AI User** | **Agent Application** | **Law firm users (Teams access)** |

---

## 7. Local Development Workflow

```bash
# 1. Clone / create the directory
mkdir legal-agent-framework && cd legal-agent-framework

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up .env
cp .env.example .env
# Edit .env with your values

# 5. Login to Azure
az login

# 6. Run locally (starts Conductor agent on port 8088)
python main.py
# → Legal Agent Framework running on http://localhost:8088
# → Conductor routes to: CPR Research, Court Guide

# 7. Test — simple CPR question (routes to CPR Research)
curl -sS -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d '{
    "input": "What are the time limits for filing a defence under CPR?",
    "stream": false
  }'

# 8. Test — Pre-Action Protocol question (routes to CPR Research)
curl -sS -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d '{
    "input": "What steps must I take before issuing a professional negligence claim?",
    "stream": false
  }'

# 9. Test — ambiguous question (Conductor asks for clarification)
curl -sS -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d '{
    "input": "What is the procedure for applications?",
    "stream": false
  }'

# 10. Test — multi-turn
curl -sS -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d '{
    "input": "The Chancery Division",
    "stream": false,
    "previous_response_id": "<id-from-previous-response>"
  }'
```

---

## 8. Deployment Workflow

### 8.1 Build & Push Container

```bash
# 1. Build container image (MUST be linux/amd64 for Apple Silicon)
docker build --platform linux/amd64 -t legal-agent-framework:v1 .

# 2. Tag for ACR
docker tag legal-agent-framework:v1 \
  cprragacrot6tupm5qi5wy.azurecr.io/legal-agent-framework:v1

# 3. Login and push
az acr login --name cprragacrot6tupm5qi5wy
docker push cprragacrot6tupm5qi5wy.azurecr.io/legal-agent-framework:v1
```

### 8.2 Create Hosted Agent Version

```python
"""Deploy the legal agent framework to Foundry Agent Service."""
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import HostedAgentDefinition, ProtocolVersionRecord, AgentProtocol
from azure.identity import DefaultAzureCredential

PROJECT_ENDPOINT = "https://cog-gz2m4s637t5me-us2.services.ai.azure.com/api/projects/<project>"
ACR_IMAGE = "cprragacrot6tupm5qi5wy.azurecr.io/legal-agent-framework:v1"

client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

agent = client.agents.create_version(
    agent_name="legal-assistant",
    description="Legal agent framework: Conductor + CPR Research + Court Guide (CPR, PDs, Pre-Action Protocols, Court Guides)",
    definition=HostedAgentDefinition(
        container_protocol_versions=[
            ProtocolVersionRecord(protocol=AgentProtocol.RESPONSES, version="v1")
        ],
        cpu="2",
        memory="4Gi",
        image=ACR_IMAGE,
        environment_variables={
            "PROJECT_ENDPOINT": PROJECT_ENDPOINT,
            "AZURE_OPENAI_ENDPOINT": "https://cog-gz2m4s637t5me-us2.openai.azure.com/",
            "ORCHESTRATOR_DEPLOYMENT": "legal-orchestrator",
            "REASONING_DEPLOYMENT": "legal-reasoning",
            "SEARCH_DEPLOYMENT": "searchagent",
            "FALLBACK_DEPLOYMENT": "gpt-5-mini",
            "REASONING_EFFORT": "medium",  # low (~5-8s) | medium (~15-20s) | high (~25-40s)
            "AZURE_SEARCH_SERVICE": "cpr-rag",
            "AZURE_SEARCH_INDEX": "legal-court-rag-index-v3",
        },
    ),
)

print(f"Agent created: {agent.name} (version: {agent.version})")
```

### 8.3 Start Deployment

```bash
az cognitiveservices agent start \
  --account-name cog-gz2m4s637t5me-us2 \
  --project-name <project-name> \
  --name legal-assistant \
  --agent-version 1 \
  --min-replicas 0 \
  --max-replicas 2
```

### 8.4 Publish to Teams

```bash
# 1. Publish as Agent Application (creates Azure resource)
#    Done via Foundry portal: Agent Builder → Publish Agent

# 2. Configure protocol for Teams
#    Set protocol to "Activity Protocol" (required for Teams channel)

# 3. Configure Teams channel authentication
#    Set up OAuth handshake in Agent Application settings

# 4. Share to Teams
#    Foundry portal → Agent Application → Channels → Add Teams
#    Result: @LegalAssistant available in Teams chat

# 5. Assign Azure AI User role to law firm users
az role assignment create \
  --assignee <user-or-group-principal-id> \
  --role "Azure AI User" \
  --scope /subscriptions/0d1ec78c-510f-4a29-b851-be9a980219cb/resourceGroups/rg-cpr-rag
```

---

## 9. Multi-Agent Architecture Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Why Multi-Agent?** | Single agent with 10+ tools is unreliable; specialists with focused toolsets are more accurate | Proven pattern: specialist agents have higher task adherence than generalist agents |
| **Why Foundry Hosted Agents?** | Built-in Teams/M365 publishing via Activity Protocol | Custom agents would need separate Bot Framework (months of work) |
| **Why Conductor pattern?** | Central routing + response synthesis | Users interact with ONE agent in Teams; Conductor manages complexity behind the scenes |
| **Why start with OpenAI?** | Simplest path; best documentation; `AzureAIAgentClient` works natively | Non-OpenAI models (DeepSeek-R1, Grok-4, Llama) ARE supported — evaluate in Phase 4 after baseline |
| **Why one container?** | All agents share search tools + config | Simpler deployment; agents are Python classes, not separate services |
| **Why Foundry Workflows?** | Visual orchestration of agent-to-agent routing | Sequential/Group Chat patterns handle multi-specialist queries natively |
| **Why Activity Protocol for Teams?** | Required by Foundry for Teams channel delivery | Responses API for REST/Playground, Activity Protocol for Teams/M365 |
| **Why narrow scope first?** | CPR + Court Guides cover 850 indexed docs | Prove quality before adding complexity (Costs, Deadlines, Drafter in future phases) |

---

## 10. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hosted Agents is in **preview** | May have breaking changes | Pin SDK version (`1.0.0b12`), keep fallback to local testing |
| Routing errors (CPR vs Court Guide) | Wrong specialist answers question | Routing test suite; Conductor prompt tuning; fallback: Conductor answers directly |
| `gpt-5.2` rate-limited during peak | Conductor bottleneck | Fallback to `gpt-5-mini`; `gpt-4.1-mini` (12K TPM) handles specialist work |
| `o3` deep reasoning adds latency | Slow on complex questions (~15-40s depending on effort) | Adaptive `reasoning_effort`: low for simple, medium for standard, high only for 3+ Part cascading; cap `max_completion_tokens` |
| Non-OpenAI model evaluation overhead | Extra Phase 4 work | Structured benchmarks; only evaluate top 2 models (DeepSeek-R1, Grok-4) |
| Teams publishing blockers | Can't reach law firm users | Run on Foundry Playground in parallel; REST API as backup channel |
| Foundry project creation requires Owner role | May need admin help | Get RBAC assignment early (Phase 1, Day 1) |
| Pre-Action Protocol coverage gaps | Index may not cover all protocols | Audit index for protocol completeness; add missing protocols to `data/` |
| Billing (preview free until April 2026) | Cost increase after preview | Monitor usage, use `--min-replicas 0`, evaluate at Phase 4 |

---

## 11. Success Criteria

### Phase 1–2 (CPR Research + Conductor + Court Guide)

| Metric | Target | Method |
|--------|--------|--------|
| CPR Research answers correctly | ≥90% on ground truth | Foundry evaluators |
| Correct source documents cited | ≥95% precedent matching | Custom eval script |
| UK legal terminology | 100% | Legal terminology evaluator |
| Routing accuracy (Conductor) | ≥95% correct specialist selection | Routing test suite |
| Pre-Action Protocol awareness | ≥85% correct protocol identification | Manual test suite |
| Average searches per question | 2–3 (not 1, not >5) | Log analysis |

### Phase 3–4 (Deployment + Evaluation)

| Metric | Target | Method |
|--------|--------|--------|
| End-to-end latency (simple) | ≤15 seconds | Latency monitoring |
| End-to-end latency (complex) | ≤45 seconds (includes o3 at high effort) | Latency monitoring |
| Deep reasoning invoked appropriately | o3 only for complex multi-rule questions | Trace analysis |
| Non-OpenAI benchmark complete | DeepSeek-R1 + Grok-4 evaluated on legal questions | Evaluation report |
| Fallback works | Agent continues if quota exhausted | Chaos test |

### Phase 5 (Teams Publishing)

| Metric | Target | Method |
|--------|--------|--------|
| Teams bot responds in chat | 100% availability | Teams channel test |
| User auth (Entra) works | All firm users can access | RBAC test |
| Natural Teams UX | Users type questions naturally | User acceptance testing |
| Response formatting in Teams | Markdown renders correctly | Manual QA |

---

## 12. Next Steps (After Plan Approval)

1. ☐ Deploy `legal-orchestrator` (gpt-5.2) and `legal-reasoning` (o3) model deployments
2. ☐ Create Foundry project in Foundry portal (`rg-cpr-rag`, East US 2)
3. ☐ Record project endpoint in `.env`
4. ☐ Scaffold `legal-agent-framework/` directory (Section 3 structure)
5. ☐ Implement Phase 1 — CPR Research agent (CPR rules, PDs, Pre-Action Protocols)
6. ☐ Test locally with ground truth questions
7. ☐ Implement Phase 2 — Conductor + Court Guide
8. ☐ Phase 3 — Containerise + deploy to Foundry
9. ☐ Phase 4 — Evaluate + benchmark non-OpenAI models (DeepSeek-R1, Grok-4)
10. ☐ Phase 5 — Publish to Teams + M365 Copilot

---

## Appendix A: Foundry SDK Package Versions (as of Feb 2026)

| Package | Version | Purpose |
|---------|---------|---------|
| `azure-ai-agentserver-agentframework` | 1.0.0b12 | Hosting adapter for Agent Framework |
| `azure-ai-agentserver-core` | 1.0.0b12 | Core hosting adapter (used by framework-specific adapters) |
| `azure-ai-projects` | ≥2.0.0b4 | Foundry project client, agent management, publishing |
| `agent-framework` | (bundled) | Microsoft Agent Framework (included in agentserver package) |
| `azure-search-documents` | ≥11.6.0 | Azure AI Search client |
| `azure-identity` | ≥1.19.0 | DefaultAzureCredential |

## Appendix B: Resources Reference

### Infrastructure

| Resource | Value |
|----------|-------|
| Subscription | `0d1ec78c-510f-4a29-b851-be9a980219cb` |
| Tenant | `3bfe16b2-5fcc-4565-b1f1-15271d20fecf` |
| Resource Group | `rg-cpr-rag` |
| Region | East US 2 |
| Search Service | `cpr-rag.search.windows.net` |
| Search Index | `legal-court-rag-index-v3` (852 docs) |
| OpenAI Endpoint | `https://cog-gz2m4s637t5me-us2.openai.azure.com/` |
| ACR | `cprragacrot6tupm5qi5wy.azurecr.io` |

### Model Deployments

| Deployment Name | Model | Version | SKU | TPM | Agent Assignment |
|-----------------|-------|---------|-----|-----|------------------|
| `legal-orchestrator` | gpt-5.2 | 2025-12-11 | GlobalStandard | 10,000 | **Conductor** |
| `legal-reasoning` | o3 | 2025-04-16 | GlobalStandard | 5,000 | **Deep reasoning** (shared across all agents) |
| `searchagent` | gpt-4.1-mini | — | GlobalStandard | 12,000 | **CPR Research**, Court Guide |
| `gpt-5-mini` | gpt-5-mini | — | GlobalStandard | 1,517 | **Fallback orchestrator** |
| `gpt-5-nano` | gpt-5-nano | — | GlobalStandard | 1,500 | Legacy (existing RAG app) |
| `text-embedding-3-large` | text-embedding-3-large | — | GlobalStandard | 2,000 | Vector embeddings (3072 dims) |

### Specialist Agents

| Agent | Model | Tools | Phase |
|-------|-------|-------|-------|
| Conductor | gpt-5.2 | handle_cpr_question, handle_court_guide_question | Phase 2 |
| CPR Research | gpt-4.1-mini | search_cpr_rules, get_subsections, get_document_by_id, deep_legal_analysis | Phase 1 (MVP) |
| Court Guide | gpt-4.1-mini | search_court_guides, deep_legal_analysis | Phase 2 |
| *Costs Advisor* | *gpt-5.2* | *search_cpr_rules, search_court_guides, deep_legal_analysis* | *Future (Phase 6)* |
| *Deadline Tracker* | *gpt-4.1-mini* | *search_cpr_rules, calculate_deadline, count_business_days* | *Future (Phase 7)* |
| *Drafter* | *gpt-5.2* | *search_cpr_rules, search_court_guides* | *Future (Phase 8)* |
| *Case Analyser* | *gpt-5.2* | *search_cpr_rules, search_court_guides, deep_legal_analysis* | *Future (Phase 9)* |

### Non-OpenAI Models to Evaluate (Phase 4)

| Model | Deployment Type | Evaluation Purpose |
|-------|----------------|--------------------|
| DeepSeek-R1-0528 | Foundry direct model | Deep reasoning alternative to o3 |
| MAI-DS-R1 | Foundry direct model | Deterministic reasoning for rule lookup |
| Grok-4 | Foundry direct model | Conductor/orchestrator alternative |

### Foundry Resources (To Create)

| Resource | Purpose |
|----------|---------|
| AI Foundry Project | Hosts the multi-agent framework, provides endpoint |
| Capability Host | Enables hosted agent deployment |
| Agent Application (Conductor) | Published as `@LegalAssistant` in Teams |
