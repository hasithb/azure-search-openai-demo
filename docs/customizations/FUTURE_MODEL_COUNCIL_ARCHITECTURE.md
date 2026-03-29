# Future Architecture: Multi-Model + Model Council

This document describes a **future-state** architecture (not the current deployment) that keeps the same Legal RAG capabilities while enabling:

- model choice beyond a single provider
- multi-model "Model Council" answering with synthesis
- use of Azure AI Search as the shared retrieval layer
- orchestration via Agent Framework and/or Azure AI Foundry

***

## Scope

This is a target blueprint for future implementation. It is intentionally separate from the as-built system documentation to avoid confusion.

## Goals

1. Preserve existing legal-domain behavior (citations, source traceability, category-awareness, feedback telemetry).
2. Add provider-agnostic model routing.
3. Add council mode for higher-confidence answers on complex legal questions.
4. Keep latency and cost controllable with policy-driven routing.

## Reference Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                    FUTURE LEGAL RAG (AGENT FRAMEWORK + FOUNDRY READY)                   │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  Client Apps                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Web / Teams / Internal Tools                                                        │  │
│  │ + Mode Selection: Single Model | Model Council                                      │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                                │
│                                         ▼                                                │
│  Agent Orchestrator Layer                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Agent Framework runtime OR Azure AI Foundry Agent runtime                          │  │
│  │ - tool orchestration                                                                │  │
│  │ - policy enforcement                                                                │  │
│  │ - trace and telemetry                                                               │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                                │
│                                         ▼                                                │
│  Retrieval Tool                                                                            │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Azure AI Search (hybrid + semantic)                                                │  │
│  │ - shared evidence retrieval for all candidate models                               │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                         │                                           │                    │
│                  Single-Model Path                           Council Path                │
│                         │                                           │                    │
│                         ▼                                           ▼                    │
│               ┌───────────────────┐                     ┌──────────────────────────┐    │
│               │ One selected model │                    │ Candidate models (N=2..5)│    │
│               │ generates answer   │                    │ each drafts answer        │    │
│               └─────────┬──────────┘                    └──────────────┬───────────┘    │
│                         │                                              │                 │
│                         └──────────────────────┬───────────────────────┘                 │
│                                                ▼                                         │
│                                 ┌──────────────────────────────────────┐                  │
│                                 │ Judge / Synthesizer model            │                  │
│                                 │ - legal accuracy scoring             │                  │
│                                 │ - citation validation checks         │                  │
│                                 │ - contradiction resolution           │                  │
│                                 └──────────────────┬───────────────────┘                  │
│                                                    ▼                                      │
│                                 Output Guardrails                                         │
│                                 - citation format normalization                           │
│                                 - safety/compliance filters                              │
│                                 - confidence + rationale metadata                         │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Model Council Execution Pattern

```text
1) Retrieve evidence once from Azure AI Search
2) Fan out identical evidence + prompt constraints to candidate models
3) Score each candidate using legal/citation rubric
4) Synthesize final response in judge step
5) Run output guardrails and return final answer + trace metadata
```

## Candidate Model Sources

- Azure OpenAI deployments
- Azure AI model inference endpoints (open model catalog)
- self-hosted endpoints (for example vLLM OpenAI-compatible APIs)
- additional enterprise model providers through adapter connectors

## Suggested Core Modules (Future)

```text
future-agent/
├── orchestrator/
│   ├── runtime_adapter.py         # Agent Framework or Foundry runtime binding
│   ├── policy_router.py           # single vs council vs fallback routing
│   └── telemetry.py               # traces, latency, cost, quality signals
├── retrieval/
│   └── azure_search_tool.py       # shared retrieval tool
├── model_gateway/
│   ├── registry.py                # model/provider registry
│   ├── adapters/                  # provider-specific adapters
│   └── council/
│       ├── executor.py            # fan-out execution and partial failure handling
│       ├── judge.py               # synthesis and winner selection
│       └── rubric.py              # legal-domain quality criteria
└── guardrails/
    ├── citations.py               # [1][2][3] normalization and validation
    └── compliance.py              # redaction and policy checks
```

## Operating Modes

- Fast mode: single model path for routine queries.
- High-assurance mode: council path for high-stakes, ambiguous, or low-confidence queries.
- Degraded mode: if one or more candidate models fail, continue with available candidates.

## Rollout Path

1. Build provider-agnostic gateway with one model provider first.
2. Add one additional non-primary provider and run parity tests.
3. Enable council mode behind feature flag for internal users.
4. Add dynamic routing policies (confidence, query class, SLA).
5. Promote to production after quality/latency/cost thresholds are met.

## Success Metrics

- Legal answer quality: precedent matching, citation correctness, contradiction rate.
- User trust: feedback score and answer acceptance.
- Operational: p95 latency, cost per answer, fallback frequency.
- Safety/compliance: policy violation rate and redaction correctness.
