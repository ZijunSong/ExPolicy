# Architecture

ExPolicy is a closed loop over **experience**, not a one-shot memory+LoRA stack.

```
Current Agent Ξ
    → Capability Diagnosis C
    → Experience Policy π(q, r | Ξ, C)
        q: acquisition (passive | uncertainty | failure-replay | frontier | learned | reward-seeking)
        r: assimilation carrier
    → Real environment interaction (Helion BM25 search)
    → Verification + typed abstraction
    → Adaptive routing into carriers
    → Realized utility on future/retention probes
    → Policy update from realized utility
```

## Persistent state

`PersistentAgentState` versions as `agent_v000`, `agent_v001`, … Each version directory stores manifest, LoRA adapter, episodic/semantic/skill/harness stores, capability profile, metrics, and lineage.

## Search agent

`LLMSearchAgent` is a ReAct loop with tools `search`, `open`, `find`, `answer`. Every model call and tool observation is stored on `Experience`. The agent must open evidence before answering (harness-configurable).

## Carriers

| Carrier | Implementation |
|---|---|
| discard | verifier / low utility |
| episodic | MiniLM + FAISS/NumPy cosine index, recency, capability tags |
| semantic | merge / supersede / deprecate rules |
| skill | executable `SkillStep` sequences that call real tools |
| harness | allow-listed config patches, regression probes, rollback |
| weight | PEFT LoRA SFT on verified successful transcripts |

## Experience Policy

A small shared MLP (`ExperiencePolicyModel`) with:

- acquisition head (rank candidates by predicted utility)
- assimilation head (carrier classification)
- utility head (regress realized ΔC / cost − λI)

Inputs are capability scores, failure distribution, candidate **metadata** (not gold answers), budget. Training uses `UtilityRecord`s from A/B intervention evaluation.

## Logging

`EventLogger` appends JSONL events: `TASK_STARTED`, `TOOL_CALL`, `EXPERIENCE_*`, `ACQUISITION_DECISION`, `ASSIMILATION_DECISION`, `WEIGHT_UPDATED`, `UTILITY_MEASURED`, `AGENT_VERSION_CREATED`, …
