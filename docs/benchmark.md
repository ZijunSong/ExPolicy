# Benchmark: Helion Archive continual search

The environment is a **local, deterministic retrieval corpus**, not a mock search API.

Generator: `expolicy.environments.search.world.build_world(seed)`.

## World

Procedurally generated but internally consistent entities (people, organizations, events) with:

- encyclopedia / official / news / blog documents of known reliability
- intentional rumor documents (low reliability) for conflict and recovery items
- gold document ids stored only on the `Task` object for evaluation, never given to the agent

Tools:

```
search(query) → ranked BM25 hits with snippet, source_type, reliability, year
open(document_id) → full text
find(document_id, pattern) → regex windows
```

## Splits

Each task id is hashed with the world seed into:

| split | role |
|---|---|
| deployment | passive OEL stream |
| acquisition | pool the Experience Policy may choose from |
| validation | probes / model selection |
| future | held-out capability evaluation |
| retention | old-stage replay after updates |

`SearchBenchmark.future_guard` raises `LeakageError` if future ids appear in training/acquisition sets.

## Stages

| Stage | Capabilities |
|---|---|
| 0 | single-hop retrieval |
| 1 | multi-hop, query decomposition |
| 2 | entity disambiguation, temporal reasoning |
| 3 | conflicting evidence, source reliability |
| 4 | long-context, cross-document synthesis, failure recovery |

Paper scale: ≥200 tasks / stage. Development: same generator, smaller `budget.max_tasks`.
