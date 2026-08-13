# Phase 2 report

## Implemented

- `PersistentAgentState` versioned directories (`agent_vNNN`)
- Episodic memory: MiniLM embeddings + cosine/FAISS index, recency, capability tags, provenance
- Semantic memory: merge, support counting, supersede, deprecate, conflict detection
- `PersistentAgent` injects retrieved memories and rules into the next-task prompt
- `expolicy inspect-agent` dumps store sizes, harness, lineage

## Commands

```bash
pytest tests/unit/test_checkpoint.py tests/unit/test_memory_skills.py -q
python -m expolicy.cli.main inspect-agent outputs/phase1_static/agent_v000
```

## Results

Checkpoint roundtrip and semantic merge/supersede tests pass. Skills invoke real BM25 tools.

## Known limitations

Episodic retrieval quality depends on MiniLM; not trained on Helion names.

## Next phase

Phase 3: capability diagnosis from ≥100 real trajectories.
