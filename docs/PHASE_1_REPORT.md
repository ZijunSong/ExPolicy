# Phase 1 report

## Implemented

- Pydantic experience schema, costs, verification/abstraction/utility objects
- Helion Archive world (seed 13): **262 documents, 638 tasks**, 5 stages, hash splits
- Real BM25 `search` / `open` / `find` (no fabricated hits)
- LLM ReAct agent on Qwen2.5-3B-Instruct (transformers backend)
- Event logger + trajectory JSONL + evaluator (EM / F1 / tokens / steps)
- CLI: `setup`, `run-agent`, `evaluate`, `inspect-agent`

## Commands

```bash
cd /data/ppnm/expolicy
.venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/run_phase1.py
```

## Results

| Item | Value |
|---|---|
| pytest | **23 passed** |
| Round 1 (URL-style outputs, weak parser) | 50 tasks, accuracy **0.02**, 21 tool calls |
| Round 2 (robust parser + few-shot) | 50 tasks, accuracy **0.86**, EM 0.86, **148 tool calls**, 229,742 tokens, 113.6s wall |
| Model | `/data/ppnm/models/Qwen2.5-3B-Instruct` on H20 GPU 5 |
| Capability | stage-0 `single_hop` validation |
| Artifacts | `outputs/phase1_static/{metrics.json,trajectories.jsonl,experiences.jsonl,events.jsonl,agent_v000/}` |

Round 1 is retained at `outputs/phase1_static_round1/` as a real failure log (planning_failure dominant). It was **not** converted into a success.

Sample Round 2 behavior: search → open official/encyclopedia docs → short span answer (e.g. birth city `Aurel`).

## Known limitations

- 3B instruct model still needs a strict XML/URL parser; larger models should follow the schema more cleanly.
- Phase 1 static run does not write memories (Phase 2 attaches them).
- Only stage-0 single-hop was executed for the 50-task gate; later stages share the same agent.

## Next phase

Phase 2–11 code is in-tree. Phase 2 memory retrieval was smoke-tested with MiniLM: query “In which city was Lira Venn born?” retrieved that episode first among 12 stored experiences.
