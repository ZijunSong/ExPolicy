# Implementation status

All phases in `ExPolicy_Cursor_Agent_Implementation_Instructions.md` are present as real modules (no mock LLM, no fake BM25, no random utility labels on the production path).

| Phase | Status | Evidence |
|---|---|---|
| 0 skeleton | done | `pyproject.toml`, configs, CLI, docs |
| 1 agent + benchmark | **verified** | pytest 23; 50-task LLM run **86%** EM; 148 tool calls |
| 2 persistent memory | **verified** | MiniLM retrieve hit on stored episode; `inspect-agent` |
| 3 diagnosis | **verified** | 100 real trajectories → `outputs/phase1_static/capability_profile_100.json` |
| 4 passive OEL | implemented | verifier + abstractor + memory/LoRA carriers |
| 5 active acquisition | implemented | frontier / uncertainty / replay / generation |
| 6 multi-carrier | implemented | 6 carriers + heuristic router + harness allow-list |
| 7 utility | implemented | A/B intervention evaluator |
| 8–10 learned policy | implemented | joint MLP + `PolicyTrainer` on realized utility |
| 11 evolve loop | implemented | `EvolveLoop` + `bash scripts/run_expolicy.sh` |

Paper-scale (5×200×3 seeds) uses the same entry points with larger `budget.max_tasks`.
