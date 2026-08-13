# ExPolicy

**Learning What to Experience and How to Assimilate It for Self-Evolving Agents**

ExPolicy treats *experience itself* as a decision object. An agent learns a policy

\[
\pi_\phi(q_t, r_t \mid \Xi_t, C_t)
\]

that chooses **what to experience next** (\(q_t\)) and **how to assimilate it** (\(r_t\)) into a persistent state

\[
\Xi_t = (\theta_t, M_t^e, M_t^s, K_t, H_t)
\]

(weights, episodic memory, semantic memory, skills, harness). Supervision is **realized future learning utility**, not LLM self-scores and not immediate episode reward.

This repository is a runnable research codebase: real search over a local corpus, real LLM trajectories, real verification/abstraction/routing, real LoRA updates, and a trainable Experience Policy.

---

## 1. Project overview

Domain (MVP → paper): **Search / research agent** on the **Helion Archive**, a deterministic, fully searchable closed world. Retrieval is BM25 over real documents (`search` / `open` / `find`). Answers are never injected into the agent.

Continual protocol: five capability stages with isolated `deployment / acquisition / validation / future / retention` splits.

## 2. Research motivation

Standard experiential learning assumes deployment trajectories are the right things to learn from, and that a single carrier (memory *or* weights) is enough. ExPolicy tests the opposite: the highest-utility experiences are often **frontier failures**, and different experiences belong on different carriers.

## 3. Architecture

See [`docs/architecture.md`](docs/architecture.md). Loop:

```
diagnose capability → acquire experience → interact → verify → abstract
→ route (discard | episodic | semantic | skill | harness | weight)
→ measure realized utility → update Experience Policy
```

## 4. Installation

Python 3.11+. On this machine the `bishop` conda env already has `torch` / `transformers` / `peft`.

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

`pip install -e .` installs the package. GPU extras live in `.[train]` / `.[serve]`.

## 5. Model setup

Default instruct model: `/data/ppnm/models/Qwen2.5-3B-Instruct`.

Backends (no silent fallback to a fake LM):

| `model.backend` | Use |
|---|---|
| `transformers` | local HF checkpoint (`model.path`) |
| `vllm` | in-process vLLM |
| `openai_compat` | vLLM HTTP server (`model.base_url` + `model.name`) |

Embedder: `/data/ppnm/models/all-MiniLM-L6-v2`.

Override with `EXPOLICY_MODEL` / `EXPOLICY_API_BASE`.

## 6. Environment setup

```bash
python -m expolicy.cli.main setup --seed 13 --world-path data/world/helion.json
# or
python scripts/build_benchmark.py
```

This writes a reproducible corpus + task stream. Search results come from BM25, never from a stub.

## 7. Benchmark setup

See [`docs/benchmark.md`](docs/benchmark.md). Capabilities:

`single_hop`, `multi_hop`, `entity_disambiguation`, `temporal_reasoning`, `conflicting_evidence`, `source_reliability`, `long_context`, `query_decomposition`, `cross_document_synthesis`, `failure_recovery`.

Held-out `future` ids are hash-split and guarded against leakage.

## 8. Running the static baseline (B0)

```bash
bash scripts/run_baseline.sh          # 60 validation tasks, GPU
# or
python -m expolicy.cli.run --config configs/experiment/base.yaml
```

## 9. Running passive OEL (B2)

```bash
python -m expolicy.cli.run --config configs/experiment/passive_oel.yaml --evolve
```

## 10. Running active-only (B3)

```bash
python -m expolicy.cli.run --config configs/experiment/active_only.yaml --evolve
```

## 11. Running adaptive-only (B4)

```bash
python -m expolicy.cli.run --config configs/experiment/adaptive_only.yaml --evolve
```

## 12. Running full ExPolicy (B7)

```bash
bash scripts/run_expolicy.sh
# or
python -m expolicy.cli.main evolve --config configs/experiment/expolicy.yaml
```

## 13. Training the Experience Policy

Utility records are written by the evolve loop (`outputs/<run>/utility_records.jsonl`) from **before/after probe evaluation**, not from verbalized scores.

```bash
bash scripts/train_experience_policy.sh outputs/full_expolicy/utility_records.jsonl
python -m expolicy.cli.train --buffer outputs/full_expolicy/utility_records.jsonl --out outputs/experience_policy.npz
```

## 14. Evaluation

```bash
bash scripts/evaluate_all.sh outputs/static_base/agent_v000
expolicy evaluate outputs/agent_v010 --config configs/experiment/base.yaml
```

Metrics: current accuracy, future generalization, retention, learning/cost efficiency, realized utility, experience hit rate, routing regret, acquisition regret.

## 15. Reproducing paper experiments

See [`docs/reproduction.md`](docs/reproduction.md). Paper-scale: 5 stages × ≥200 tasks × 3 seeds. Development suite uses the same code with smaller `budget.max_tasks`.

Baselines B0–B7 and ablations are YAML configs under `configs/experiment/`.

## 16. Output structure

```
outputs/<run_id>/
  config.yaml
  manifest.json
  metrics.json
  trajectories.jsonl
  experiences.jsonl
  events.jsonl
  utility_records.jsonl
  agent_v000/ … agent_vNNN/
    manifest.json
    model_adapter/
    episodic_memory/
    semantic_memory/
    skills/
    harness/
    capability_profile.json
    metrics.json
    lineage.json
```

```bash
expolicy inspect-agent outputs/<run>/agent_v003
expolicy inspect-experience <experience_id> --run-dir outputs/<run>
```

## 17. Troubleshooting

| Symptom | Recovery |
|---|---|
| `BackendUnavailableError` | set `model.path` to a local instruct checkpoint or start vLLM |
| `model path does not exist` | check `/data/ppnm/models/Qwen2.5-3B-Instruct` |
| embedder missing | place MiniLM at `model.embedder_path` |
| `TrainingDivergenceError` | inspect `train_metrics.json`; do **not** skip the optimizer step |
| `LeakageError` | acquisition/eval used future ids — check splits |
| `HarnessPatchRejected` | illegal key or failed regression probe; harness is not auto-committed |
| CUDA OOM | do **not** fall back to a random policy; lower batch / use 3B / set `CUDA_VISIBLE_DEVICES` |

Failures are logged as `RUN_FAILED`. They are never rewritten as success.

---

## Tests

```bash
pytest -q
ruff check expolicy tests
```

GPU tests (real LM) are marked `@pytest.mark.gpu`.

## License

Apache-2.0
