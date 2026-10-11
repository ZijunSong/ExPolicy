# THIRD_PARTY_PATCH_AUDIT.md

Date: 2026-08-21  
Scope: R8-B MemRL / SkillOpt official checkouts used as assimilator backends, **not** as ExPolicy novelty.

## Repositories

| backend | local path | official URL | commit | license |
|---|---|---|---|---|
| MemRL | `/data/ppnm/third_party/memrl` | https://github.com/MemTensor/MemRL | `c1b322ca43de36ddf64c6712f89d0095bfc35ce0` | MIT (Copyright 2026 jiaqian) |
| SkillOpt | `/data/ppnm/third_party/skillopt` | https://github.com/microsoft/SkillOpt | `3c8873f016397817dcd40c3e5436d92fe19372b8` | MIT (Copyright 2026 Microsoft Corporation) |

Clone notes:

- MemRL cloned via `https://ghproxy.net/https://github.com/MemTensor/MemRL.git` after GitHub HTTPS timeout.
- SkillOpt cloned via `https://ghproxy.net/https://github.com/microsoft/SkillOpt.git` (direct GitHub clone stalled).
- `git remote` currently points at the ghproxy mirror; canonical upstream URLs are recorded in this table.

ALFWorld payload:

- Official game files: `/data/ppnm/data/alfworld` (`json_2.1.1`, logic).
- MemRL expected relative path: `data/alfworld` → symlink ` /data/ppnm/third_party/memrl/data/alfworld` → `/data/ppnm/data/alfworld`.
- SkillOpt expects `$ALFWORLD_DATA` pointing at the same root; gamefiles in manifests are relative to that root.

## Source-tree patches

**None.** Third-party Python sources are used unmodified.

All R8 changes live in:

- `/data/ppnm/expolicy/scripts/r8/*`
- `/data/ppnm/expolicy/expolicy/r8/*`
- generated configs under `outputs/r8_backend_validation/` (not in the vendor trees)

## Runtime / config overrides (not source edits)

These are required to run the official entry points against frozen **Qwen2.5-3B-Instruct**.

### Shared foundation model

- Local weights: `/data/ppnm/models/Qwen2.5-3B-Instruct`
- Served with vLLM OpenAI-compatible API (`scripts/r8/serve_qwen_vllm.sh`) on `127.0.0.1:18090`
- Actor remains frozen (no LoRA / no weight update)

### MemRL

Official entry: `python run/run_alfworld.py --config configs/rl_alf_config.yaml`

R8 uses a **generated** config `outputs/r8_backend_validation/configs/memrl_*.yaml` that only changes:

- `llm.base_url` / `llm.model` → local vLLM Qwen2.5-3B-Instruct
- `llm.api_key` = `EMPTY`
- `embedding.base_url` → local padded MiniLM server (`scripts/r8/serve_local_embedder.py`, port 18091)
- `embedding.model` advertised as `text-embedding-3-large` so MemRL keeps its official 3072-d Qdrant collection; the server **zero-pads** MiniLM-L6-v2 (384-d) to 3072-d. Cosine similarity of padded vectors equals the unpadded cosine. This avoids editing `MemoryService` hardcoded `vector_dimension: 3072`.
- `environment.alfworld_config_path` → generated yaml whose `dataset.data_path` is the **acquisition_pool** symlink view (never `future_test`)
- `experiment.few_shot_path` → `/data/ppnm/third_party/memrl/configs/alfworld/alfworld_examples.json`
- `experiment.num_sections`, `batch_size`, `dataset_ratio` recorded per job
- `memory.mos_config_path` left unused; MemoryService still synthesizes MOS/Qdrant under the job output dir via cwd isolation

No MemRL algorithm flags (`build_strategy`, `update_strategy`, `alpha`, `tau`, …) are changed from the official ALFWorld yaml defaults.

### SkillOpt

Official entry: `python scripts/train.py --config configs/alfworld/default.yaml`

R8 CLI overrides only:

- `--backend openai_compatible`
- `--optimizer_model` / `--target_model` = `Qwen2.5-3B-Instruct`
- `OPENAI_COMPATIBLE_BASE_URL=http://127.0.0.1:18090/v1`
- `--split_dir` = frozen R8 SkillOpt manifest (`train=acquisition_pool`, `val=assim_validation`, `test=shadow_reward_probe`)
- `ALFWORLD_DATA=/data/ppnm/data/alfworld`

**Never** pass `future_test` as SkillOpt `val` or `test` during training.

Official skill init remains `skillopt/envs/alfworld/skills/initial.md`.
Validation gate (`evaluation.use_gate: true`) is left on.

### Isolation

Each `(condition, seed)` job has its own:

- cwd / `out_root`
- MemRL Qdrant / mem-cube directory
- SkillOpt `best_skill.md` version directory

Adapters copy these directories for counterfactual clones; they do not share live handles.

## What is explicitly not a patch / not novelty

- Using MemRL episodic memory Q-update
- Using SkillOpt held-out-gated skill edits
- Serving Qwen through vLLM
- Restricting official runners to the frozen R8 split

ExPolicy novelty, if Gate H later passes, is the joint acquisition × assimilator policy — not these backends.
