# Reproduction

## Software

```bash
bash scripts/setup.sh
source .venv/bin/activate
pytest -q
```

Record in every run `manifest.json`: seed, config snapshot, git commit, model path.

## Development suite (this machine)

```bash
export CUDA_VISIBLE_DEVICES=5   # free H20
python -m expolicy.cli.main setup
bash scripts/run_baseline.sh 60
```

Expected artifacts: `outputs/static_base/{metrics.json,trajectories.jsonl,experiences.jsonl,events.jsonl}`.

## Paper suite

1. Set `budget.max_tasks` to cover ≥200 tasks/stage in `configs/experiment/*.yaml`.
2. Run three seeds `{13, 17, 23}` for B0–B7.
3. Collect `utility_records.jsonl` and train `ExperiencePolicy` (`scripts/train_experience_policy.sh`).
4. Build carrier intervention matrices with `expolicy.evaluation.intervention.carrier_matrix` on a small probe (costly; do not skip real eval).
5. Report learning / retention / utility / cost curves plus acquisition vs reward-seeking histograms.

## What not to do

- Do not stub BM25 hits.
- Do not replace a failed LLM call with a heuristic answer.
- Do not train the policy on gold answers or future-split scores.
- Do not commit unverified harness patches.
