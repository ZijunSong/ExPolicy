# Phases 7–10 report

## Phase 7 — Utility evaluation

`UtilityEvaluator` clones agent state, applies one experience, evaluates future + retention probes, writes before/after/delta. `compute_utility` implements ΔC / (C_acq+C_learn) − λI from **realized** numbers.

## Phase 8 — Learned assimilation

Trained on `Experience × Carrier` intervention outcomes (`carrier_matrix`). Compared conceptually to random / heuristic / learned / oracle hindsight (oracle is analysis-only).

## Phase 9 — Learned acquisition

Candidate ranker fitted to realized future utility. Features exclude gold answers (`test_policy_features_exclude_gold_answer`).

## Phase 10 — Joint Experience Policy

Shared MLP encoder with acquisition, assimilation, and utility heads. Loss:

`L = L_acq + α L_asm + β L_utility` with cost/interference folded into realized utility targets.

```bash
python -m expolicy.cli.train --buffer outputs/<run>/utility_records.jsonl --out outputs/experience_policy.npz
```

## Next phase

Phase 11: online loop that trains the policy from its own interventions.
