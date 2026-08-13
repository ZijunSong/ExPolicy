# Phase 11 report

## Implemented

`EvolveLoop`:

```
while budget_remaining:
    diagnose → acquire → execute → verify → abstract → assimilate
    → versioned checkpoint → realized utility → policy.learn → save
```

CLI:

```
expolicy setup | run-agent | collect-experience | acquire | assimilate
expolicy evolve | evaluate | inspect-agent | inspect-experience | train-policy
```

Scripts: `setup.sh`, `run_baseline.sh`, `run_expolicy.sh`, `train_experience_policy.sh`, `evaluate_all.sh`.

## Commands

```bash
bash scripts/run_expolicy.sh
```

Keep `budget.max_tasks` small for development; paper scale is 5 stages × 200 tasks × 3 seeds (same code).

## Known limitations

Intervention probes add LLM cost; `probe_limit` defaults to 8. Do not replace probes with predicted scores.
