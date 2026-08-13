# Phases 4–6 report

## Phase 4 — Passive OEL

- Experience extraction → `ExperienceVerifier` → `ExperienceAbstractor` → heuristic router
- Memory baseline: episodic/semantic writes
- Weight baseline: PEFT LoRA SFT (`WeightUpdater`) on verified successes only
- Config: `configs/experiment/passive_oel.yaml`

## Phase 5 — Active acquisition

Policies: passive, uncertainty (real eval/model uncertainty), failure replay, frontier, reward-seeking, learned.
Active generation: hard variants with validity checker; **real** branch re-execution (not imagined); failure-seeking near the frontier.

```bash
pytest tests/unit/test_policies.py -q
python -m expolicy.cli.run --config configs/experiment/active_only.yaml --evolve
```

## Phase 6 — Multi-carrier assimilation

Carriers: discard, episodic, semantic, skill (executable), harness (allow-listed + rollback), weight.
Heuristic routing baseline + learned assimilation head.
Illegal harness keys (`shell_command`, …) raise `HarnessPatchRejected`.

## Known limitations

Full LoRA internalization is expensive; default evolve budgets keep LoRA steps small but real.
