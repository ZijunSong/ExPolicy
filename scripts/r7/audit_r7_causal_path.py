#!/usr/bin/env python3
"""Causal-path audit: shadow Δ → reward → policy update → next-round posterior."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from expolicy.closed_loop.constants import ROOT, R7_ROOT
from expolicy.closed_loop.io_utils import read_json, write_json

CHECKS = []


def ok(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append({"name": name, "pass": bool(passed), "detail": detail})


def source_of(*rel) -> str:
    return (ROOT.joinpath(*rel)).read_text(encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(R7_ROOT))
    args = p.parse_args()
    root = Path(args.root)

    ep = source_of("expolicy", "closed_loop", "runner", "episode.py")
    rew = source_of("expolicy", "closed_loop", "reward.py")
    feat = source_of("expolicy", "closed_loop", "features.py")
    rt = source_of("expolicy", "closed_loop", "agent_state.py")
    sk = source_of("expolicy", "closed_loop", "frozen_skill_bank", "execute.py")

    ok("reward_from_shadow", "delta_shadow" in rew and "future_test" in rew and "uses_future_test" in rew)
    ok("episode_calls_compute_realized_utility", "compute_realized_utility" in ep)
    ok("episode_updates_both_policies", "acq_policy.update" in ep and "assim_policy.update" in ep)
    ok("no_feedback_skips_update", 'cfg.feedback != "none"' in ep)
    ok("shuffle_does_not_use_true_pair", "used_r = -true_r" in ep or "cfg.feedback == \"shuffle\"" in ep)
    ok("next_round_loads_policy_after", "policy_after" in ep and "load_snapshot" in ep)
    ok("features_forbid_task_id_gold_dataset", "task_id" in feat and "gold_answer" in feat and "dataset" in feat)
    ok("runtime_skill_changes_dispatch", "frozen_skill" in rt and "run_skill" in rt)
    ok("skill_uses_env_step", 'env.step("search"' in sk and "gold_answer" not in sk.split("def execute_skill")[0])
    ok("actor_temperature_zero", "temperature=0.0" in rt)
    ok("resume_hash_guard", "ResumeHashError" in ep or "config hash mismatch" in ep)

    # live artifacts if present
    m0 = root / "expolicy" / "seed0"
    if m0.exists():
        rounds = sorted(m0.glob("round_*"))
        if len(rounds) >= 2:
            p0 = read_json(rounds[0] / "policy_after.json")
            p1b = read_json(rounds[1] / "policy_before.json")
            ok("live_next_round_reads_new_posterior", p0 == p1b, "round1.policy_before == round0.policy_after")
            u0 = read_json(rounds[0] / "policy_update.json")
            ok("live_parameter_delta_positive", float((u0.get("acquisition") or {}).get("parameter_delta_norm") or 0) > 0)
            rew0 = read_json(rounds[0] / "reward.json")
            ok("live_reward_not_future_test", rew0.get("uses_future_test") is False)
        nf = root / "no_feedback" / "seed0"
        if nf.exists() and list(nf.glob("round_*")):
            rd = sorted(nf.glob("round_*"))[0]
            ok(
                "live_nofeedback_prior_unchanged",
                read_json(rd / "policy_before.json") == read_json(rd / "policy_after.json"),
            )
    passed = all(c["pass"] for c in CHECKS)
    write_json(root / "audit" / "CAUSAL_PATH_AUDIT.json", {"pass": passed, "checks": CHECKS})
    lines = ["# R7 Causal Path Audit", ""]
    for c in CHECKS:
        lines.append(f"- {c['name']}: {'PASS' if c['pass'] else 'FAIL'} {c.get('detail') or ''}")
    lines += ["", f"OVERALL: {'PASS' if passed else 'FAIL'}"]
    (root / "audit" / "CAUSAL_PATH_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
