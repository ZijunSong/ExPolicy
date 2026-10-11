#!/usr/bin/env python3
"""R8-A: R7 structural identifiability audit. Does not train a new controller.

Phases:
  cases     — read-only case traces (CPU)
  worker    — GPU jobs: 18x4 counterfactual matrix, 8-mask ceiling, sequential oracle
  summarize — S1/S2/S3 + R8_A gate (CPU after GPU artifacts exist)
  all       — cases, then summarize if GPU artifacts complete
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from expolicy.closed_loop.constants import (
    ACTIVATE_EXISTING_SKILL,
    ASSIM_ACTIONS,
    C1,
    C2,
    C3,
    CAPS,
    DISCARD,
    EPISODIC_MEMORY,
    R7_ROOT,
    ROOT,
    SEMANTIC_MEMORY,
)
from expolicy.closed_loop.experience import ExperienceBatchSummary
from expolicy.closed_loop.io_utils import read_json, write_json
from expolicy.closed_loop.reward import compute_realized_utility
from expolicy.closed_loop.state import ExPolicyState
from expolicy.core.experience import Experience
from expolicy.experiments.r_common import apply_gpu

OUT = ROOT / "outputs" / "r8_preflight"
R7_GATE = R7_ROOT / "R7_GATE_DECISION.json"
CONDITION = "expolicy"
UNIQUE_EPS = 0.005  # filter R7 shadow-eval noise (~0.01 on DISCARD)

MASKS: list[tuple[str, tuple[str, ...]]] = [
    ("none", ()),
    ("C1", (C1,)),
    ("C2", (C2,)),
    ("C3", (C3,)),
    ("C1+C2", (C1, C2)),
    ("C1+C3", (C1, C3)),
    ("C2+C3", (C2, C3)),
    ("C1+C2+C3", (C1, C2, C3)),
]


def now() -> str:
    return time.strftime("%F %T")


def heartbeat(out: Path, **payload: Any) -> None:
    payload = {"updated_at": time.time(), "pid": os.getpid(), **payload}
    write_json(out / "STATUS_LIVE.json", payload)
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    if gpu:
        write_json(OUT / "logs" / f"worker_gpu{gpu}.live.json", payload)


def m0_seed_dir(seed: int) -> Path:
    return R7_ROOT / CONDITION / f"seed{seed}"


def m0_round(seed: int, t: int) -> Path:
    return m0_seed_dir(seed) / f"round_{t:02d}"


def job_dir(kind: str, job_id: str) -> Path:
    return OUT / "jobs" / kind / job_id


def load_experiences(path: Path) -> list[Experience]:
    rows: list[Experience] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(Experience.model_validate(json.loads(line)))
    return rows


def frozen_costs(cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "retention_weight": float(cfg.get("retention_weight", 0.5)),
        "token_cost_per_million": float(cfg.get("token_cost_per_million", 0.01)),
        "tool_cost_per_100": float(cfg.get("tool_cost_per_100", 0.005)),
    }


def utility_from_parts(
    *,
    shadow_before: float,
    shadow_after: float,
    retention_before: float,
    retention_after: float,
    token_cost: int,
    tool_cost: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    costs = frozen_costs(cfg)
    blob = compute_realized_utility(
        shadow_macro_before=shadow_before,
        shadow_macro_after=shadow_after,
        retention_before=retention_before,
        retention_after=retention_after,
        token_cost=token_cost,
        tool_cost=tool_cost,
        **costs,
    )
    blob["U"] = blob["reward"]
    return blob


# ---------------------------------------------------------------------------
# Case audit (CPU)
# ---------------------------------------------------------------------------


def _round_row(condition: str, seed: int, t: int) -> dict[str, Any]:
    rd = R7_ROOT / condition / f"seed{seed}" / f"round_{t:02d}"
    state_b = read_json(rd / "state_before.json")
    state_a = read_json(rd / "state_after.json")
    acq = read_json(rd / "acquisition_decision.json")
    assim = read_json(rd / "assimilation_decision.json")
    summary = read_json(rd / "experience_summary.json")
    reward = read_json(rd / "reward.json")
    apply_info = read_json(rd / "apply.json") if (rd / "apply.json").exists() else {}
    upd = read_json(rd / "policy_update.json") if (rd / "policy_update.json").exists() else {}
    shadow = read_json(rd / "shadow_summary.json")
    pol_b = read_json(rd / "policy_before.json")
    pol_a = read_json(rd / "policy_after.json")
    batch_ids = []
    if (rd / "experience_batch.jsonl").exists():
        for line in (rd / "experience_batch.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                batch_ids.append(json.loads(line).get("task_id"))
    return {
        "condition": condition,
        "seed": seed,
        "round": t,
        "round_dir": str(rd),
        "acquired_capability": acq.get("capability"),
        "acq_scores": acq.get("scores"),
        "acq_sampled": acq.get("sampled_values"),
        "acq_features_chosen": (acq.get("features") or {}).get(acq.get("capability")),
        "assim_action": assim.get("action"),
        "assim_scores": assim.get("scores"),
        "assim_sampled": assim.get("sampled_values"),
        "assim_valid": assim.get("valid_actions"),
        "assim_features_chosen": (assim.get("features") or {}).get(assim.get("action")),
        "batch_success_rate": summary.get("success_rate"),
        "batch_n": summary.get("n"),
        "batch_task_ids": batch_ids,
        "apply": apply_info,
        "reward": reward.get("reward"),
        "delta_shadow": reward.get("delta_shadow"),
        "delta_retention": reward.get("delta_retention"),
        "shadow_before": reward.get("shadow_macro_before"),
        "shadow_after": reward.get("shadow_macro_after"),
        "retention_before": reward.get("retention_before"),
        "retention_after": reward.get("retention_after"),
        "shadow_em_by_cap_after": shadow.get("em_by_capability"),
        "dispatch_after": shadow.get("dispatch_counts"),
        "skills_before": state_b.get("active_skills"),
        "skills_after": state_a.get("active_skills"),
        "epi_before": state_b.get("episodic_memory_size"),
        "epi_after": state_a.get("episodic_memory_size"),
        "sem_before": state_b.get("semantic_memory_size"),
        "sem_after": state_a.get("semantic_memory_size"),
        "feedback": upd.get("feedback"),
        "acq_delta_norm": (upd.get("acquisition") or {}).get("parameter_delta_norm"),
        "assim_delta_norm": (upd.get("assimilation") or {}).get("parameter_delta_norm"),
        "acq_skipped": (upd.get("acquisition") or {}).get("skipped"),
        "assim_skipped": (upd.get("assimilation") or {}).get("skipped"),
        "policy_acq_n_before": _n_updates(pol_b.get("acquisition")),
        "policy_acq_n_after": _n_updates(pol_a.get("acquisition")),
    }


def _n_updates(snap: Any) -> int:
    if not isinstance(snap, dict):
        return 0
    heads = snap.get("heads") or {}
    return int(sum(int((h or {}).get("n_updates") or 0) for h in heads.values()))


def write_case_audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        for t in range(6):
            traces.append(_round_row("expolicy", seed, t))
    for t in range(6):
        traces.append(_round_row("no_feedback", 1, t))
    write_json(OUT / "R7_CASE_TRACE.json", traces)
    with (OUT / "R7_CASE_TRACE.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "condition",
            "seed",
            "round",
            "acquired_capability",
            "assim_action",
            "batch_success_rate",
            "reward",
            "delta_shadow",
            "shadow_before",
            "shadow_after",
            "acq_delta_norm",
            "assim_delta_norm",
            "feedback",
        ]
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in traces:
            w.writerow({k: row.get(k) for k in fields})

    m0s1_r2 = next(r for r in traces if r["condition"] == "expolicy" and r["seed"] == 1 and r["round"] == 2)
    m0s2_c2 = [r for r in traces if r["condition"] == "expolicy" and r["seed"] == 2 and r["acquired_capability"] == C2]
    m0s0 = [r for r in traces if r["condition"] == "expolicy" and r["seed"] == 0]
    nf_s1 = [r for r in traces if r["condition"] == "no_feedback" and r["seed"] == 1]
    m0s0_final = read_json(m0_seed_dir(0) / "final_metrics.json")
    nf_final = read_json(R7_ROOT / "no_feedback" / "seed1" / "final_metrics.json")
    gate = read_json(R7_GATE) if R7_GATE.exists() else {}

    md = []
    md.append("# R7 Case Audit (R8-A §3.1)")
    md.append("")
    md.append(f"- Written: {now()}")
    md.append("- Source: `outputs/r7_closed_loop` round-level JSON (not summary CSV only).")
    md.append("- Cost λ frozen: `retention_weight=0.5`, `token_cost_per_million=0.01`, `tool_cost_per_100=0.005`.")
    md.append("")
    md.append("## Naming")
    md.append("")
    md.append("| Doc | Code `condition` |")
    md.append("|---|---|")
    md.append("| M0 | `expolicy` |")
    md.append("| A1 NoFeedback | `no_feedback` |")
    md.append("| C1 / C2 / C3 | `bridge_decomposition` / `comparison_alignment` / `long_chain_search` |")
    md.append("")
    md.append("## Case 1 — M0 seed1 round_02: C2 + ACTIVATE, shadow Δ≈+0.105")
    md.append("")
    md.append(f"- Path: `{m0s1_r2['round_dir']}`")
    md.append(f"- Acquisition: `{m0s1_r2['acquired_capability']}`")
    md.append(f"- Thompson sampled acq: `{json.dumps(m0s1_r2['acq_sampled'], ensure_ascii=False)}`")
    md.append(f"- Assimilation: `{m0s1_r2['assim_action']}`")
    md.append(f"- Thompson sampled assim: `{json.dumps(m0s1_r2['assim_sampled'], ensure_ascii=False)}`")
    md.append(f"- Batch: n={m0s1_r2['batch_n']}, success_rate={m0s1_r2['batch_success_rate']}")
    md.append(f"- Experience IDs: `{m0s1_r2['batch_task_ids']}`")
    md.append(f"- Skills before: `{m0s1_r2['skills_before']}`")
    md.append(f"- Skills after: `{m0s1_r2['skills_after']}`")
    md.append(f"- Apply: `{json.dumps(m0s1_r2['apply'], ensure_ascii=False)}`")
    md.append(f"- Shadow {m0s1_r2['shadow_before']:.6f} → {m0s1_r2['shadow_after']:.6f} (Δ={m0s1_r2['delta_shadow']:.6f})")
    md.append(f"- Realized utility (frozen λ): **{m0s1_r2['reward']:.6f}**")
    md.append(f"- Dispatch after: `{m0s1_r2['dispatch_after']}`")
    md.append(f"- Posterior update: acq Δ={m0s1_r2['acq_delta_norm']}, assim Δ={m0s1_r2['assim_delta_norm']}")
    md.append("- Reading: controller *did* see a large positive shadow signal and updated. Feature vector at decision time has `skill_active=0` and failure-heavy batch stats; ACTIVATE won via Thompson draw, not because mean scores already preferred it (`scores.ACTIVATE=0.0`).")
    md.append("")
    md.append("## Case 2 — M0 seed2: acquired C2 but continuously DISCARD")
    md.append("")
    md.append("| round | cap | action | Δshadow | reward | skills_after |")
    md.append("|---|---|---|---|---|---|")
    for r in m0s2_c2:
        md.append(
            f"| {r['round']:02d} | {r['acquired_capability']} | {r['assim_action']} | "
            f"{r['delta_shadow']:.4f} | {r['reward']:.4f} | `{r['skills_after']}` |"
        )
    md.append("")
    md.append(f"- C2 rounds: {len(m0s2_c2)}; all DISCARD; C2 skill never activated.")
    md.append(f"- Final M0 seed2 macro = {read_json(m0_seed_dir(2) / 'final_metrics.json')['macro']:.4f}")
    md.append("- Reading: the environment *would* have paid ~+0.10 for ACTIVATE (Case 1). Here the same capability was sampled three times and discarded. This is **saw-the-family-but-policy-did-not-use-it**, not missing C2 in the stream.")
    md.append("")
    md.append("## Case 3 — M0 seed0: activated C3, still low final")
    md.append("")
    md.append("| round | cap | action | Δshadow | reward | skills_after |")
    md.append("|---|---|---|---|---|---|")
    for r in m0s0:
        md.append(
            f"| {r['round']:02d} | {r['acquired_capability']} | {r['assim_action']} | "
            f"{r['delta_shadow']:.4f} | {r['reward']:.4f} | `{r['skills_after']}` |"
        )
    md.append("")
    md.append(f"- Final macro = **{m0s0_final['macro']:.4f}**")
    md.append(f"- Final skills: `{m0s0_final['active_skills']}`")
    md.append(f"- Final EM by cap: `{m0s0_final['em_by_capability']}`")
    md.append("- C3 `long_chain_search` future EM stays **0.0** after activation. C2 was never acquired. Activating the wrong frozen skill does not buy the C2-only ceiling.")
    md.append("")
    md.append("## Case 4 — A1 NoFeedback seed1: frozen posterior, random C2 hit, final≈0.089")
    md.append("")
    md.append("| round | cap | action | Δshadow | acq_n_before→after | skipped |")
    md.append("|---|---|---|---|---|---|")
    for r in nf_s1:
        md.append(
            f"| {r['round']:02d} | {r['acquired_capability']} | {r['assim_action']} | "
            f"{r['delta_shadow']:.4f} | {r['policy_acq_n_before']}→{r['policy_acq_n_after']} | "
            f"acq={r['acq_skipped']} assim={r['assim_skipped']} |"
        )
    md.append("")
    md.append(f"- Final macro = **{nf_final['macro']:.4f}**")
    md.append(f"- Final skills: `{nf_final['active_skills']}`")
    md.append("- Posterior `n_updates` never increases. Feedback=`none`. The C2 activation still moved shadow/future because **state mutation is not gated on learning**. This is lucky prior sampling, not credit assignment.")
    md.append("")
    md.append("## Cross-case implications (before GPU counterfactuals)")
    md.append("")
    md.append("- Future utility is dominated by whether `comparison_alignment` is activated.")
    md.append(f"- R7 strongest non-oracle = `{gate.get('strongest_non_oracle')}` (CountOnly = all_skills).")
    md.append("- M0 mean 0.0474 < CountOnly 0.0911. Seed2 had C2 in the stream and still lost it.")
    md.append("- Assimilation regret is mixed: seed2 is **policy failed to use a visible C2 batch**; seed0 is **acquisition never brought C2**, so assimilation had no C2 lever.")
    md.append("")
    (OUT / "R7_CASE_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'R7_CASE_AUDIT.md'}", flush=True)


# ---------------------------------------------------------------------------
# GPU jobs
# ---------------------------------------------------------------------------


def emit_jobs(smoke: bool = False) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        for t in range(6):
            rd = m0_round(seed, t)
            assim = read_json(rd / "assimilation_decision.json")
            original = assim["action"]
            valid = list(assim.get("valid_actions") or ASSIM_ACTIONS)
            for action in ASSIM_ACTIONS:
                jobs.append(
                    {
                        "kind": "cf",
                        "job_id": f"cf_s{seed}_r{t:02d}_{action}",
                        "seed": seed,
                        "round": t,
                        "action": action,
                        "original": original,
                        "valid": action in valid,
                        "smoke": smoke,
                    }
                )
        jobs.append({"kind": "oracle", "job_id": f"oracle_s{seed}", "seed": seed, "smoke": smoke})
    for name, caps in MASKS:
        jobs.append({"kind": "ceiling", "job_id": f"ceiling_{name}", "mask": name, "caps": list(caps), "smoke": smoke})
    qdir = OUT / "queue"
    qdir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        write_json(qdir / f"{job['job_id']}.json", job)
    write_json(OUT / "JOBS.json", jobs)
    return jobs


def _claimed_lock(job_id: str) -> FileLock:
    d = OUT / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return FileLock(str(d / f"{job_id}.lock"), timeout=0)


def claim_job(kinds: tuple[str, ...]) -> dict[str, Any] | None:
    qdir = OUT / "queue"
    if not qdir.exists():
        return None

    def _prio(job: dict[str, Any]) -> tuple[int, str]:
        kind = job.get("kind")
        if kind == "cf":
            if not job.get("valid", True):
                return (0, job["job_id"])
            if job.get("action") == DISCARD or job.get("action") == job.get("original"):
                return (1, job["job_id"])
            return (2, job["job_id"])
        if kind == "ceiling":
            if job.get("mask") in {"none", "C1+C2+C3"}:
                return (1, job["job_id"])
            return (3, job["job_id"])
        if kind == "oracle":
            return (4, job["job_id"])
        return (9, job.get("job_id") or "")

    candidates: list[dict[str, Any]] = []
    for path in qdir.glob("*.json"):
        try:
            job = read_json(path)
        except Exception:
            continue
        if job.get("kind") not in kinds:
            continue
        dest = job_dir(job["kind"], job["job_id"])
        if (dest / "DONE").exists():
            continue
        candidates.append(job)
    candidates.sort(key=_prio)
    for job in candidates:
        dest = job_dir(job["kind"], job["job_id"])
        if (dest / "DONE").exists():
            continue
        lock = _claimed_lock(job["job_id"])
        try:
            lock.acquire()
        except Timeout:
            continue
        if (dest / "DONE").exists():
            lock.release()
            continue
        job["_lock"] = lock
        return job
    return None


def load_split_and_cfg():
    from expolicy.closed_loop.runner.config import R7Config
    from expolicy.closed_loop.split_data import load_split, split_tasks

    cfg_raw = read_json(m0_seed_dir(0) / "config.json")
    cfg = R7Config.from_dict(cfg_raw)
    cfg.smoke = False
    split = load_split()
    cal = split_tasks(split, "calibration")
    acq = split_tasks(split, "acquisition_pool")
    shadow = split_tasks(split, "shadow_reward_probe")
    retention = split_tasks(split, "retention_probe")
    future = split_tasks(split, "future_test")
    return cfg, cfg_raw, cal + acq + shadow + retention + future, shadow, retention, future


def build_fresh_runtime(cfg, tasks, root: Path, state: ExPolicyState, model, client):
    from expolicy.closed_loop.runner.episode import build_runtime

    return build_runtime(cfg, tasks, root, state, model=model, client=client)


def replay_memory(runtime, seed: int, upto_round: int, model) -> None:
    from expolicy.closed_loop.assimilation.episodic import apply_episodic
    from expolicy.closed_loop.assimilation.semantic import apply_semantic

    dummy = runtime.state.copy()
    for t in range(upto_round):
        rd = m0_round(seed, t)
        action = read_json(rd / "assimilation_decision.json")["action"]
        if action not in {EPISODIC_MEMORY, SEMANTIC_MEMORY}:
            continue
        exps = load_experiences(rd / "experience_batch.jsonl")
        summary = ExperienceBatchSummary.from_dict(read_json(rd / "experience_summary.json"))
        if action == EPISODIC_MEMORY:
            apply_episodic(runtime.episodic_dir, dummy, exps)
        else:
            apply_semantic(runtime.semantic_dir, dummy, summary, exps, model=model)


def maybe_trim(tasks: list, smoke: bool, n: int) -> list:
    if not smoke:
        return tasks
    return tasks[: max(2, n)]


def run_cf_job(job: dict[str, Any], *, cfg, cfg_raw, all_tasks, shadow_tasks, retention_tasks, model, client) -> None:
    seed = int(job["seed"])
    t = int(job["round"])
    action = job["action"]
    dest = job_dir("cf", job["job_id"])
    dest.mkdir(parents=True, exist_ok=True)
    heartbeat(dest, phase="start", **{k: job[k] for k in ("kind", "job_id", "seed", "round", "action")})
    rd = m0_round(seed, t)
    original = read_json(rd / "assimilation_decision.json")["action"]
    valid = list(read_json(rd / "assimilation_decision.json").get("valid_actions") or ASSIM_ACTIONS)
    summary = ExperienceBatchSummary.from_dict(read_json(rd / "experience_summary.json"))
    state_b = ExPolicyState.from_dict(read_json(rd / "state_before.json"))

    if action not in valid:
        payload = {
            "seed": seed,
            "round": t,
            "action": action,
            "original": original,
            "valid": False,
            "U": None,
            "source": "invalid_masked",
        }
        write_json(dest / "result.json", payload)
        (dest / "DONE").write_text("ok\n", encoding="utf-8")
        heartbeat(dest, phase="done_invalid")
        return

    if action == original and not job.get("smoke"):
        reward = read_json(rd / "reward.json")
        payload = {
            "seed": seed,
            "round": t,
            "action": action,
            "original": original,
            "valid": True,
            "source": "reused_original_commit",
            **utility_from_parts(
                shadow_before=reward["shadow_macro_before"],
                shadow_after=reward["shadow_macro_after"],
                retention_before=reward["retention_before"],
                retention_after=reward["retention_after"],
                token_cost=int(reward.get("token_cost") or summary.token_cost),
                tool_cost=int(reward.get("tool_cost") or summary.tool_cost),
                cfg=cfg_raw,
            ),
            "em_by_capability": read_json(rd / "shadow_summary.json").get("em_by_capability"),
            "dispatch_counts": read_json(rd / "shadow_summary.json").get("dispatch_counts"),
        }
        write_json(dest / "result.json", payload)
        (dest / "DONE").write_text("ok\n", encoding="utf-8")
        heartbeat(dest, phase="done_reused_original")
        return

    if action == DISCARD and not job.get("smoke"):
        payload = {
            "seed": seed,
            "round": t,
            "action": action,
            "original": original,
            "valid": True,
            "source": "identity_discard",
            **utility_from_parts(
                shadow_before=state_b.shadow_macro,
                shadow_after=state_b.shadow_macro,
                retention_before=state_b.retention_score,
                retention_after=state_b.retention_score,
                token_cost=summary.token_cost,
                tool_cost=summary.tool_cost,
                cfg=cfg_raw,
            ),
            "note": "DISCARD does not mutate persistent state; J_shadow(A_DISCARD)=J_shadow(S_t) at frozen λ.",
        }
        write_json(dest / "result.json", payload)
        (dest / "DONE").write_text("ok\n", encoding="utf-8")
        heartbeat(dest, phase="done_identity_discard")
        return

    from expolicy.closed_loop.runner.evaluator import evaluate_tasks

    work = dest / "work"
    if work.exists():
        import shutil

        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    state = state_b.copy()
    runtime, model = build_fresh_runtime(cfg, all_tasks, work, state, model, client)
    heartbeat(dest, phase="replay_memory", round=t)
    replay_memory(runtime, seed, t, model)
    runtime.state = state_b.copy()
    exps = load_experiences(rd / "experience_batch.jsonl")
    heartbeat(dest, phase="apply", action=action)
    apply_info = runtime.apply_assimilation(action, summary, exps)
    write_json(dest / "apply.json", apply_info)
    sh_tasks = maybe_trim(shadow_tasks, job.get("smoke"), 2)
    rt_tasks = maybe_trim(retention_tasks, job.get("smoke"), 2)
    heartbeat(dest, phase="shadow", n=len(sh_tasks))
    shadow = evaluate_tasks(
        runtime,
        sh_tasks,
        out=dest,
        phase="shadow",
        heartbeat=lambda **k: heartbeat(dest, **k),
    )
    heartbeat(dest, phase="retention", n=len(rt_tasks))
    retention = evaluate_tasks(
        runtime,
        rt_tasks,
        out=dest,
        phase="retention",
        heartbeat=lambda **k: heartbeat(dest, **k),
    )
    util = utility_from_parts(
        shadow_before=state_b.shadow_macro,
        shadow_after=float(shadow["macro"]),
        retention_before=state_b.retention_score,
        retention_after=float(retention["macro"]),
        token_cost=summary.token_cost,
        tool_cost=summary.tool_cost,
        cfg=cfg_raw,
    )
    payload = {
        "seed": seed,
        "round": t,
        "action": action,
        "original": original,
        "valid": True,
        "source": "replayed_counterfactual",
        "smoke": bool(job.get("smoke")),
        **util,
        "em_by_capability": shadow.get("em_by_capability"),
        "dispatch_counts": shadow.get("dispatch_counts"),
        "apply": apply_info,
        "skills_after": dict(runtime.state.active_skills),
    }
    write_json(dest / "result.json", payload)
    (dest / "DONE").write_text("ok\n", encoding="utf-8")
    heartbeat(dest, phase="done", U=util["U"])


def run_ceiling_job(job: dict[str, Any], *, cfg, all_tasks, future_tasks, model, client) -> None:
    name = job["mask"]
    dest = job_dir("ceiling", job["job_id"])
    dest.mkdir(parents=True, exist_ok=True)
    heartbeat(dest, phase="start", mask=name)
    caps = set(job.get("caps") or [])

    if name == "none" and not job.get("smoke"):
        src = read_json(m0_seed_dir(0) / "eval_budget_0.json")
        write_json(dest / "result.json", {"mask": name, "caps": [], "source": "reused_m0_eval_budget_0", **src})
        (dest / "DONE").write_text("ok\n", encoding="utf-8")
        heartbeat(dest, phase="done_reused")
        return
    if name == "C1+C2+C3" and not job.get("smoke"):
        src = read_json(R7_ROOT / "all_skills" / "seed0" / "future_test_final_summary.json")
        write_json(dest / "result.json", {"mask": name, "caps": list(caps), "source": "reused_all_skills", **src})
        (dest / "DONE").write_text("ok\n", encoding="utf-8")
        heartbeat(dest, phase="done_reused")
        return

    from expolicy.closed_loop.runner.evaluator import evaluate_tasks

    work = dest / "work"
    if work.exists():
        import shutil

        shutil.rmtree(work)
    state = ExPolicyState(budget_total=48)
    for cap in CAPS:
        state.active_skills[cap] = cap in caps
    runtime, model = build_fresh_runtime(cfg, all_tasks, work, state, model, client)
    tasks = maybe_trim(future_tasks, job.get("smoke"), 6)
    heartbeat(dest, phase="future_test", n=len(tasks), mask=name)
    fut = evaluate_tasks(
        runtime,
        tasks,
        out=dest,
        phase="future_test_ceiling",
        heartbeat=lambda **k: heartbeat(dest, **k),
    )
    write_json(dest / "result.json", {"mask": name, "caps": list(caps), "source": "live_mask_eval", "smoke": bool(job.get("smoke")), **fut})
    (dest / "DONE").write_text("ok\n", encoding="utf-8")
    heartbeat(dest, phase="done", macro=fut.get("macro"))


def cf_ready_for_seed(seed: int) -> bool:
    for t in range(6):
        for action in ASSIM_ACTIONS:
            dest = job_dir("cf", f"cf_s{seed}_r{t:02d}_{action}")
            if not (dest / "DONE").exists():
                return False
    return True


def load_cf_u(seed: int, t: int, action: str) -> float | None:
    path = job_dir("cf", f"cf_s{seed}_r{t:02d}_{action}") / "result.json"
    if not path.exists():
        return None
    rec = read_json(path)
    if rec.get("valid") is False:
        return None
    return rec.get("U")


def run_oracle_job(job: dict[str, Any], *, cfg, cfg_raw, all_tasks, shadow_tasks, retention_tasks, future_tasks, model, client) -> None:
    """Hindsight oracle on the *actual* M0 acquired stream (O0-style). Sequential commit."""
    seed = int(job["seed"])
    dest = job_dir("oracle", job["job_id"])
    dest.mkdir(parents=True, exist_ok=True)
    if not cf_ready_for_seed(seed):
        heartbeat(dest, phase="wait_cf", seed=seed)
        return
    heartbeat(dest, phase="start_oracle", seed=seed)

    from expolicy.closed_loop.runner.evaluator import evaluate_tasks

    work = dest / "live"
    if work.exists():
        import shutil

        shutil.rmtree(work)
    state0 = ExPolicyState.from_dict(read_json(m0_round(seed, 0) / "state_before.json"))
    runtime, model = build_fresh_runtime(cfg, all_tasks, work, state0.copy(), model, client)
    runtime.state = state0.copy()
    trajectory: list[dict[str, Any]] = []
    diverged = False

    for t in range(6):
        rd = m0_round(seed, t)
        original = read_json(rd / "assimilation_decision.json")["action"]
        valid = list(read_json(rd / "assimilation_decision.json").get("valid_actions") or ASSIM_ACTIONS)
        orig_state = ExPolicyState.from_dict(read_json(rd / "state_before.json"))
        mask_match = dict(runtime.state.active_skills) == dict(orig_state.active_skills)
        summary = ExperienceBatchSummary.from_dict(read_json(rd / "experience_summary.json"))
        exps = load_experiences(rd / "experience_batch.jsonl")
        us: dict[str, float | None] = {}
        sources: dict[str, str] = {}
        can_reuse_matrix = (not diverged) and mask_match
        for action in ASSIM_ACTIONS:
            if action not in valid:
                us[action] = None
                sources[action] = "invalid"
                continue
            if can_reuse_matrix:
                us[action] = load_cf_u(seed, t, action)
                sources[action] = "independent_matrix"
                continue
            # live counterfactual clone
            import shutil

            clone_root = dest / f"round_{t:02d}" / action
            if clone_root.exists():
                shutil.rmtree(clone_root)
            clone_state = runtime.state.copy()
            clone_rt, model = build_fresh_runtime(cfg, all_tasks, clone_root, clone_state, model, client)
            if (runtime.episodic_dir / "records.jsonl").exists():
                clone_rt.episodic_dir.mkdir(parents=True, exist_ok=True)
                (clone_rt.episodic_dir / "records.jsonl").write_text(
                    (runtime.episodic_dir / "records.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
                )
            if (runtime.semantic_dir / "notes.jsonl").exists():
                clone_rt.semantic_dir.mkdir(parents=True, exist_ok=True)
                (clone_rt.semantic_dir / "notes.jsonl").write_text(
                    (runtime.semantic_dir / "notes.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
                )
            clone_rt.state = runtime.state.copy()
            heartbeat(dest, phase="oracle_cf", seed=seed, round=t, action=action)
            mask_match_now = dict(runtime.state.active_skills) == dict(orig_state.active_skills)
            if action == DISCARD:
                us[action] = utility_from_parts(
                    shadow_before=runtime.state.shadow_macro,
                    shadow_after=runtime.state.shadow_macro,
                    retention_before=runtime.state.retention_score,
                    retention_after=runtime.state.retention_score,
                    token_cost=summary.token_cost,
                    tool_cost=summary.tool_cost,
                    cfg=cfg_raw,
                )["U"]
                sources[action] = "identity_discard"
                continue
            if action == ACTIVATE_EXISTING_SKILL and mask_match_now:
                reused = load_cf_u(seed, t, action)
                if reused is not None:
                    us[action] = reused
                    sources[action] = "matrix_activate_same_mask"
                    continue
            clone_rt.apply_assimilation(action, summary, exps)
            sh = evaluate_tasks(
                clone_rt,
                maybe_trim(shadow_tasks, job.get("smoke"), 2),
                out=clone_root,
                phase="shadow",
                heartbeat=lambda **k: heartbeat(dest, **k),
            )
            rt = evaluate_tasks(
                clone_rt,
                maybe_trim(retention_tasks, job.get("smoke"), 2),
                out=clone_root,
                phase="retention",
                heartbeat=lambda **k: heartbeat(dest, **k),
            )
            us[action] = utility_from_parts(
                shadow_before=runtime.state.shadow_macro,
                shadow_after=float(sh["macro"]),
                retention_before=runtime.state.retention_score,
                retention_after=float(rt["macro"]),
                token_cost=summary.token_cost,
                tool_cost=summary.tool_cost,
                cfg=cfg_raw,
            )["U"]
            sources[action] = "live_clone"
        ranked = [(a, us[a]) for a in ASSIM_ACTIONS if us[a] is not None]
        ranked.sort(key=lambda x: x[1], reverse=True)
        chosen = ranked[0][0] if ranked else DISCARD
        if chosen != original:
            diverged = True
        heartbeat(dest, phase="oracle_commit", round=t, chosen=chosen, original=original)
        apply_info = runtime.apply_assimilation(chosen, summary, exps)
        if chosen == DISCARD:
            commit_src = "identity_discard_commit"
        elif chosen == original and not diverged and not job.get("smoke"):
            orig_reward = read_json(rd / "reward.json")
            orig_shadow = read_json(rd / "shadow_summary.json")
            runtime.state.shadow_macro = float(orig_reward["shadow_macro_after"])
            runtime.state.retention_score = float(orig_reward["retention_after"])
            runtime.state.shadow_em_by_capability = dict(orig_shadow.get("em_by_capability") or {})
            commit_src = "copied_original_round"
        else:
            sh_live = evaluate_tasks(
                runtime,
                maybe_trim(shadow_tasks, job.get("smoke"), 2),
                out=dest / f"round_{t:02d}" / "commit",
                phase="shadow",
                heartbeat=lambda **k: heartbeat(dest, **k),
            )
            rt_live = evaluate_tasks(
                runtime,
                maybe_trim(retention_tasks, job.get("smoke"), 2),
                out=dest / f"round_{t:02d}" / "commit",
                phase="retention",
                heartbeat=lambda **k: heartbeat(dest, **k),
            )
            runtime.state.shadow_macro = float(sh_live["macro"])
            runtime.state.retention_score = float(rt_live["macro"])
            runtime.state.shadow_em_by_capability = dict(sh_live.get("em_by_capability") or {})
            commit_src = "live_eval"
        trajectory.append(
            {
                "round": t,
                "acquired": read_json(rd / "acquisition_decision.json")["capability"],
                "original": original,
                "chosen": chosen,
                "U": us,
                "sources": sources,
                "commit_source": commit_src,
                "apply": apply_info,
                "skills_after": dict(runtime.state.active_skills),
                "shadow_after": runtime.state.shadow_macro,
            }
        )
        write_json(dest / "trajectory.json", trajectory)

    heartbeat(dest, phase="oracle_future")
    fut = evaluate_tasks(
        runtime,
        maybe_trim(future_tasks, job.get("smoke"), 6),
        out=dest,
        phase="future_test_oracle",
        heartbeat=lambda **k: heartbeat(dest, **k),
    )
    write_json(
        dest / "result.json",
        {
            "seed": seed,
            "diverged": diverged,
            "macro": fut["macro"],
            "em_by_capability": fut.get("em_by_capability"),
            "trajectory": trajectory,
            "source": "hindsight_oracle_on_m0_stream",
        },
    )
    (dest / "DONE").write_text("ok\n", encoding="utf-8")
    heartbeat(dest, phase="done", macro=fut["macro"])


def worker_loop(gpu: str, smoke: bool = False) -> None:
    apply_gpu(gpu)
    os.environ["PYTHONUNBUFFERED"] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    logp = OUT / "logs" / f"worker_gpu{gpu}.log"
    logp.parent.mkdir(parents=True, exist_ok=True)

    def wlog(msg: str) -> None:
        line = f"{now()} gpu={gpu} {msg}"
        print(line, flush=True)
        with logp.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    cfg, cfg_raw, all_tasks, shadow_tasks, retention_tasks, future_tasks = load_split_and_cfg()
    from expolicy.environments.public_search.retriever import RetrieverClient

    client = RetrieverClient(os.environ.get("EXPOLICY_RETRIEVER_URL") or cfg.retriever_url)
    model = None
    idle = 0
    while idle < 8:
        job = claim_job(("cf", "ceiling"))
        if job is None:
            job = claim_job(("oracle",))
            if job is not None and job["kind"] == "oracle" and not cf_ready_for_seed(int(job["seed"])):
                job["_lock"].release()
                idle += 1
                wlog(f"oracle seed{job['seed']} waiting for cf matrix")
                time.sleep(30)
                continue
        if job is None:
            idle += 1
            wlog(f"no job, idle {idle}/8")
            time.sleep(20)
            continue
        idle = 0
        job["smoke"] = bool(smoke or job.get("smoke"))
        write_json(OUT / "logs" / f"worker_gpu{gpu}.live.json", {"updated_at": time.time(), "job": job["job_id"], "pid": os.getpid()})
        wlog(f"CLAIM {job['job_id']}")
        try:
            if model is None and job["kind"] in {"cf", "ceiling", "oracle"}:
                need_model = True
                if job["kind"] == "cf":
                    # reuse/identity paths do not need a model
                    action = job["action"]
                    original = job.get("original")
                    valid = job.get("valid", True)
                    if (not valid) or action == DISCARD or (action == original and not job.get("smoke")):
                        need_model = False
                if job["kind"] == "ceiling" and job.get("mask") in {"none", "C1+C2+C3"} and not job.get("smoke"):
                    need_model = False
                if job["kind"] == "oracle":
                    need_model = True
                if need_model:
                    wlog("loading Qwen2.5-3B-Instruct")
                    from expolicy.models.factory import build_backend

                    model = build_backend({"backend": "transformers", "path": cfg.model, "dtype": "bfloat16"})
            if job["kind"] == "cf":
                run_cf_job(
                    job,
                    cfg=cfg,
                    cfg_raw=cfg_raw,
                    all_tasks=all_tasks,
                    shadow_tasks=shadow_tasks,
                    retention_tasks=retention_tasks,
                    model=model,
                    client=client,
                )
            elif job["kind"] == "ceiling":
                run_ceiling_job(job, cfg=cfg, all_tasks=all_tasks, future_tasks=future_tasks, model=model, client=client)
            elif job["kind"] == "oracle":
                run_oracle_job(
                    job,
                    cfg=cfg,
                    cfg_raw=cfg_raw,
                    all_tasks=all_tasks,
                    shadow_tasks=shadow_tasks,
                    retention_tasks=retention_tasks,
                    future_tasks=future_tasks,
                    model=model,
                    client=client,
                )
            dest = job_dir(job["kind"], job["job_id"])
            if (dest / "DONE").exists():
                wlog(f"DONE {job['job_id']}")
            else:
                wlog(f"RELEASE {job['job_id']} (not finished, will retry)")
        except Exception:
            wlog(f"FAIL {job['job_id']}\n{traceback.format_exc()[-2500:]}")
            dest = job_dir(job["kind"], job["job_id"])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "FAIL").write_text(traceback.format_exc(), encoding="utf-8")
            heartbeat(dest, phase="fail", error=traceback.format_exc()[-1000:])
        finally:
            try:
                job["_lock"].release()
            except Exception:
                pass
    wlog("worker exit")


# ---------------------------------------------------------------------------
# Summarize / gate
# ---------------------------------------------------------------------------


def collect_matrix() -> list[dict[str, Any]]:
    rows = []
    for seed in (0, 1, 2):
        for t in range(6):
            rd = m0_round(seed, t)
            acq = read_json(rd / "acquisition_decision.json")
            assim = read_json(rd / "assimilation_decision.json")
            rec: dict[str, Any] = {
                "seed": seed,
                "round": t,
                "capability": acq["capability"],
                "original": assim["action"],
                "valid": list(assim.get("valid_actions") or ASSIM_ACTIONS),
                "features": (assim.get("features") or {}).get(assim["action"]),
                "scores": assim.get("scores"),
                "sampled_values": assim.get("sampled_values"),
            }
            for action in ASSIM_ACTIONS:
                path = job_dir("cf", f"cf_s{seed}_r{t:02d}_{action}") / "result.json"
                if path.exists():
                    blob = read_json(path)
                    rec[f"U_{action}"] = blob.get("U")
                    rec[f"src_{action}"] = blob.get("source")
                    rec[f"valid_{action}"] = blob.get("valid")
                else:
                    rec[f"U_{action}"] = None
                    rec[f"src_{action}"] = "missing"
                    rec[f"valid_{action}"] = None
            usable = {a: rec[f"U_{a}"] for a in ASSIM_ACTIONS if rec[f"U_{a}"] is not None}
            if usable:
                best = max(usable, key=usable.get)
                rec["best"] = best
                rec["best_U"] = usable[best]
                rec["original_U"] = rec.get(f"U_{rec['original']}")
                rec["regret"] = (usable[best] - rec["original_U"]) if rec["original_U"] is not None else None
                unique = True
                for a, u in usable.items():
                    if a != best and u is not None and u >= usable[best] - UNIQUE_EPS:
                        unique = False
                rec["unique_best"] = unique
            else:
                rec["best"] = None
                rec["best_U"] = None
                rec["original_U"] = None
                rec["regret"] = None
                rec["unique_best"] = False
            rows.append(rec)
    return rows


def write_matrix_csv(rows: list[dict[str, Any]]) -> None:
    path = OUT / "R7_COUNTERFACTUAL_ASSIM_MATRIX.csv"
    fields = [
        "seed",
        "round",
        "capability",
        "original",
        "best",
        "unique_best",
        "best_U",
        "original_U",
        "regret",
        *[f"U_{a}" for a in ASSIM_ACTIONS],
        *[f"src_{a}" for a in ASSIM_ACTIONS],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def feature_separability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    labeled = [r for r in rows if r.get("best") and r.get("features")]
    if len(labeled) < 6:
        return {"error": "not enough labeled rows", "n": len(labeled)}
    X = np.array([r["features"] for r in labeled], dtype=float)
    y = np.array([r["best"] for r in labeled])
    majority = Counter(y).most_common(1)[0]
    maj_name, maj_n = majority
    maj_acc = maj_n / len(y)
    seeds = sorted(set(r["seed"] for r in labeled))
    clf_hits = []
    mlp_hits = []
    ridge_sp = []
    mlp_sp = []
    # per-action U regression, leave-one-seed-out
    for hold in seeds:
        tr = [r for r in labeled if r["seed"] != hold]
        te = [r for r in labeled if r["seed"] == hold]
        Xtr = np.array([r["features"] for r in tr], dtype=float)
        ytr = np.array([r["best"] for r in tr])
        Xte = np.array([r["features"] for r in te], dtype=float)
        yte = np.array([r["best"] for r in te])
        clf = LogisticRegression(max_iter=400)
        try:
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            clf_hits.extend((pred == yte).tolist())
        except Exception:
            clf_hits.extend([False] * len(yte))
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0)
        try:
            mlp.fit(Xtr, ytr)
            pred = mlp.predict(Xte)
            mlp_hits.extend((pred == yte).tolist())
        except Exception:
            mlp_hits.extend([False] * len(yte))
        # regression on original U vector
        yUtr = np.array([[r.get(f"U_{a}") or 0.0 for a in ASSIM_ACTIONS] for r in tr], dtype=float)
        yUte = np.array([[r.get(f"U_{a}") or 0.0 for a in ASSIM_ACTIONS] for r in te], dtype=float)
        for i, _a in enumerate(ASSIM_ACTIONS):
            ridge = Ridge(alpha=1.0)
            ridge.fit(Xtr, yUtr[:, i])
            pred_u = ridge.predict(Xte)
            if np.std(yUte[:, i]) > 1e-8:
                ridge_sp.append(float(np.corrcoef(pred_u, yUte[:, i])[0, 1]))
            mlp_r = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0)
            try:
                mlp_r.fit(Xtr, yUtr[:, i])
                pred_m = mlp_r.predict(Xte)
                if np.std(yUte[:, i]) > 1e-8:
                    mlp_sp.append(float(np.corrcoef(pred_m, yUte[:, i])[0, 1]))
            except Exception:
                pass
    return {
        "n": len(labeled),
        "majority_label": maj_name,
        "majority_acc": maj_acc,
        "logreg_loso_acc": float(sum(clf_hits) / max(1, len(clf_hits))),
        "mlp_loso_acc": float(sum(mlp_hits) / max(1, len(mlp_hits))),
        "mlp_minus_majority_pp": 100.0 * (float(sum(mlp_hits) / max(1, len(mlp_hits))) - maj_acc),
        "ridge_mean_pearson": float(sum(ridge_sp) / max(1, len(ridge_sp))) if ridge_sp else None,
        "mlp_reg_mean_pearson": float(sum(mlp_sp) / max(1, len(mlp_sp))) if mlp_sp else None,
        "label_counts": dict(Counter(y)),
    }


def summarize() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect_matrix()
    write_matrix_csv(rows)
    write_json(OUT / "R7_COUNTERFACTUAL_ASSIM_MATRIX.json", rows)
    complete_cf = all(r.get(f"U_{ASSIM_ACTIONS[0]}") is not None or r.get("best") is not None or True for r in rows)
    n_missing = sum(1 for r in rows for a in ASSIM_ACTIONS if r.get(f"src_{a}") == "missing")
    complete_cf = n_missing == 0

    unique_counts = Counter(r["best"] for r in rows if r.get("unique_best") and r.get("best"))
    n_states = len(rows)
    unique_frac = {a: unique_counts[a] / max(1, n_states) for a in ASSIM_ACTIONS}
    n_unique_actions = sum(1 for a, f in unique_frac.items() if f >= 0.20)

    mean_best_u = float(sum(r["best_U"] for r in rows if r.get("best_U") is not None) / max(1, sum(1 for r in rows if r.get("best_U") is not None)))
    mean_orig_u = float(
        sum(r["original_U"] for r in rows if r.get("original_U") is not None)
        / max(1, sum(1 for r in rows if r.get("original_U") is not None))
    )
    # best fixed assimilation on this stream ≈ always the globally most frequent unique-best, else ACTIVATE-if-valid else DISCARD
    # Compare mean U of always-ACTIVATE (when valid) vs always-DISCARD vs original vs oracle-myopic.
    def mean_u(action: str) -> float:
        xs = [r[f"U_{action}"] for r in rows if r.get(f"U_{action}") is not None]
        return float(sum(xs) / len(xs)) if xs else 0.0

    always = {a: mean_u(a) for a in ASSIM_ACTIONS}
    best_fixed_u = max(always.values()) if always else 0.0
    oracle_u = mean_best_u
    util_gap = oracle_u - best_fixed_u

    gate_r7 = read_json(R7_GATE) if R7_GATE.exists() else {}
    strongest_fixed_final = float(gate_r7.get("strongest_non_oracle") or 0.09111111111111113)
    m0_final = float((gate_r7.get("means") or {}).get("expolicy") or 0.0474)
    count_only = float((gate_r7.get("means") or {}).get("passive_count_skill") or 0.0911)
    all_skills = float((gate_r7.get("means") or {}).get("all_skills") or 0.0911)

    ceiling_rows = []
    for name, caps in MASKS:
        path = job_dir("ceiling", f"ceiling_{name}") / "result.json"
        if path.exists():
            blob = read_json(path)
            ceiling_rows.append({"mask": name, "caps": list(caps), "macro": blob.get("macro"), "source": blob.get("source"), "em_by_capability": blob.get("em_by_capability")})
    write_json(OUT / "R7_MASK_CEILING.json", ceiling_rows)
    max_mask = max((c["macro"] or 0.0) for c in ceiling_rows) if ceiling_rows else all_skills
    s2_gap = float(max_mask) - strongest_fixed_final

    oracle_finals = []
    for seed in (0, 1, 2):
        path = job_dir("oracle", f"oracle_s{seed}") / "result.json"
        if path.exists():
            blob = read_json(path)
            oracle_finals.append({"seed": seed, "macro": blob.get("macro"), "diverged": blob.get("diverged"), "trajectory": blob.get("trajectory")})
    write_json(OUT / "R7_ORACLE_TRAJECTORIES.json", oracle_finals)
    oracle_joint = float(sum(o["macro"] for o in oracle_finals) / len(oracle_finals)) if oracle_finals else None

    feat = feature_separability(rows) if complete_cf else {"error": "cf incomplete", "n_missing": n_missing}

    c2_activate_margins = []
    always_activate_or_discard = True
    for r in rows:
        if r["capability"] == C2 and r.get(f"U_{ACTIVATE_EXISTING_SKILL}") is not None:
            others = [r[f"U_{a}"] for a in ASSIM_ACTIONS if a != ACTIVATE_EXISTING_SKILL and r.get(f"U_{a}") is not None]
            if others:
                c2_activate_margins.append(r[f"U_{ACTIVATE_EXISTING_SKILL}"] - max(others))
        if r.get("unique_best") and r.get("best") not in {ACTIVATE_EXISTING_SKILL, DISCARD, None}:
            always_activate_or_discard = False

    s2_fail = s2_gap < 0.02
    s1_ok = (oracle_joint is not None) and (oracle_joint >= strongest_fixed_final + 0.03)
    unique_ok = n_unique_actions >= 2
    util_ok = util_gap >= 0.02
    mlp_acc = feat.get("mlp_loso_acc")
    maj = feat.get("majority_acc")
    mlp_ok = (mlp_acc is not None and maj is not None and (mlp_acc - maj) >= 0.15)

    reusable = bool(s1_ok and unique_ok and util_ok and mlp_ok)
    decision = "R7_ENV_REUSABLE" if reusable else "R7_ENV_STRUCTURALLY_DEGENERATE"

    audit = {
        "decision": decision,
        "reusable": reusable,
        "complete_cf": complete_cf,
        "n_missing_cf": n_missing,
        "n_states": n_states,
        "unique_best_counts": dict(unique_counts),
        "unique_best_fraction": unique_frac,
        "n_actions_unique_best_ge_20pct": n_unique_actions,
        "mean_oracle_step_U": oracle_u,
        "mean_original_step_U": mean_orig_u,
        "always_action_mean_U": always,
        "best_fixed_step_U": best_fixed_u,
        "step_utility_gap_oracle_minus_best_fixed": util_gap,
        "oracle_joint_final": oracle_joint,
        "oracle_finals": oracle_finals,
        "strongest_fixed_final": strongest_fixed_final,
        "m0_final": m0_final,
        "count_only_final": count_only,
        "all_skills_final": all_skills,
        "max_mask_final": max_mask,
        "s2_gap_max_mask_minus_best_fixed": s2_gap,
        "ceiling": ceiling_rows,
        "feature_separability": feat,
        "c2_activate_margins": c2_activate_margins,
        "mean_c2_activate_margin": float(sum(c2_activate_margins) / len(c2_activate_margins)) if c2_activate_margins else None,
        "optimal_almost_always_activate_or_discard": always_activate_or_discard,
        "gates": {
            "S1_oracle_joint_ge_fixed_plus_03": s1_ok,
            "unique_best_ge_2_actions_20pct": unique_ok,
            "step_utility_gap_ge_02": util_ok,
            "mlp_loso_ge_majority_plus_15pp": mlp_ok,
            "S2_max_mask_minus_fixed_ge_02": (not s2_fail),
        },
        "lambda_frozen": {"retention_weight": 0.5, "token_cost_per_million": 0.01, "tool_cost_per_100": 0.005},
    }
    write_json(OUT / "R8_STRUCTURAL_AUDIT.json", audit)

    md = []
    md.append("# R8 Structural Identifiability Audit")
    md.append("")
    md.append(f"- Written: {now()}")
    md.append(f"- Decision: **`{decision}`**")
    md.append("- λ frozen from R7; no new cost tuning.")
    md.append("")
    md.append("## S1 Oracle assimilation gap")
    md.append("")
    md.append("| quantity | value |")
    md.append("|---|---|")
    md.append(f"| oracle_joint final (hindsight assim on M0 stream) | {oracle_joint} |")
    md.append(f"| best_fixed final (CountOnly / all_skills) | {strongest_fixed_final:.4f} |")
    md.append(f"| M0 final | {m0_final:.4f} |")
    md.append(f"| mean per-step oracle U | {oracle_u:.4f} |")
    md.append(f"| mean per-step best-fixed U | {best_fixed_u:.4f} |")
    md.append(f"| per-step utility gap | {util_gap:.4f} |")
    md.append(f"| S1 gate (oracle_joint ≥ fixed + 0.03) | {s1_ok} |")
    md.append("")
    if oracle_finals:
        md.append("| seed | oracle final | diverged from M0 |")
        md.append("|---|---|---|")
        for o in oracle_finals:
            md.append(f"| {o['seed']} | {o['macro']} | {o['diverged']} |")
        md.append("")
    md.append("## S2 Final-state ceiling (frozen C1/C2/C3 masks)")
    md.append("")
    md.append("| mask | macro | source |")
    md.append("|---|---|---|")
    for c in ceiling_rows:
        md.append(f"| {c['mask']} | {c.get('macro')} | {c.get('source')} |")
    md.append("")
    md.append(f"- max_mask − strongest_fixed = **{s2_gap:.4f}** (threshold 0.02).")
    md.append(f"- CountOnly = all_skills = {all_skills:.4f}. Substrate ceiling is the C2 activation mask.")
    md.append("")
    md.append("## 18×4 counterfactual assimilation matrix")
    md.append("")
    md.append(f"- States: {n_states}. Missing cells: {n_missing}.")
    md.append(f"- Unique-best counts: `{dict(unique_counts)}`")
    md.append(f"- Unique-best fractions: `{unique_frac}`")
    md.append(f"- Actions with unique-best ≥20% states: **{n_unique_actions}** (need ≥2).")
    md.append(f"- Mean C2 ACTIVATE margin vs next-best: {audit['mean_c2_activate_margin']}")
    md.append(f"- Optimal action almost always ACTIVATE or DISCARD: {always_activate_or_discard}")
    md.append("")
    md.append("See `R7_COUNTERFACTUAL_ASSIM_MATRIX.csv`.")
    md.append("")
    md.append("## S3 Feature separability (offline, no online policy update)")
    md.append("")
    md.append(f"```json\n{json.dumps(feat, indent=2)}\n```")
    md.append("")
    md.append("## Gate R8-A")
    md.append("")
    for k, v in audit["gates"].items():
        md.append(f"- `{k}`: **{v}**")
    md.append("")
    md.append(f"### `{decision}`")
    md.append("")
    if reusable:
        md.append("R7 SearchEnv may be reused for a new controller, still subject to Gate H/C.")
    else:
        md.append("Do **not** train a new controller on the current frozen-mask SearchEnv.")
        md.append("Next: R8-B heterogeneous assimilator testbed (MemRL + SkillOpt on ALFWorld).")
    md.append("")
    md.append("## Answers to §3.2 questions")
    md.append("")
    md.append(f"1. Is the optimal action almost always ACTIVATE / DISCARD? **{always_activate_or_discard}**")
    md.append(f"2. When C2 is acquired, is ACTIVATE's margin large? mean margin={audit['mean_c2_activate_margin']}")
    md.append("3. Are Linear-TS features enough? See S3; features omit an explicit capability-id and share the same φ across actions.")
    md.append("4. Assimilation regret source: Case 2 = saw C2 and discarded (policy); Case 3 = never acquired C2 (acquisition). Counterfactual matrix separates these.")
    md.append("")
    (OUT / "R8_STRUCTURAL_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"decision={decision} wrote {OUT / 'R8_STRUCTURAL_AUDIT.md'}", flush=True)
    return audit


def progress() -> dict[str, Any]:
    jobs = read_json(OUT / "JOBS.json") if (OUT / "JOBS.json").exists() else []
    done = nfail = 0
    by = defaultdict(lambda: {"n": 0, "done": 0, "fail": 0})
    for job in jobs:
        by[job["kind"]]["n"] += 1
        dest = job_dir(job["kind"], job["job_id"])
        if (dest / "DONE").exists():
            done += 1
            by[job["kind"]]["done"] += 1
        if (dest / "FAIL").exists() and not (dest / "DONE").exists():
            nfail += 1
            by[job["kind"]]["fail"] += 1
    return {"n": len(jobs), "done": done, "fail": nfail, "by": dict(by)}


def main() -> None:
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--r7_dir", default=str(R7_ROOT))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--mode", default="all", choices=["cases", "emit", "worker", "cheap", "summarize", "progress", "all"])
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.mode in {"cases", "all", "emit"}:
        write_case_audit()
    if args.mode in {"emit", "all"}:
        jobs = emit_jobs(smoke=args.smoke)
        print(f"emitted {len(jobs)} jobs", flush=True)
    if args.mode == "cheap":
        if not (OUT / "JOBS.json").exists():
            write_case_audit()
            emit_jobs(smoke=args.smoke)
        from expolicy.closed_loop.runner.config import R7Config
        from expolicy.closed_loop.split_data import load_split, split_tasks

        cfg_raw = read_json(m0_seed_dir(0) / "config.json")
        cfg = R7Config.from_dict(cfg_raw)
        split = load_split()
        dummy_tasks = split_tasks(split, "shadow_reward_probe")[:1]
        n = 0
        for job in read_json(OUT / "JOBS.json"):
            if job.get("kind") != "cf":
                continue
            dest = job_dir("cf", job["job_id"])
            if (dest / "DONE").exists():
                continue
            action = job["action"]
            original = job.get("original")
            valid = job.get("valid", True)
            cheap = (not valid) or action == DISCARD or action == original
            if not cheap:
                continue
            lock = _claimed_lock(job["job_id"])
            try:
                lock.acquire()
            except Timeout:
                continue
            try:
                if (dest / "DONE").exists():
                    continue
                run_cf_job(
                    job,
                    cfg=cfg,
                    cfg_raw=cfg_raw,
                    all_tasks=dummy_tasks,
                    shadow_tasks=dummy_tasks,
                    retention_tasks=dummy_tasks,
                    model=None,
                    client=None,
                )
                n += 1
            finally:
                try:
                    lock.release()
                except Exception:
                    pass
        # reused ceilings
        for job in read_json(OUT / "JOBS.json"):
            if job.get("kind") != "ceiling":
                continue
            if job.get("mask") not in {"none", "C1+C2+C3"}:
                continue
            dest = job_dir("ceiling", job["job_id"])
            if (dest / "DONE").exists():
                continue
            lock = _claimed_lock(job["job_id"])
            try:
                lock.acquire()
            except Timeout:
                continue
            try:
                run_ceiling_job(job, cfg=cfg, all_tasks=dummy_tasks, future_tasks=dummy_tasks, model=None, client=None)
                n += 1
            finally:
                try:
                    lock.release()
                except Exception:
                    pass
        print(f"cheap jobs finished n={n}", flush=True)
    if args.mode == "worker":
        if not (OUT / "JOBS.json").exists():
            write_case_audit()
            emit_jobs(smoke=args.smoke)
        worker_loop(str(args.gpu), smoke=args.smoke)
    if args.mode in {"summarize", "all"}:
        prog = progress()
        print("progress", json.dumps(prog), flush=True)
        if args.mode == "summarize" or (prog["n"] and prog["done"] == prog["n"]):
            summarize()
        elif args.mode == "all":
            print("GPU jobs not complete; skip final gate until worker/summarize.", flush=True)
    if args.mode == "progress":
        print(json.dumps(progress(), indent=2), flush=True)


if __name__ == "__main__":
    main()
