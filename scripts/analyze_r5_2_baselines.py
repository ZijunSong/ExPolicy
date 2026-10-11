#!/usr/bin/env python3
"""Aggregate R5.2 baseline collision into master tables and a Case A/B/C decision."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from expolicy.experiments.common import ROOT, mean, write_json
from expolicy.experiments.r5_1 import (
    R51_ROOT,
    R5_ROOT,
    TARGET_CAP_LIST,
    auc_over_budget,
    bootstrap_ci,
    load_json,
    mean_std,
    write_csv,
)
from expolicy.experiments.r5_2 import (
    R52_ROOT,
    acquisition_distribution,
    load_acquired,
    maybe_text_router_eval,
    resolve_reused_run,
    target_skill_times,
    unique_counts,
    write_audit,
)

METHODS = ("frontier", "ace_skill_prioritized", "passive")
PAIRS = (
    ("frontier", "ace_skill_prioritized"),
    ("frontier", "passive"),
    ("ace_skill_prioritized", "passive"),
)


def _try_json(path: Path):
    if path.exists():
        return load_json(path)
    return None


def _read_csv(path: Path) -> list[dict]:
    import csv

    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _run_dir(root: Path, method: str, seed: int, r5: Path, r51: Path) -> Path | None:
    local = root / method / f"seed_{seed}"
    if (local / "final_metrics.json").exists():
        return local
    reused = resolve_reused_run(method, seed, r5=r5, r51=r51)
    return reused


def _checkpoints(run: Path) -> dict[int, dict]:
    path = run / "checkpoint_metrics.jsonl"
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows[int(rec["budget"])] = rec
    return rows


def _fmt_time(val) -> str:
    if val is None:
        return ">48"
    if isinstance(val, str):
        return val
    return str(int(val))


def collect_seed(root: Path, method: str, seed: int, r5: Path, r51: Path) -> dict | None:
    run = _run_dir(root, method, seed, r5, r51)
    if run is None:
        return None
    metrics = _try_json(run / "final_metrics.json")
    if metrics is None:
        return None
    acquired = []
    try:
        acquired = load_acquired(run)
    except Exception:
        acquired = []
    dist = _try_json(run / "acquisition_distribution.json") or (
        acquisition_distribution(acquired) if acquired else {}
    )
    first = metrics.get("first_skill_activation") or {}
    times = target_skill_times({k: int(v) for k, v in first.items()}, budget=int(metrics.get("n_acquired") or 48))
    ckpt = _checkpoints(run)
    curve = []
    for b in (0, 8, 16, 24, 32, 40, 48):
        row = ckpt.get(b)
        if row is None:
            continue
        em = row.get("future", {}).get("em") if isinstance(row.get("future"), dict) else row.get("future_em")
        if em is None and b == 0:
            em = metrics.get("base_future_em")
        if em is None:
            continue
        curve.append((b, float(em)))
    if not curve and metrics.get("base_future_em") is not None and metrics.get("future_em") is not None:
        curve = [(0, float(metrics["base_future_em"])), (48, float(metrics["future_em"]))]
    aulc = auc_over_budget(curve) if len(curve) >= 2 else None
    unique_caps = dist.get("unique_tasks_per_capability") or unique_counts(acquired)
    tokens = float(metrics.get("tokens") or 0.0)
    tools = float(metrics.get("tool_calls") or 0.0)
    d_f = float(metrics.get("delta_future") or 0.0)
    n_unique = float(dist.get("unique_tasks") or metrics.get("unique_tasks") or 0.0)
    n_roll = float(dist.get("total_rollouts") or metrics.get("n_acquired") or 0.0)
    rec = {
        "method": method,
        "seed": seed,
        "base_future_em": metrics.get("base_future_em"),
        "future_em": metrics.get("future_em"),
        "delta_future": metrics.get("delta_future"),
        "retention_em": metrics.get("retention_em"),
        "delta_retention": metrics.get("delta_retention"),
        "n_acquired": metrics.get("n_acquired"),
        "n_active_skills": metrics.get("n_active_skills"),
        "tokens": metrics.get("tokens"),
        "tool_calls": metrics.get("tool_calls"),
        "total_rollouts": dist.get("total_rollouts", metrics.get("n_acquired")),
        "unique_tasks": dist.get("unique_tasks", metrics.get("unique_tasks")),
        "duplicate_rollouts": dist.get("duplicate_rollouts", metrics.get("duplicate_rollouts")),
        "unique_mh": unique_caps.get("multi_hop"),
        "unique_temporal": unique_caps.get("temporal_reasoning"),
        "unique_s4": unique_caps.get("cross_document_synthesis"),
        "budget_to_first_target_skill": times["budget_to_first_target_skill"],
        "budget_to_second_target_skill": times["budget_to_second_target_skill"],
        "n_activated_target_skills": times["n_activated_target_skills"],
        "first_skill_activation": json.dumps(first),
        "aulc": aulc,
        "aulc_points": json.dumps(curve),
        "delta_future_per_100k_tokens": (d_f / tokens * 100_000) if tokens else None,
        "delta_future_per_100_tools": (d_f / tools * 100.0) if tools else None,
        "delta_future_per_rollout": (d_f / n_roll) if n_roll else None,
        "delta_future_per_unique_task": (d_f / n_unique) if n_unique else None,
        "run_dir": str(run),
    }
    per_cap = metrics.get("future_per_cap") or {}
    for cap in TARGET_CAP_LIST:
        rec[f"future_em_{cap}"] = ((per_cap.get(cap) or {}).get("em") if isinstance(per_cap.get(cap), dict) else None)
    return rec


def paired_report(rows: list[dict], a: str, b: str) -> dict:
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[int(r["seed"])][r["method"]] = r
    diffs = []
    per_seed = []
    for seed in sorted(by_seed):
        ra, rb = by_seed[seed].get(a), by_seed[seed].get(b)
        if ra is None or rb is None:
            continue
        da = _f(ra.get("delta_future"))
        db = _f(rb.get("delta_future"))
        if da is None or db is None:
            continue
        gap = da - db
        diffs.append(gap)
        per_seed.append({"seed": seed, "gap": gap, f"{a}_delta": da, f"{b}_delta": db})
    mu, sd = mean_std(diffs)
    lo, hi = bootstrap_ci(diffs, seed=52)
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    return {
        "a": a,
        "b": b,
        "n": len(diffs),
        "mean_gap": mu,
        "std": sd,
        "paired_bootstrap_95ci": [lo, hi],
        "win_rate": (wins / len(diffs)) if diffs else None,
        "wins": wins,
        "ties": ties,
        "losses": len(diffs) - wins - ties,
        "per_seed": per_seed,
        "diffs": diffs,
    }


def decide(reports: dict[tuple[str, str], dict], aulc: dict[str, float], t2: dict[str, list]) -> dict:
    fa = reports[("frontier", "ace_skill_prioritized")]
    mu = float(fa.get("mean_gap") or 0.0)
    lo, hi = fa.get("paired_bootstrap_95ci") or [float("nan"), float("nan")]
    wins = int(fa.get("wins") or 0)
    n = int(fa.get("n") or 0)
    aulc_f = aulc.get("frontier")
    aulc_a = aulc.get("ace_skill_prioritized")
    def _mean_time(vals: list) -> float:
        nums = []
        for v in vals:
            if isinstance(v, str) and v.startswith(">"):
                nums.append(49.0)
            elif v is not None:
                nums.append(float(v))
        return mean(nums) if nums else float("nan")

    t2_f = _mean_time(t2.get("frontier") or [])
    t2_a = _mean_time(t2.get("ace_skill_prioritized") or [])
    efficiency_ok = False
    if aulc_f is not None and aulc_a is not None and aulc_f > aulc_a:
        efficiency_ok = True
    if t2_f < t2_a:
        efficiency_ok = True
    case_a = mu >= 0.05 and lo > 0 and wins >= 7 and n >= 10
    ace_beats = (-mu) >= 0.05 and hi < 0 and (n - wins - int(fa.get("ties") or 0)) >= 7
    if case_a and efficiency_ok:
        case = "A"
        story = (
            "Frontier is empirically stronger than Ace-Skill Prioritized sampling "
            "under identical CountOnly unique-task skill activation."
        )
    elif ace_beats:
        case = "C"
        story = (
            "Ace-Skill Prioritized sampling beats frontier. Frontier sampling "
            "cannot be claimed as the paper's core acquisition contribution."
        )
    else:
        case = "B"
        story = (
            "Frontier and Ace-Skill are statistically tied at the 10-seed grain. "
            "The current frontier rule is likely a special case of generic prioritized curriculum."
        )
    grain = 1.0 / 36.0
    return {
        "case": case,
        "story": story,
        "mean_frontier_minus_ace": mu,
        "ci95": [lo, hi],
        "wins": wins,
        "n": n,
        "case_a_mean_ge_0.05": mu >= 0.05,
        "case_a_ci_gt_0": bool(lo > 0),
        "case_a_wins_ge_7_of_10": wins >= 7 and n >= 10,
        "sample_efficiency_advantage": efficiency_ok,
        "aulc_frontier": aulc_f,
        "aulc_ace": aulc_a,
        "mean_time_to_second_skill_frontier": t2_f,
        "mean_time_to_second_skill_ace": t2_a,
        "one_item_grain": grain,
        "gap_smaller_than_one_item": abs(mu) < grain,
        "do_not_claim_material_if_gap_lt_one_item": abs(mu) < grain,
    }


def analyze(root: Path, r5: Path, r51: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    write_audit(root)
    per_seed_dir = root / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        for seed in range(10):
            rec = collect_seed(root, method, seed, r5, r51)
            if rec is None:
                continue
            rows.append(rec)
            write_json(per_seed_dir / f"{method}_seed_{seed}.json", rec)
    write_csv(root / "R5_2_FINAL_METRICS.csv", rows)

    curve_rows = []
    for rec in rows:
        pts = json.loads(rec.get("aulc_points") or "[]")
        for b, em in pts:
            curve_rows.append({"method": rec["method"], "seed": rec["seed"], "budget": b, "future_em": em})
    write_csv(root / "R5_2_LEARNING_CURVES.csv", curve_rows)

    act_rows = [
        {
            "method": r["method"],
            "seed": r["seed"],
            "budget_to_first_target_skill": r["budget_to_first_target_skill"],
            "budget_to_second_target_skill": r["budget_to_second_target_skill"],
            "n_activated_target_skills": r["n_activated_target_skills"],
            "unique_mh": r["unique_mh"],
            "unique_temporal": r["unique_temporal"],
            "unique_s4": r["unique_s4"],
            "first_skill_activation": r["first_skill_activation"],
        }
        for r in rows
    ]
    write_csv(root / "R5_2_ACTIVATION_TIMES.csv", act_rows)
    write_csv(
        root / "R5_2_ACQUISITION_DISTRIBUTION.csv",
        [
            {
                "method": r["method"],
                "seed": r["seed"],
                "total_rollouts": r["total_rollouts"],
                "unique_tasks": r["unique_tasks"],
                "duplicate_rollouts": r["duplicate_rollouts"],
                "unique_mh": r["unique_mh"],
                "unique_temporal": r["unique_temporal"],
                "unique_s4": r["unique_s4"],
            }
            for r in rows
        ],
    )
    write_csv(
        root / "R5_2_EFFICIENCY.csv",
        [
            {
                "method": r["method"],
                "seed": r["seed"],
                "delta_future": r["delta_future"],
                "delta_future_per_100k_tokens": r["delta_future_per_100k_tokens"],
                "delta_future_per_100_tools": r["delta_future_per_100_tools"],
                "delta_future_per_rollout": r["delta_future_per_rollout"],
                "delta_future_per_unique_task": r["delta_future_per_unique_task"],
                "aulc": r["aulc"],
            }
            for r in rows
        ],
    )

    reports = {pair: paired_report(rows, pair[0], pair[1]) for pair in PAIRS}
    write_json(root / "R5_2_PAIRED_BOOTSTRAP.json", {f"{a}-minus-{b}": v for (a, b), v in reports.items()})

    aulc_mean = {}
    t2 = defaultdict(list)
    for method in METHODS:
        xs = [_f(r["aulc"]) for r in rows if r["method"] == method]
        xs = [x for x in xs if x is not None]
        aulc_mean[method] = mean(xs) if xs else None
        t2[method] = [r["budget_to_second_target_skill"] for r in rows if r["method"] == method]

    decision = decide(reports, aulc_mean, t2)

    # text-router sanity if present
    router_rows = []
    for method in METHODS:
        for seed in range(10):
            path = root / method / f"seed_{seed}" / "text_router" / "metrics.json"
            rec = _try_json(path)
            if rec:
                router_rows.append(rec)
    router_order_ok = None
    if router_rows:
        by_m = defaultdict(list)
        for r in router_rows:
            by_m[r["method"]].append(float(r["delta_future"]))
        means = {m: mean(xs) for m, xs in by_m.items() if xs}
        if "frontier" in means and "ace_skill_prioritized" in means:
            router_order_ok = means["frontier"] > means["ace_skill_prioritized"]
        elif "frontier" in means and "passive" in means:
            router_order_ok = means["frontier"] > means["passive"]
    decision["text_router_frontier_still_ahead"] = router_order_ok

    write_json(root / "R5_2_DECISION.json", decision)

    def _mean_method(method: str, key: str):
        xs = [_f(r.get(key)) for r in rows if r["method"] == method]
        xs = [x for x in xs if x is not None]
        return mean(xs) if xs else None

    dup_ace = _mean_method("ace_skill_prioritized", "duplicate_rollouts")
    uniq_t_f = _mean_method("frontier", "unique_temporal")
    uniq_t_a = _mean_method("ace_skill_prioritized", "unique_temporal")
    uniq_t_p = _mean_method("passive", "unique_temporal")

    fa = reports[("frontier", "ace_skill_prioritized")]
    fp = reports[("frontier", "passive")]
    ap = reports[("ace_skill_prioritized", "passive")]

    md = [
        "# R5.2 MASTER SUMMARY",
        "",
        "Strong-baseline collision test: frontier vs Ace-Skill Prioritized Sampler vs passive.",
        "Backend is frozen CountOnly unique-task gating (K=8 distinct task IDs). No RealFit, no R6 router.",
        "",
        f"**Decision: Case {decision['case']}**",
        "",
        decision["story"],
        "",
        "## 1. Frontier vs Ace-Skill?",
        "",
        f"- n={fa['n']} mean(frontier − Ace-Skill) ΔFuture = {fa['mean_gap']}",
        f"- std = {fa['std']}",
        f"- paired bootstrap 95% CI = {fa['paired_bootstrap_95ci']}",
        f"- win rate = {fa['wins']}/{fa['n']} (ties={fa['ties']})",
        f"- one-item grain 1/36 ≈ {decision['one_item_grain']:.4f}; |gap| < one item: {decision['gap_smaller_than_one_item']}",
        "",
        "## 2. Does the gap hold across 10 seeds?",
        "",
        f"- Case A win bar (≥7/10): {decision['case_a_wins_ge_7_of_10']}",
        f"- CI excludes 0 on the positive side: {decision['case_a_ci_gt_0']}",
        "",
    ]
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[int(r["seed"])][r["method"]] = r
    for seed in range(10):
        recs = by_seed.get(seed, {})
        bits = [f"seed {seed}"]
        for m in METHODS:
            if m in recs:
                bits.append(f"{m} Δ={recs[m].get('delta_future')}")
        md.append("- " + " | ".join(bits))
    md += [
        "",
        "## 3. Final performance vs faster skill activation?",
        "",
        f"- mean AULC frontier={aulc_mean.get('frontier')} Ace-Skill={aulc_mean.get('ace_skill_prioritized')} passive={aulc_mean.get('passive')}",
        f"- mean time-to-second-target-skill frontier={decision['mean_time_to_second_skill_frontier']} Ace-Skill={decision['mean_time_to_second_skill_ace']}",
        f"- sample-efficiency advantage (AULC or time-to-2nd): {decision['sample_efficiency_advantage']}",
        "",
        "## 4. Does Ace-Skill lose unique evidence by revisiting?",
        "",
        f"- mean Ace-Skill duplicate_rollouts = {dup_ace}",
        f"- mean unique tasks: frontier={_mean_method('frontier','unique_tasks')} Ace-Skill={_mean_method('ace_skill_prioritized','unique_tasks')} passive={_mean_method('passive','unique_tasks')}",
        "",
        "## 5. Is temporal still a blind spot for every policy?",
        "",
        f"- mean unique temporal evidence: frontier={uniq_t_f} Ace-Skill={uniq_t_a} passive={uniq_t_p}",
        f"- mean future EM temporal: frontier={_mean_method('frontier','future_em_temporal_reasoning')} Ace-Skill={_mean_method('ace_skill_prioritized','future_em_temporal_reasoning')} passive={_mean_method('passive','future_em_temporal_reasoning')}",
        "",
        "## 6. Can the current contribution still be written as an independent acquisition method?",
        "",
    ]
    if decision["case"] == "A":
        md.append("Yes, with the collision test passed: frontier is not just Ace-Skill-style generic prioritized sampling.")
    elif decision["case"] == "B":
        md.append("No. Do not sell frontier sampling as novelty. Upgrade the contribution to Evidence-Gated Skill Activation and read SLIM before designing the new gate.")
    else:
        md.append("No. Keep R5/R5.1 as a mechanism finding. Design a new evidence/utility-aware acquisition that must beat Ace-Skill before any external benchmark.")
    md += [
        "",
        "## 7. Next step: Case A / B / C",
        "",
        f"**Case {decision['case']}**",
        "",
        "## Other paired gaps",
        "",
        f"- frontier − passive: mean={fp['mean_gap']} CI={fp['paired_bootstrap_95ci']} wins={fp['wins']}/{fp['n']}",
        f"- Ace-Skill − passive: mean={ap['mean_gap']} CI={ap['paired_bootstrap_95ci']} wins={ap['wins']}/{ap['n']}",
        "",
        f"- text-router order preserved (frontier ahead): {decision.get('text_router_frontier_still_ahead')}",
        "",
        "AULC uses full-eval checkpoints 0/16/32/48 (intermediate 8/24/40 record activation only).",
        "",
    ]
    (root / "R5_2_MASTER_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"case": decision["case"], "n_rows": len(rows), "out": str(root)}, indent=2))
    return decision


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", "--input_dir", dest="input_dir", default=str(R52_ROOT))
    p.add_argument("--reuse-r5", dest="reuse_r5", default=str(R5_ROOT))
    p.add_argument("--reuse-r5-1", dest="reuse_r51", default=str(R51_ROOT))
    p.add_argument("--text-router", dest="text_router", action="store_true")
    args = p.parse_args()
    root = Path(args.input_dir)
    r5 = Path(args.reuse_r5)
    r51 = Path(args.reuse_r51)
    if args.text_router:
        from expolicy.experiments.common import load_world

        world = load_world()
        for method in METHODS:
            for seed in range(10):
                maybe_text_router_eval(
                    method=method,
                    seed=seed,
                    output_dir=root,
                    r5_root=r5,
                    r51_root=r51,
                    world=world,
                )
    analyze(root, r5, r51)


if __name__ == "__main__":
    main()
