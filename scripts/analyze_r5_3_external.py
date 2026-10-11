#!/usr/bin/env python3
"""Aggregate R5.3 public-search results into Case A/B/C."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from expolicy.environments.public_search.capability_map import PUBLIC_CAPS
from expolicy.experiments.common import ROOT, mean, write_json
from expolicy.experiments.r5_1 import auc_over_budget, bootstrap_ci, load_json, mean_std, write_csv
from expolicy.experiments.r5_3 import DATASETS, PRIMARY_METHODS, R53_ROOT, target_skill_times

PAIRS = (
    ("frontier", "ace_skill_prioritized"),
    ("frontier", "passive"),
    ("ace_skill_prioritized", "passive"),
)


def _run(root: Path, method: str, seed: int) -> Path | None:
    p = root / method / f"seed_{seed}"
    return p if (p / "final_metrics.json").exists() else None


def _checkpoints(run: Path) -> dict[int, dict]:
    path = run / "checkpoint_metrics.jsonl"
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            rows[int(rec["budget"])] = rec
    return rows


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def paired(a: list[float], b: list[float]) -> dict:
    diff = [x - y for x, y in zip(a, b)]
    mu, sd = mean_std(diff)
    lo, hi = bootstrap_ci(diff)
    wins = sum(1 for d in diff if d > 0)
    ties = sum(1 for d in diff if d == 0)
    return {
        "mean": mu,
        "std": sd,
        "ci95": [lo, hi],
        "wins": wins,
        "ties": ties,
        "n": len(diff),
        "ci_lower_gt0": lo > 0,
    }


def analyze(root: Path, stage: str) -> dict:
    seeds = []
    methods_data: dict[str, dict[int, dict]] = {m: {} for m in PRIMARY_METHODS}
    for method in PRIMARY_METHODS:
        for seed in range(10):
            run = _run(root, method, seed)
            if run is None:
                continue
            rec = load_json(run / "final_metrics.json")
            methods_data[method][seed] = rec
            seeds.append(seed)
    seeds = sorted(set(seeds))
    if stage == "pilot":
        seeds = [s for s in seeds if s in {0, 1, 2}]
    per_seed_rows = []
    for seed in seeds:
        row = {"seed": seed}
        for method in PRIMARY_METHODS:
            rec = methods_data[method].get(seed) or {}
            row[f"{method}_base"] = rec.get("base_future_macro")
            row[f"{method}_final"] = rec.get("future_macro")
            row[f"{method}_delta"] = rec.get("delta_future")
            row[f"{method}_aulc"] = None
            run = _run(root, method, seed)
            if run:
                cps = _checkpoints(run)
                pts = [(b, float(cps[b]["macro"])) for b in (0, 16, 32, 48) if b in cps]
                row[f"{method}_aulc"] = auc_over_budget(pts) if len(pts) >= 2 else None
                times = rec.get("first_by_cap") or rec.get("first_skill_activation") or {}
                row[f"{method}_n_skills"] = rec.get("n_activated_target_skills") or rec.get("n_active_skills")
                row[f"{method}_t2"] = rec.get("budget_to_second_skill")
        per_seed_rows.append(row)

    def col(method, key):
        return [methods_data[method][s][key] for s in seeds if s in methods_data[method] and key in methods_data[method][s]]

    deltas = {m: col(m, "delta_future") for m in PRIMARY_METHODS}
    pair_stats = {f"{a}-vs-{b}": paired(deltas[a], deltas[b]) for a, b in PAIRS if deltas[a] and deltas[b] and len(deltas[a]) == len(deltas[b])}

    # dataset-wise mean gain frontier vs others
    ds_gain = {}
    for ds in DATASETS:
        fvals, avals, pvals = [], [], []
        for seed in seeds:
            for method, bucket in (("frontier", fvals), ("ace_skill_prioritized", avals), ("passive", pvals)):
                rec = methods_data[method].get(seed) or {}
                per = rec.get("per_dataset") or {}
                base = rec.get("base_future_macro")
                # per-dataset delta approximated vs that method's own final-base is not per-ds; use final per-ds
                bucket.append(_f((rec.get("per_dataset") or {}).get(ds)))
        ds_gain[ds] = {
            "frontier_mean": mean([x for x in fvals if x is not None]),
            "ace_mean": mean([x for x in avals if x is not None]),
            "passive_mean": mean([x for x in pvals if x is not None]),
        }

    aulc = {}
    t2 = {}
    for method in PRIMARY_METHODS:
        vals = [_f(r.get(f"{method}_aulc")) for r in per_seed_rows]
        aulc[method] = mean([v for v in vals if v is not None])
        raw = []
        for seed in seeds:
            rec = methods_data[method].get(seed) or {}
            v = rec.get("budget_to_second_skill")
            if isinstance(v, (int, float)):
                raw.append(float(v))
            elif isinstance(v, str) and v.startswith(">"):
                raw.append(49.0)
        t2[method] = mean(raw) if raw else None

    n = len(seeds)
    front_d = deltas.get("frontier") or []
    ace_d = deltas.get("ace_skill_prioritized") or []
    pas_d = deltas.get("passive") or []
    stage1 = None
    if stage == "pilot":
        wins_ace = sum(1 for a, b in zip(front_d, ace_d) if a > b) if front_d and ace_d else 0
        more_skill = False
        for seed in seeds:
            nf = int((methods_data["frontier"].get(seed) or {}).get("n_activated_target_skills") or 0)
            na = int((methods_data["ace_skill_prioritized"].get(seed) or {}).get("n_activated_target_skills") or 0)
            if nf > na:
                more_skill = True
        aulc_better = (aulc.get("frontier") or 0) > (aulc.get("ace_skill_prioritized") or 0)
        stage1 = {
            "mean_frontier_gt_ace": mean(front_d) > mean(ace_d) if front_d and ace_d else False,
            "mean_frontier_gt_passive": mean(front_d) > mean(pas_d) if front_d and pas_d else False,
            "frontier_wins_vs_ace": f"{wins_ace}/{len(front_d)}",
            "wins_at_least_2_of_3": wins_ace >= 2,
            "more_skill_or_aulc": more_skill or aulc_better,
            "pass": bool(
                front_d
                and ace_d
                and pas_d
                and mean(front_d) > mean(ace_d)
                and mean(front_d) > mean(pas_d)
                and wins_ace >= 2
                and (more_skill or aulc_better)
            ),
        }

    case = "incomplete"
    if stage == "final" and n >= 10 and front_d and ace_d and pas_d:
        ps = pair_stats["frontier-vs-ace_skill_prioritized"]
        fp = pair_stats["frontier-vs-passive"]
        ds_nonneg = sum(1 for ds in DATASETS if ds_gain[ds]["frontier_mean"] - ds_gain[ds]["ace_mean"] >= -1e-12 or ds_gain[ds]["frontier_mean"] >= ds_gain[ds]["passive_mean"])
        # "至少 2/3 dataset 的 mean gain 不为负" — frontier-ace or frontier-passive gain
        ds_ok = 0
        for ds in DATASETS:
            gain = ds_gain[ds]["frontier_mean"] - ds_gain[ds]["ace_mean"]
            if gain >= 0:
                ds_ok += 1
        eff = (aulc["frontier"] > aulc["ace_skill_prioritized"]) or (
            (t2["frontier"] or 99) < (t2["ace_skill_prioritized"] or 99)
        )
        if (
            ps["mean"] >= 0.03
            and ps["ci_lower_gt0"]
            and ps["wins"] >= 7
            and fp["mean"] > 0
            and ds_ok >= 2
            and eff
        ):
            case = "A"
        elif fp["mean"] > 0 and (not ps["ci_lower_gt0"] or ps["mean"] < 0.03):
            case = "B"
        elif fp["mean"] <= 0:
            case = "C"
        else:
            case = "B"

    final_metrics = []
    for method in PRIMARY_METHODS:
        recs = [methods_data[method][s] for s in seeds if s in methods_data[method]]
        if not recs:
            continue
        final_metrics.append(
            {
                "method": method,
                "n_seeds": len(recs),
                "base_macro": mean(float(r["base_future_macro"]) for r in recs),
                "final_macro": mean(float(r["future_macro"]) for r in recs),
                "delta_future": mean(float(r["delta_future"]) for r in recs),
                "aulc": aulc.get(method),
                "budget_to_second_skill": t2.get(method),
                "unique_tasks": mean(float(r.get("unique_tasks") or 0) for r in recs),
                "duplicates": mean(float(r.get("duplicate_rollouts") or 0) for r in recs),
                "tokens": mean(float(r.get("tokens") or 0) for r in recs),
                "tool_calls": mean(float(r.get("tool_calls") or 0) for r in recs),
            }
        )

    curves = []
    dist_rows = []
    act_rows = []
    eff_rows = []
    per_ds_rows = []
    per_cap_rows = []
    diag_rows = []
    for method in PRIMARY_METHODS:
        for seed in seeds:
            run = _run(root, method, seed)
            if run is None:
                continue
            rec = methods_data[method][seed]
            for b, cp in _checkpoints(run).items():
                curves.append({"method": method, "seed": seed, "budget": b, "macro": cp.get("macro"), **(cp.get("per_dataset") or {})})
            dist_rows.append({"method": method, "seed": seed, **{k: rec.get(k) for k in ("unique_tasks", "duplicate_rollouts", "rollouts_per_capability", "rollouts_per_dataset", "capability_entropy", "n_active_skills")}})
            act_rows.append({"method": method, "seed": seed, **{k: rec.get(k) for k in ("budget_to_first_skill", "budget_to_second_skill", "budget_to_third_skill", "first_by_cap", "n_activated_target_skills")}})
            delta = float(rec.get("delta_future") or 0)
            tokens = float(rec.get("tokens") or 0)
            tools = float(rec.get("tool_calls") or 0)
            uniq = float(rec.get("unique_tasks") or 0)
            eff_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "delta_per_rollout": delta / max(1.0, float(rec.get("n_acquired") or 48)),
                    "delta_per_unique": delta / max(1.0, uniq),
                    "delta_per_100k_tokens": delta / max(1.0, tokens / 1e5),
                    "delta_per_100_tools": delta / max(1.0, tools / 100.0),
                    "tokens": tokens,
                    "tool_calls": tools,
                }
            )
            for ds, val in (rec.get("per_dataset") or {}).items():
                per_ds_rows.append({"method": method, "seed": seed, "dataset": ds, "accuracy": val})
            for cap, val in (rec.get("per_capability") or {}).items():
                per_cap_rows.append({"method": method, "seed": seed, "capability": cap, "accuracy": val})

    decision = {
        "stage": stage,
        "n_seeds": n,
        "seeds": seeds,
        "case": case,
        "stage1": stage1,
        "pairs": pair_stats,
        "aulc": aulc,
        "time_to_second_skill": t2,
        "dataset_means": ds_gain,
    }
    write_json(root / "R5_3_DECISION.json", decision)
    write_csv(root / "R5_3_FINAL_METRICS.csv", final_metrics)
    write_csv(root / "R5_3_PER_SEED.csv", per_seed_rows)
    write_csv(root / "R5_3_LEARNING_CURVES.csv", curves)
    write_csv(root / "R5_3_ACQUISITION_DISTRIBUTION.csv", dist_rows)
    write_csv(root / "R5_3_ACTIVATION_TIMES.csv", act_rows)
    write_csv(root / "R5_3_EFFICIENCY.csv", eff_rows)
    write_csv(root / "R5_3_PER_DATASET.csv", per_ds_rows)
    write_csv(root / "R5_3_PER_CAPABILITY.csv", per_cap_rows)
    write_json(root / "R5_3_PAIRED_BOOTSTRAP.json", pair_stats)

    # master summary questions
    fa = pair_stats.get("frontier-vs-ace_skill_prioritized") or {}
    fp = pair_stats.get("frontier-vs-passive") or {}
    ap = pair_stats.get("ace_skill_prioritized-vs-passive") or {}
    lines = [
        "# R5_3 MASTER SUMMARY",
        "",
        f"Stage = {stage}. Seeds = {seeds}. Decision case = **{case}**.",
        "",
        "## 1. Helion `frontier > Ace-Skill` on public Search-Agent benchmarks?",
        f"- mean(frontier − Ace) ΔFuture = {fa.get('mean')}",
        f"- paired bootstrap 95% CI = {fa.get('ci95')}",
        f"- wins = {fa.get('wins')}/{fa.get('n')}",
        "",
        "## 2. Seed stability?",
        json.dumps(per_seed_rows, indent=2, default=str),
        "",
        "## 3. Which datasets reproduce?",
        json.dumps(ds_gain, indent=2),
        "",
        "## 4. Which capabilities did frontier cover first?",
        json.dumps(act_rows, indent=2, default=str)[:4000],
        "",
        "## 5. Capability allocation vs dataset bias?",
        json.dumps(dist_rows, indent=2, default=str)[:4000],
        "",
        "## 6. Does Ace-Skill still collapse to passive?",
        f"- Ace − passive mean Δ = {ap.get('mean')} CI={ap.get('ci95')} wins={ap.get('wins')}/{ap.get('n')}",
        "",
        "## 7. Second skill earlier for frontier?",
        f"- mean time-to-second-skill frontier={t2.get('frontier')} Ace={t2.get('ace_skill_prioritized')} passive={t2.get('passive')}",
        "",
        "## 8. External executable skill positive transfer?",
        "See SKILL_VALIDATION.md (pre-acquisition). Downstream ΔFuture is the acquisition-gated transfer.",
        "",
        "## 9. Oracle vs text router gap?",
        "See ROUTER_AUDIT.md.",
        "",
        "## 10. Bamboogle zero-shot?",
        "See R5_3_BAMBOOGLE_TRANSFER.csv (filled after Step 8).",
        "",
        "## 11. Supports capability-frontier acquisition or Helion-specific heuristic?",
        f"Case {case}: A = external generalization of the mechanism; B = beats passive but not Ace; C = Helion-specific.",
        "",
        "## 12. Next step Case A / B / C?",
        f"**Case {case}**",
        "",
        "## Stage-1 gate" if stage1 else "",
        json.dumps(stage1, indent=2) if stage1 else "",
        "",
        f"AULC frontier={aulc.get('frontier')} Ace={aulc.get('ace_skill_prioritized')} passive={aulc.get('passive')}",
    ]
    (root / "R5_3_MASTER_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "case": case, "n": n, "stage1": stage1}, indent=2, default=str), flush=True)
    if stage == "pilot" and stage1 and not stage1["pass"]:
        raise SystemExit("Stage-1 CONTINUE conditions not met; stopping before 10-seed expansion.")
    return decision


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=str(R53_ROOT))
    p.add_argument("--stage", choices=["pilot", "final"], default="pilot")
    args = p.parse_args()
    analyze(Path(args.input_dir), args.stage)


if __name__ == "__main__":
    main()
