#!/usr/bin/env python3
"""Aggregate R5.1 tracks into master tables and a decision JSON. No training."""

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
    bootstrap_ci,
    load_json,
    mean_std,
    write_csv,
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


def _f(row: dict, key: str) -> float | None:
    val = row.get(key)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def analyze_seed_extension(root: Path, r5: Path) -> dict:
    rows = []
    for seed in range(10):
        src = r5 if seed <= 2 else root / "seed_extension"
        base_path = src / "base" / f"seed_{seed}" / "final_metrics.json"
        if seed <= 2:
            base_path = r5 / "base" / f"seed_{seed}" / "final_metrics.json"
        if not base_path.exists():
            continue
        base = load_json(base_path)
        rec = {
            "seed": seed,
            "base_future_em": base["future"]["em"],
            "base_retention_em": base["retention"]["em"],
        }
        complete = True
        for method in ("frontier", "passive"):
            mpath = (r5 if seed <= 2 else root / "seed_extension") / method / f"seed_{seed}" / "final_metrics.json"
            if not mpath.exists():
                complete = False
                continue
            m = load_json(mpath)
            rec[f"{method}_future_em"] = m.get("future_em")
            rec[f"{method}_delta_future"] = m.get("delta_future")
            rec[f"{method}_retention_em"] = m.get("retention_em")
            rec[f"{method}_tokens"] = m.get("tokens")
            rec[f"{method}_tool_calls"] = m.get("tool_calls")
            rec[f"{method}_first_skill"] = json.dumps(m.get("first_skill_activation") or {})
            rec[f"{method}_caps"] = json.dumps(m.get("caps_chosen") or {})
            rec[f"{method}_n_active"] = m.get("n_active_skills")
            for cap in TARGET_CAP_LIST:
                em = float(((m.get("future_per_cap") or {}).get(cap) or {}).get("em") or 0.0)
                bem = float(((base.get("future_per_cap") or {}).get(cap) or {}).get("em") or 0.0)
                rec[f"{method}_{cap}_gain"] = em - bem
        if rec.get("frontier_delta_future") is not None and rec.get("passive_delta_future") is not None:
            rec["frontier_minus_passive"] = float(rec["frontier_delta_future"]) - float(rec["passive_delta_future"])
        rec["complete"] = complete and "frontier_minus_passive" in rec
        rows.append(rec)
    write_csv(root / "seed_extension" / "SEED_EXTENSION_PER_SEED.csv", rows)
    diffs = [float(r["frontier_minus_passive"]) for r in rows if r.get("complete")]
    mu, sd = mean_std(diffs)
    lo, hi = bootstrap_ci(diffs, seed=51)
    wins = sum(1 for d in diffs if d > 0)
    payload = {
        "n_seeds": len(diffs),
        "mean_frontier_minus_passive": mu,
        "std": sd,
        "paired_bootstrap_95ci": [lo, hi],
        "win_rate": (wins / len(diffs)) if diffs else None,
        "wins": wins,
        "criterion": {
            "mean_gap_gt_0": bool(diffs) and mu > 0,
            "ci_excludes_0": bool(diffs) and lo > 0,
            "at_least_7_of_10": wins >= 7 and len(diffs) >= 10,
        },
        "mean_frontier_delta": mean([float(r["frontier_delta_future"]) for r in rows if r.get("frontier_delta_future") is not None]),
        "mean_passive_delta": mean([float(r["passive_delta_future"]) for r in rows if r.get("passive_delta_future") is not None]),
        "per_seed_gap": diffs,
    }
    write_json(root / "seed_extension" / "SEED_EXTENSION_SUMMARY.json", payload)
    write_json(root / "seed_extension" / "SEED_EXTENSION_BOOTSTRAP.json", {"diffs": diffs, "ci95": [lo, hi], "n": 2000})
    md = [
        "# SEED_EXTENSION",
        "",
        f"n={payload['n_seeds']}  mean(frontier-passive)={mu:.4f}  std={sd:.4f}  paired 95% CI=[{lo:.4f}, {hi:.4f}]",
        f"win-rate={wins}/{len(diffs) or 0}",
        "",
        "Robust-support bar: mean gap>0, CI excludes 0, ≥7/10 wins. Recorded regardless of pass/fail; no seed dropped.",
        "",
        f"- mean_gap_gt_0: {payload['criterion']['mean_gap_gt_0']}",
        f"- ci_excludes_0: {payload['criterion']['ci_excludes_0']}",
        f"- at_least_7_of_10: {payload['criterion']['at_least_7_of_10']}",
        "",
    ]
    for r in rows:
        md.append(
            f"- seed {r['seed']}: base={r.get('base_future_em')} frontierΔ={r.get('frontier_delta_future')} "
            f"passiveΔ={r.get('passive_delta_future')} gap={r.get('frontier_minus_passive')} complete={r.get('complete')}"
        )
    (root / "seed_extension" / "SEED_EXTENSION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def analyze_all(root: Path, r5: Path) -> dict:
    seed_ext = analyze_seed_extension(root, r5)
    thresh = _read_csv(root / "activation_threshold" / "THRESHOLD_PER_SEED.csv")
    content = _read_csv(root / "content_causality" / "CONTENT_CAUSALITY_PER_SEED.csv")
    dispatch = _read_csv(root / "deployable_dispatch" / "DISPATCH_DOWNSTREAM.csv")
    temporal = _read_csv(root / "temporal" / "TEMPORAL_ACQUISITION_COUNTS.csv")

    # Table 2: threshold
    thresh_summary = []
    by_mk = defaultdict(list)
    for row in thresh:
        by_mk[(row.get("method"), row.get("k"))].append(_f(row, "delta_future"))
    for (method, k), xs in sorted(by_mk.items(), key=lambda kv: (str(kv[0][0]), float(kv[0][1] or 0))):
        xs = [x for x in xs if x is not None]
        thresh_summary.append({"method": method, "k": k, "mean_delta_future": mean(xs), "n": len(xs)})

    # Table 3: content
    content_summary = []
    by_mm = defaultdict(list)
    for row in content:
        by_mm[(row.get("method"), row.get("fit_mode"))].append(_f(row, "delta_future"))
    for (method, mode), xs in sorted(by_mm.items()):
        xs = [x for x in xs if x is not None]
        content_summary.append({"method": method, "fit_mode": mode, "mean_delta_future": mean(xs), "n": len(xs)})

    def _mean_mode(method: str, mode: str) -> float:
        xs = [x for x in by_mm.get((method, mode), []) if x is not None]
        return mean(xs) if xs else float("nan")

    real_f = _mean_mode("frontier", "real")
    count_f = _mean_mode("frontier", "count_only")
    shuf_f = _mean_mode("frontier", "within_cap_shuffle")
    cross_f = _mean_mode("frontier", "cross_cap_content")
    content_is_count_gated = (not math.isnan(real_f)) and abs(real_f - count_f) < 0.05
    content_is_material = (
        (not math.isnan(real_f))
        and real_f > count_f + 0.03
        and real_f > shuf_f + 0.03
        and real_f > cross_f + 0.08
    )

    # Table 4: dispatch
    disp_summary = []
    by_md = defaultdict(list)
    for row in dispatch:
        by_md[(row.get("method"), row.get("dispatch"))].append(_f(row, "future_em"))
    for (method, mode), xs in sorted(by_md.items()):
        xs = [x for x in xs if x is not None]
        disp_summary.append({"method": method, "dispatch": mode, "mean_future_em": mean(xs), "n": len(xs)})

    def _em(method, mode):
        xs = [x for x in by_md.get((method, mode), []) if x is not None]
        return mean(xs) if xs else float("nan")

    oracle_f = _em("frontier", "oracle")
    router_f = _em("frontier", "text_router")
    router_p = _em("passive", "text_router")
    deployable_keeps_gap = (not math.isnan(router_f)) and (not math.isnan(router_p)) and router_f > router_p
    router_drop = (oracle_f - router_f) if (not math.isnan(oracle_f) and not math.isnan(router_f)) else float("nan")

    # Table 5 temporal
    temporal_summary = []
    for row in temporal:
        temporal_summary.append(
            {
                "method": row.get("method"),
                "seed": row.get("seed"),
                "temporal_acquired": row.get("temporal_acquired"),
                "s2_activated": row.get("s2_activated"),
                "delta_future": row.get("delta_future"),
            }
        )

    seed_ok = bool(seed_ext.get("criterion", {}).get("mean_gap_gt_0")) and bool(
        seed_ext.get("criterion", {}).get("ci_excludes_0")
    )
    seed_wins_ok = bool(seed_ext.get("criterion", {}).get("at_least_7_of_10"))
    thresh_dir_ok = False
    if thresh_summary:
        front = [r for r in thresh_summary if r["method"] == "frontier"]
        pas = {str(r["k"]): r["mean_delta_future"] for r in thresh_summary if r["method"] == "passive"}
        thresh_dir_ok = all(
            (r["mean_delta_future"] or 0) >= (pas.get(str(r["k"])) or 0) for r in front if r["k"] is not None
        ) and len(front) >= 2

    if seed_ok and seed_wins_ok and thresh_dir_ok and content_is_material and deployable_keeps_gap:
        case = "A"
        story = "active acquisition → experience-content-dependent executable assimilation → deployable skill dispatch → robust future capability gain"
    elif seed_ok and content_is_count_gated:
        case = "B"
        story = (
            "Experience-Gated Skill Activation: active acquisition determines which "
            "capability-specific executable procedures become eligible / activated. "
            "Do not claim that experience content is deeply absorbed by the assimilator "
            f"(frontier RealFit Δ={real_f:.3f} vs CountOnly Δ={count_f:.3f}). "
            "WithinCapShuffle can still move per-cap EM, so report gating as the main mechanism "
            "and keep the shuffle result as a secondary caveat."
        )
    elif seed_ok and (math.isnan(router_drop) or router_drop > 0.08) and not deployable_keeps_gap:
        case = "C"
        story = "acquisition matters under controlled capability dispatch; deployable skill dispatch remains unsolved."
    elif seed_ext.get("n_seeds", 0) >= 7 and not seed_ok:
        case = "D"
        story = "R5 is pilot / mechanism evidence; do not expand into full ExPolicy."
    else:
        case = "INCOMPLETE_OR_MIXED"
        story = "Wait for remaining tracks or mixed evidence; do not start R6 learned carrier router."

    decision = {
        "case": case,
        "story": story,
        "seed_extension": seed_ext,
        "content_is_count_gated": content_is_count_gated,
        "content_is_material": content_is_material,
        "realfit_minus_count_only": (real_f - count_f) if not (math.isnan(real_f) or math.isnan(count_f)) else None,
        "realfit_minus_shuffle": (real_f - shuf_f) if not (math.isnan(real_f) or math.isnan(shuf_f)) else None,
        "deployable_keeps_gap": deployable_keeps_gap,
        "text_router_drop_from_oracle": router_drop,
        "threshold_direction_ok": thresh_dir_ok,
        "do_not_start_r6_carrier_router": True,
        "consider_learned_acquisition_only_if_case_A": case == "A",
    }
    write_json(root / "R5_1_DECISION.json", decision)

    master_rows = []
    for rec in _read_csv(root / "seed_extension" / "SEED_EXTENSION_PER_SEED.csv"):
        master_rows.append({"table": "seed_extension", **rec})
    for rec in thresh_summary:
        master_rows.append({"table": "threshold", **rec})
    for rec in content_summary:
        master_rows.append({"table": "content", **rec})
    for rec in disp_summary:
        master_rows.append({"table": "dispatch", **rec})
    write_csv(root / "R5_1_MASTER_TABLE.csv", master_rows)

    md = [
        "# R5.1 MASTER SUMMARY",
        "",
        f"**Decision case: {case}**",
        "",
        story,
        "",
        "Do not start R6 learned carrier router. E5–E8 remain off.",
        "",
        "## 1. 10-seed frontier vs passive",
        "",
        f"- n={seed_ext.get('n_seeds')} mean gap={seed_ext.get('mean_frontier_minus_passive')} "
        f"CI={seed_ext.get('paired_bootstrap_95ci')} wins={seed_ext.get('wins')}/{seed_ext.get('n_seeds')}",
        "",
        "## 2. Threshold sensitivity",
        "",
    ]
    for row in thresh_summary:
        md.append(f"- {row}")
    md += ["", "## 3. RealFit / CountOnly / WithinCapShuffle / CrossCapContent", ""]
    for row in content_summary:
        md.append(f"- {row}")
    md += [
        "",
        f"- RealFit vs CountOnly (frontier mean Δ): {real_f} vs {count_f}",
        f"- content_is_material={content_is_material} content_is_count_gated={content_is_count_gated}",
        "",
        "## 4. Oracle vs deployable text-router dispatch",
        "",
    ]
    for row in disp_summary:
        md.append(f"- {row}")
    md += [
        "",
        f"- frontier oracle EM={oracle_f} text-router EM={router_f} drop={router_drop}",
        f"- frontier>passive under text-router: {deployable_keeps_gap}",
        "",
        "## 5. Temporal coverage diagnostic",
        "",
    ]
    for row in temporal_summary:
        md.append(f"- {row}")
    md += ["", "## Decision tree", "", f"- Case {case}: {story}", ""]
    (root / "R5_1_MASTER_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"case": case, "n_seed_ext": seed_ext.get("n_seeds")}, indent=2))
    return decision


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze R5.1 robustness outputs")
    p.add_argument("--root", default=str(R51_ROOT))
    p.add_argument("--r5-root", default=str(R5_ROOT))
    p.add_argument("--only", default=None, help="Optional subset: seed_extension|temporal|all")
    args = p.parse_args()
    root = Path(args.root)
    r5 = Path(args.r5_root)
    root.mkdir(parents=True, exist_ok=True)
    if args.only == "seed_extension":
        analyze_seed_extension(root, r5)
        return
    if args.only == "temporal":
        print("temporal tables are written by the driver")
        return
    analyze_all(root, r5)


if __name__ == "__main__":
    main()
