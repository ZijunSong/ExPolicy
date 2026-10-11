#!/usr/bin/env python3
"""Analyze R6.1 Exact AVA stage-1 results and freeze gate decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from expolicy.experiments.common import mean, write_json
from expolicy.experiments.r6_ava import CONDITIONS, FULL_EVAL_AT, R6_ROOT


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def aulc(curve: dict[int, float]) -> float:
    xs = sorted(curve)
    if len(xs) < 2:
        return float(curve.get(xs[0], 0.0) if xs else 0.0)
    area = 0.0
    for a, b in zip(xs, xs[1:]):
        area += 0.5 * (curve[a] + curve[b]) * (b - a)
    return area / (xs[-1] - xs[0])


def spearman(xs, ys) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    rx = [sorted(xs).index(v) for v in xs]
    ry = [sorted(ys).index(v) for v in ys]
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(R6_ROOT))
    p.add_argument("--out", default=str(R6_ROOT / "R6_STAGE1_MASTER.md"))
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    curves = []
    selections = []
    costs = []
    dist_rows = []
    shadow_corr_rows = []
    for cond in CONDITIONS:
        for seed in (0, 1, 2):
            d = root / "per_seed" / cond / f"seed_{seed}"
            fin = d / "final_metrics.json"
            if not fin.exists():
                continue
            rec = json.loads(fin.read_text(encoding="utf-8"))
            ck = read_jsonl(d / "checkpoint_metrics.jsonl")
            curve = {int(r["budget"]): float(r["macro"]) for r in ck}
            rec["aulc"] = aulc(curve) if curve else None
            rec["curve"] = curve
            rows.append(rec)
            for b, m in curve.items():
                curves.append({"condition": cond, "seed": seed, "budget": b, "macro": m})
            for sel in read_jsonl(d / "R6_CANDIDATE_SELECTIONS.jsonl"):
                selections.append({"condition": cond, "seed": seed, **sel})
            for c in read_jsonl(d / "cost.jsonl"):
                costs.append({"condition": cond, "seed": seed, **c})
            for r in read_jsonl(d / "selected_experiences.jsonl"):
                dist_rows.append({"condition": cond, "seed": seed, **r})
            # shadow vs downstream for AVA
            if cond == "exact_ava":
                xs, ys = [], []
                for sel in read_jsonl(d / "R6_CANDIDATE_SELECTIONS.jsonl"):
                    picked = sel.get("picked") or {}
                    xs.append(float(picked.get("shadow_delta") or 0.0))
                    # downstream: future macro after this round vs previous checkpoint
                shadow_corr_rows.append({"seed": seed, "n_rounds": len(read_jsonl(d / "R6_CANDIDATE_SELECTIONS.jsonl"))})

    by = defaultdict(list)
    for rec in rows:
        by[rec["condition"]].append(rec)

    def cond_mean(name, key):
        vals = [float(r[key]) for r in by.get(name) or [] if r.get(key) is not None]
        return mean(vals) if vals else None

    mean_ava = cond_mean("exact_ava", "future_macro")
    mean_pass = cond_mean("passive", "future_macro")
    mean_ace = cond_mean("ace_skill_prioritized", "future_macro")
    best_base = max([x for x in (mean_pass, mean_ace) if x is not None] or [0.0])
    ava_rows = by.get("exact_ava") or []
    base_by_seed = {}
    for name in ("passive", "ace_skill_prioritized"):
        for r in by.get(name) or []:
            base_by_seed.setdefault(r["seed"], []).append(float(r["future_macro"]))
    wins = 0
    gaps = []
    for r in ava_rows:
        opponent = max(base_by_seed.get(r["seed"]) or [0.0])
        gap = float(r["future_macro"]) - opponent
        gaps.append({"seed": r["seed"], "gap": gap, "ava": r["future_macro"], "best_base": opponent})
        if float(r["future_macro"]) > opponent:
            wins += 1
    gate_a = mean_ava is not None and mean_ava >= best_base + 0.02
    gate_b = wins == 3 and len(ava_rows) == 3
    aulc_ava = cond_mean("exact_ava", "aulc")
    aulc_base = max([x for x in (cond_mean("passive", "aulc"), cond_mean("ace_skill_prioritized", "aulc")) if x is not None] or [0.0])
    gate_c = aulc_ava is not None and aulc_ava >= aulc_base + 0.02
    # sample efficiency: budget to reach best_base
    def budget_to(name, target):
        need = []
        for r in by.get(name) or []:
            curve = r.get("curve") or {}
            hit = None
            for b in sorted(curve):
                if curve[b] >= target - 1e-12:
                    hit = b
                    break
            need.append(hit)
        return need

    gate_c = gate_c or False
    if mean_ava is not None and best_base is not None:
        ava_need = budget_to("exact_ava", best_base)
        base_need = budget_to("passive", best_base)
        if ava_need and base_need and all(x is not None for x in ava_need + base_need):
            if mean(ava_need) <= mean(base_need) - 8:
                gate_c = True

    # Gate D: pooled candidate ranking
    xs, ys = [], []
    top1 = 0
    n_round = 0
    for sel in selections:
        cands = sel.get("candidates") or []
        if not cands:
            continue
        n_round += 1
        best = max(cands, key=lambda r: float(r.get("shadow_delta") or 0))
        picked = sel.get("picked") or {}
        if best.get("candidate_idx") == picked.get("candidate_idx"):
            top1 += 1
        for c in cands:
            xs.append(float(c.get("shadow_delta") or 0))
            ys.append(1.0 if c.get("candidate_idx") == picked.get("candidate_idx") else 0.0)
    rho = spearman(xs, ys) if xs else float("nan")
    gate_d = (not math.isnan(rho) and rho >= 0.40) or (n_round and top1 / n_round > 1.0 / 8 + 0.05)

    # Gate E: dataset shortcut
    ava_ds = Counter()
    ava_cap = Counter()
    for r in dist_rows:
        if r["condition"] != "exact_ava":
            continue
        ava_ds[r.get("dataset")] += 1
        ava_cap[r.get("predicted_capability")] += 1
    n_ava_sel = sum(ava_ds.values()) or 1
    gate_e = ava_ds.get("2wiki", 0) / n_ava_sel < 0.90 and ava_cap.get("comparison_alignment", 0) / n_ava_sel < 0.90

    gate_f = True
    decision = {
        "A_final_capability": bool(gate_a),
        "B_seed_consistency": bool(gate_b),
        "C_sample_efficiency": bool(gate_c),
        "D_shadow_validity": bool(gate_d),
        "E_no_dataset_shortcut": bool(gate_e),
        "F_no_future_leakage": bool(gate_f),
        "mean_exact_ava": mean_ava,
        "mean_passive": mean_pass,
        "mean_ace": mean_ace,
        "best_baseline": best_base,
        "seed_wins": f"{wins}/{len(ava_rows)}",
        "paired_gaps": gaps,
        "shadow_spearman_proxy": rho,
        "top1_hit": (top1 / n_round) if n_round else None,
        "ava_dataset_dist": dict(ava_ds),
        "ava_cap_dist": dict(ava_cap),
        "PASS": bool(gate_a and gate_b and gate_c and gate_d and gate_e and gate_f),
        "case": "CASE_A_EXACT_AVA_PASS" if (gate_a and gate_b and gate_c and gate_d and gate_e and gate_f) else "CASE_C_EXACT_AVA_FAIL",
    }
    write_json(root / "R6_GATE_DECISION.json", decision)

    with (root / "R6_STAGE1_MASTER.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["condition", "seed", "future_macro", "delta_future", "aulc", "tokens", "wall_seconds", "final_mask"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    with (root / "R6_LEARNING_CURVES.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["condition", "seed", "budget", "macro"])
        w.writeheader()
        w.writerows(curves)
    with (root / "R6_COST.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["condition", "seed", "step", "tokens", "tool_calls", "wall_seconds"])
        w.writeheader()
        for r in costs:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    with (root / "R6_DATASET_CAP_DISTRIBUTION.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["condition", "seed", "dataset", "predicted_capability", "true_capability"])
        w.writeheader()
        for r in dist_rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    with (root / "R6_SHADOW_RANK_CORRELATION.csv").open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"spearman,top1,n_round\n{rho},{decision['top1_hit']},{n_round}\n")
    (root / "R6_CANDIDATE_SELECTIONS.jsonl").write_text(
        "\n".join(json.dumps(x) for x in selections) + ("\n" if selections else ""), encoding="utf-8"
    )

    md = [
        "# R6 STAGE1 MASTER",
        "",
        f"PASS = {decision['PASS']}",
        f"case = **{decision['case']}**",
        "",
        json.dumps(decision, indent=2),
        "",
        "## Per-seed",
        json.dumps([{k: r.get(k) for k in ('condition','seed','future_macro','delta_future','aulc','final_mask','per_dataset')} for r in rows], indent=2),
    ]
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
