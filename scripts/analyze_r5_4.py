#!/usr/bin/env python3
"""Aggregate R5.4 results and emit a frozen Case A–E decision."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from expolicy.environments.public_search.capability_map import PUBLIC_CAPS
from expolicy.experiments.common import ROOT, mean, write_json
from expolicy.experiments.r5_1 import auc_over_budget, bootstrap_ci, mean_std, write_csv
from expolicy.experiments.r5_4 import CONDITIONS, R54_ROOT

STAGE1_CONDS = list(CONDITIONS)
STAGE2_CONDS = ["rasa_full", "ace_k8", "passive_k8"]


def _run(root: Path, condition: str, seed: int) -> Path | None:
    p = root / "per_seed" / condition / f"seed_{seed}"
    return p if (p / "final_metrics.json").exists() else None


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def paired(a: list[float], b: list[float], n_boot: int) -> dict:
    diff = [x - y for x, y in zip(a, b)]
    mu, sd = mean_std(diff)
    lo, hi = bootstrap_ci(diff, n=n_boot)
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


def collect(root: Path, conditions: list[str], seeds: list[int]) -> dict[str, dict[int, dict]]:
    data: dict[str, dict[int, dict]] = {c: {} for c in conditions}
    for cond in conditions:
        for seed in seeds:
            run = _run(root, cond, seed)
            if run is None:
                continue
            rec = _load(run / "final_metrics.json")
            rec["_run"] = str(run)
            rec["_checkpoints"] = _checkpoints(run)
            data[cond][seed] = rec
    return data


def gate_decision(data: dict[str, dict[int, dict]], seeds: list[int]) -> dict:
    def col(cond, key):
        return [_f(data[cond][s].get(key)) for s in seeds if s in data.get(cond, {}) and _f(data[cond][s].get(key)) is not None]

    def mean_or_nan(cond, key):
        xs = col(cond, key)
        return mean(xs) if xs else float("nan")

    rasa = "rasa_full"
    rasa_final = col(rasa, "future_macro")
    rasa_delta = col(rasa, "delta_future")
    passive = col("passive_k8", "future_macro")
    ace = col("ace_k8", "future_macro")
    passive_srt = col("passive_srt", "future_macro")
    rasa_k8 = col("rasa_k8", "future_macro")
    rasa_qonly = col("rasa_qonly", "future_macro")

    mean_rasa = mean(rasa_final) if rasa_final else float("nan")
    mean_passive = mean(passive) if passive else float("nan")
    mean_ace = mean(ace) if ace else float("nan")
    strongest = max(mean_passive, mean_ace) if not math.isnan(mean_passive) and not math.isnan(mean_ace) else float("nan")
    strongest_name = "ace_k8" if mean_ace >= mean_passive else "passive_k8"

    common = [s for s in seeds if s in data.get(rasa, {}) and s in data.get(strongest_name, {})]
    rasa_vs_strong = [
        float(data[rasa][s]["future_macro"]) > float(data[strongest_name][s]["future_macro"]) for s in common
    ]

    def aulc(cond, seed):
        cps = data[cond][seed].get("_checkpoints") or {}
        pts = [(b, float(cps[b]["macro"])) for b in (0, 16, 32, 48) if b in cps]
        return auc_over_budget(pts) if len(pts) >= 2 else None

    rasa_aulc = [aulc(rasa, s) for s in seeds if s in data.get(rasa, {})]
    qonly_aulc = [aulc("rasa_qonly", s) for s in seeds if s in data.get("rasa_qonly", {})]
    rasa_aulc = [x for x in rasa_aulc if x is not None]
    qonly_aulc = [x for x in qonly_aulc if x is not None]

    def term_budget(cond, seed):
        rec = data[cond][seed]
        states = rec.get("rasa_states") or {}
        posts = rec.get("posteriors") or {}
        vals = []
        for cap in PUBLIC_CAPS:
            p = posts.get(cap) or {}
            b = p.get("budget_to_terminal")
            if b is not None:
                vals.append(int(b))
        return min(vals) if vals else rec.get("n_acquired_unique")

    rasa_term = [term_budget(rasa, s) for s in seeds if s in data.get(rasa, {})]
    q_term = [term_budget("rasa_qonly", s) for s in seeds if s in data.get("rasa_qonly", {})]

    gate_a_mean = (not math.isnan(mean_rasa) and not math.isnan(strongest) and mean_rasa >= strongest + 0.02)
    gate_a_seeds = bool(common) and all(rasa_vs_strong) and len(common) >= 3
    gate_a = gate_a_mean and gate_a_seeds
    gate_b = bool(rasa_delta) and mean(rasa_delta) > 0
    mean_srt = mean(passive_srt) if passive_srt else float("nan")
    mean_k8 = mean(rasa_k8) if rasa_k8 else float("nan")
    mean_q = mean(rasa_qonly) if rasa_qonly else float("nan")
    gate_c = (not math.isnan(mean_rasa) and not math.isnan(mean_srt) and mean_rasa - mean_srt >= 0.01)
    gate_d = (not math.isnan(mean_rasa) and not math.isnan(mean_k8) and mean_rasa - mean_k8 >= 0.01)
    gate_e_mean = (not math.isnan(mean_rasa) and not math.isnan(mean_q) and mean_rasa - mean_q >= 0.01)
    aulc_edge = bool(rasa_aulc and qonly_aulc) and (mean(rasa_aulc) - mean(qonly_aulc) > 1.0 / 450.0)
    term_edge = bool(rasa_term and q_term) and (mean(q_term) - mean(rasa_term) >= 16)
    close_final = (not math.isnan(mean_rasa) and not math.isnan(mean_q) and abs(mean_rasa - mean_q) < 0.01)
    gate_e = gate_e_mean or (close_final and (aulc_edge or term_edge))

    collapse_flags = []
    terminal_ok = []
    for s in seeds:
        rec = data.get(rasa, {}).get(s) or {}
        n_term = int(rec.get("n_terminal_caps") or 0)
        terminal_ok.append(n_term >= 2)
        collapse_flags.append(bool(rec.get("collapse_unresolved_single_cap")))
    gate_f = bool(terminal_ok) and all(terminal_ok) and not any(collapse_flags)

    gates = {
        "A_final_capability": bool(gate_a),
        "A_mean_margin": float(mean_rasa - strongest) if not math.isnan(mean_rasa) and not math.isnan(strongest) else None,
        "A_seed_wins": f"{sum(rasa_vs_strong)}/{len(common)}" if common else "0/0",
        "B_positive_delta": bool(gate_b),
        "B_mean_delta": mean(rasa_delta) if rasa_delta else None,
        "C_not_gate_only": bool(gate_c),
        "C_margin": float(mean_rasa - mean_srt) if not math.isnan(mean_rasa) and not math.isnan(mean_srt) else None,
        "D_srt_contributes": bool(gate_d),
        "D_margin": float(mean_rasa - mean_k8) if not math.isnan(mean_rasa) and not math.isnan(mean_k8) else None,
        "E_remediability_causal": bool(gate_e),
        "E_mean_margin": float(mean_rasa - mean_q) if not math.isnan(mean_rasa) and not math.isnan(mean_q) else None,
        "E_aulc_or_term_edge": bool(aulc_edge or term_edge),
        "F_no_collapse": bool(gate_f),
        "strongest_baseline": strongest_name,
        "mean_rasa_full": mean_rasa,
        "mean_passive_k8": mean_passive,
        "mean_ace_k8": mean_ace,
        "mean_passive_srt": mean_srt,
        "mean_rasa_k8": mean_k8,
        "mean_rasa_qonly": mean_q,
    }
    all_pass = gate_a and gate_b and gate_c and gate_d and gate_e and gate_f
    approx = 0.01
    if all_pass:
        case = "CASE_A_PASS"
    elif math.isnan(mean_rasa) or math.isnan(strongest) or mean_rasa < strongest + 0.02 or not gate_a_seeds:
        case = "CASE_E_METHOD_FAIL"
    elif (not math.isnan(mean_srt)) and abs(mean_rasa - mean_srt) < approx:
        case = "CASE_C_GATE_ONLY"
    elif (not math.isnan(mean_q)) and abs(mean_rasa - mean_q) < approx and not gate_e:
        case = "CASE_D_REMEDIABILITY_NOT_CAUSAL"
    elif (not math.isnan(mean_k8)) and abs(mean_rasa - mean_k8) < approx:
        case = "CASE_B_ACQ_ONLY"
    else:
        case = "CASE_E_METHOD_FAIL"
    gates["PASS"] = all_pass
    gates["case"] = case
    return gates


def analyze(root: Path, stage: str, n_boot: int) -> dict:
    seeds = list(range(3)) if stage == "1" else list(range(10))
    conds = STAGE1_CONDS if stage == "1" else STAGE2_CONDS
    data = collect(root, conds, seeds)
    present_seeds = sorted({s for cond in conds for s in data[cond]})
    master_rows = []
    curve_rows = []
    posterior_rows = []
    decision_rows = []
    cost_rows = []
    for cond in conds:
        for seed, rec in sorted(data[cond].items()):
            master_rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "base_macro": rec.get("base_future_macro"),
                    "final_macro": rec.get("future_macro"),
                    "delta_future": rec.get("delta_future"),
                    "n_acquired_unique": rec.get("n_acquired_unique"),
                    "early_stop": rec.get("early_stop"),
                    "n_active_skills": rec.get("n_active_skills"),
                    "n_terminal_caps": rec.get("n_terminal_caps"),
                    "tokens": rec.get("tokens"),
                    "tool_calls": rec.get("tool_calls"),
                    "diagnostic_replay_count": rec.get("diagnostic_replay_count"),
                    "wall_seconds": rec.get("wall_seconds"),
                    "hotpotqa": (rec.get("per_dataset") or {}).get("hotpotqa"),
                    "2wiki": (rec.get("per_dataset") or {}).get("2wiki"),
                    "musique": (rec.get("per_dataset") or {}).get("musique"),
                }
            )
            cps = rec.get("_checkpoints") or {}
            for b, row in sorted(cps.items()):
                curve_rows.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "budget": b,
                        "macro": row.get("macro"),
                        "n_active_skills": row.get("n_active_skills"),
                    }
                )
                for cap, post in (row.get("posteriors") or {}).items():
                    posterior_rows.append({"condition": cond, "seed": seed, "budget": b, "capability": cap, **post})
            decision_rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "states": json.dumps(rec.get("rasa_states") or {}),
                    "budget_to_first_ACTIVE": rec.get("budget_to_first_ACTIVE"),
                    "budget_to_C2_ACTIVE": rec.get("budget_to_C2_ACTIVE"),
                    "collapse": rec.get("collapse_unresolved_single_cap"),
                }
            )
            cost_rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "acquired_unique": rec.get("n_acquired_unique"),
                    "diagnostic_replays": rec.get("diagnostic_replay_count"),
                    "total_executor_episodes": rec.get("total_executor_episodes"),
                    "tokens": rec.get("tokens"),
                    "tool_calls": rec.get("tool_calls"),
                    "wall_seconds": rec.get("wall_seconds"),
                }
            )

    write_csv(root / "R5_4_STAGE1_MASTER.csv" if stage == "1" else root / "R5_4_STAGE2_MASTER.csv", master_rows)
    write_csv(root / "R5_4_LEARNING_CURVES.csv", curve_rows)
    write_csv(root / "R5_4_POSTERIORS.csv", posterior_rows)
    write_csv(root / "R5_4_DECISIONS.csv", decision_rows)
    write_csv(root / "R5_4_COST.csv", cost_rows)

    gates = gate_decision(data, [s for s in seeds if s in present_seeds or True])
    # restrict seeds to those that actually finished rasa_full if present
    finished = [s for s in seeds if s in data.get("rasa_full", {})]
    if finished:
        gates = gate_decision(data, finished)

    pairs = {}
    if stage == "2":
        for a, b in (("rasa_full", "ace_k8"), ("rasa_full", "passive_k8")):
            sa = [s for s in range(10) if s in data.get(a, {}) and s in data.get(b, {})]
            if sa:
                pairs[f"{a}-vs-{b}"] = paired(
                    [float(data[a][s]["future_macro"]) for s in sa],
                    [float(data[b][s]["future_macro"]) for s in sa],
                    n_boot,
                )
        if pairs:
            strongest = "ace_k8" if mean([float(data["ace_k8"][s]["future_macro"]) for s in range(10) if s in data.get("ace_k8", {})] or [0]) >= mean(
                [float(data["passive_k8"][s]["future_macro"]) for s in range(10) if s in data.get("passive_k8", {})] or [0]
            ) else "passive_k8"
            key = f"rasa_full-vs-{strongest}"
            rec = pairs.get(key) or {}
            rasa_vals = [float(data["rasa_full"][s]["future_macro"]) for s in range(10) if s in data.get("rasa_full", {})]
            base_vals = [float(data[strongest][s]["future_macro"]) for s in range(10) if s in data.get(strongest, {})]
            common = [s for s in range(10) if s in data.get("rasa_full", {}) and s in data.get(strongest, {})]
            wins = sum(1 for s in common if float(data["rasa_full"][s]["future_macro"]) > float(data[strongest][s]["future_macro"]))
            stage2_pass = (
                bool(rasa_vals)
                and bool(base_vals)
                and (mean(rasa_vals) - mean(base_vals) >= 0.02)
                and bool(rec.get("ci_lower_gt0"))
                and wins >= 8
            )
            gates["stage2"] = {"pass": stage2_pass, "wins": f"{wins}/{len(common)}", "pairs": pairs, "strongest": strongest}

    write_json(root / "R5_4_GATE_DECISION.json", gates)
    lines = [
        "# R5.4 MASTER SUMMARY" if stage == "1" else "# R5.4 FINAL SUMMARY",
        "",
        f"stage = {stage}",
        f"case = **{gates.get('case')}**",
        f"PASS = {gates.get('PASS')}",
        "",
        "## Means",
        json.dumps({k: gates[k] for k in gates if k.startswith("mean_") or k in {"strongest_baseline", "A_mean_margin", "B_mean_delta"}}, indent=2),
        "",
        "## Gates",
        json.dumps({k: v for k, v in gates.items() if k[0] in "ABCDEF" or k in {"PASS", "case"}}, indent=2),
        "",
        "## Per-seed master",
        json.dumps(master_rows, indent=2)[:8000],
        "",
        "Do not retune thresholds. Do not drop seeds. Do not rewrite a fail into a weak observation paper.",
    ]
    return {"gates": gates, "n_master": len(master_rows), "md": "\n".join(lines) + "\n"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(R54_ROOT))
    p.add_argument("--stage", default="1")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    root = Path(args.root)
    result = analyze(root, str(args.stage), int(args.bootstrap))
    out = Path(args.output) if args.output else root / ("R5_4_MASTER_SUMMARY.md" if str(args.stage) == "1" else "R5_4_FINAL_SUMMARY.md")
    out.write_text(result["md"], encoding="utf-8")
    print(result["md"], flush=True)
    print(json.dumps(result["gates"], indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
