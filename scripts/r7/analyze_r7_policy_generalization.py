#!/usr/bin/env python3
"""Analyze R7 artifacts, write matrices, Case A/B/C/D decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from expolicy.experiments.common import write_json
from expolicy.r7.constants import (
    CONTENT_RANGE_GATE,
    CONTENT_STD_GATE,
    FRONTIER_MACRO_GATE,
    H_CARRIER_GATE,
    H_POLICY_GATE,
    H_SKILL_GATE,
    ORACLE_HEADROOM_GATE,
    PUBLIC_CAPS,
    R7_DATASETS,
    R7_ROOT,
)
from expolicy.r7.skillopt import write_skillopt_port_audit


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def loadj(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def analyze_content(out: Path) -> dict:
    root = out / "assimilation" / "content_batches"
    rows = []
    per_cap_gains: dict[str, list[float]] = defaultdict(list)
    hashes = []
    for summary in root.rglob("summary.json"):
        rec = loadj(summary)
        if not rec or "future_gain" not in rec:
            continue
        rec["path"] = str(summary.parent)
        rows.append(rec)
        hashes.append(json.dumps(rec.get("hashes") or {}, sort_keys=True))
        for cap, g in (rec.get("per_capability_gain") or {}).items():
            per_cap_gains[cap].append(float(g))
    write_csv(out / "assimilation" / "CONTENT_SENSITIVITY.csv", rows)
    write_skillopt_port_audit(out / "assimilation" / "SKILLOPT_PORT_AUDIT.md")
    gains = [float(r["future_gain"]) for r in rows]
    std = statistics.pstdev(gains) if len(gains) >= 2 else 0.0
    rng = (max(gains) - min(gains)) if gains else 0.0
    n_cap_pass = 0
    cap_stats = {}
    for cap, gs in per_cap_gains.items():
        cstd = statistics.pstdev(gs) if len(gs) >= 2 else 0.0
        cr = (max(gs) - min(gs)) if gs else 0.0
        cap_stats[cap] = {"std": cstd, "range": cr, "n": len(gs)}
        if cstd >= CONTENT_STD_GATE or cr >= CONTENT_RANGE_GATE:
            n_cap_pass += 1
    valid = (std >= CONTENT_STD_GATE or rng >= CONTENT_RANGE_GATE) and n_cap_pass >= 2
    distinct_hashes = len(set(hashes))
    decision = {
        "n_batches": len(rows),
        "std_future_gain": std,
        "range_future_gain": rng,
        "n_cap_pass": n_cap_pass,
        "cap_stats": cap_stats,
        "distinct_skill_hashes": distinct_hashes,
        "content_sensitive_assimilator_valid": bool(valid),
        "CONTENT_SENSITIVE_SKILL_BACKEND": "PASS" if valid else "FAIL",
    }
    write_json(out / "assimilation" / "ASSIMILATOR_DECISION.json", decision)
    if not valid:
        (out / "assimilation" / "CONTENT_SENSITIVE_SKILL_BACKEND").write_text("FAIL\n", encoding="utf-8")
    return decision


def analyze_matrix(out: Path) -> dict:
    root = out / "model_skill_matrix"
    rows = []
    for summary in root.rglob("summary.json"):
        parts = summary.parts
        try:
            i = parts.index("model_skill_matrix")
        except ValueError:
            continue
        rest = parts[i + 1 :]
        # only seed-level summaries: {model}/{cond}/seed_N/summary.json
        if len(rest) != 4 or not str(rest[2]).startswith("seed_") or rest[3] != "summary.json":
            continue
        rec = loadj(summary) or {}
        rec.setdefault("model", rest[0])
        rec.setdefault("condition", rest[1])
        rec.setdefault("seed", str(rest[2]).replace("seed_", ""))
        rec["path"] = str(summary.parent)
        rows.append(rec)
    skill_rows = []
    cap_rows = []
    ds_rows = []
    cost_rows = []
    for rec in rows:
        skill_rows.append(
            {
                "model": rec.get("model"),
                "condition": rec.get("condition"),
                "seed": rec.get("seed"),
                "macro": rec.get("macro"),
                "n": rec.get("n"),
                "n_skill_invocations": rec.get("n_skill_invocations"),
            }
        )
        for cap, v in (rec.get("per_capability") or {}).items():
            cap_rows.append({"model": rec.get("model"), "condition": rec.get("condition"), "seed": rec.get("seed"), "capability": cap, "em": v})
        for ds, v in (rec.get("per_dataset") or {}).items():
            ds_rows.append({"model": rec.get("model"), "condition": rec.get("condition"), "seed": rec.get("seed"), "dataset": ds, "em": v})
        cost_rows.append({"model": rec.get("model"), "condition": rec.get("condition"), "seed": rec.get("seed"), "n": rec.get("n")})
    write_csv(root / "MODEL_SKILL_MATRIX.csv", skill_rows)
    write_csv(root / "MODEL_CAPABILITY_MATRIX.csv", cap_rows)
    write_csv(root / "MODEL_DATASET_MATRIX.csv", ds_rows)
    write_csv(root / "MODEL_COST_MATRIX.csv", cost_rows)

    by = defaultdict(dict)
    for rec in skill_rows:
        by[(rec["model"], str(rec["seed"]))][rec["condition"]] = rec.get("macro")
    headroom_rows = []
    md = ["# MODEL_SKILL_HEADROOM\n"]
    for (model, seed), conds in sorted(by.items()):
        base = float(conds.get("base") or 0.0)
        best_skill = max((float(v or 0.0) for k, v in conds.items() if k != "base"), default=base)
        h_skill = best_skill - base
        headroom_rows.append({"model": model, "seed": seed, "base": base, "best_skill": best_skill, "H_skill": h_skill, **{f"em_{k}": v for k, v in conds.items()}})
        md.append(f"- {model} seed={seed}: base={base:.4f} best_skill={best_skill:.4f} H_skill={h_skill:.4f} pass={h_skill >= H_SKILL_GATE}\n")
    write_csv(root / "HEADROOM.csv", headroom_rows)
    (root / "MODEL_SKILL_HEADROOM.md").write_text("".join(md), encoding="utf-8")
    return {"n": len(skill_rows)}


def analyze_parametric(out: Path) -> dict:
    root = out / "assimilation" / "parametric"
    rows = []
    if root.exists():
        for summary in root.glob("*/summary.json"):
            rec = loadj(summary) or {}
            rec.setdefault("model", summary.parent.name)
            rec["path"] = str(summary.parent)
            rows.append(rec)
    write_csv(out / "assimilation" / "CARRIER_MATRIX.csv", rows)
    h_vals = [float(r.get("H_carrier") or 0.0) for r in rows]
    decision = {
        "n_models": len(rows),
        "H_carrier_mean": mean(h_vals) if h_vals else 0.0,
        "H_carrier_max": max(h_vals) if h_vals else 0.0,
        "n_carrier_pass": sum(1 for h in h_vals if h >= H_CARRIER_GATE),
        "rows": [
            {
                "model": r.get("model"),
                "future_base": r.get("future_base"),
                "future_skill": r.get("future_skill"),
                "future_parametric": r.get("future_parametric"),
                "H_carrier": r.get("H_carrier"),
                "retention_delta_parametric": r.get("retention_delta_parametric"),
            }
            for r in rows
        ],
    }
    write_json(out / "assimilation" / "CARRIER_DECISION.json", decision)
    md = ["# Carrier headroom (Backend 3)\n", f"H_carrier_mean={decision['H_carrier_mean']:.4f} n_pass={decision['n_carrier_pass']}\n"]
    for r in decision["rows"]:
        md.append(
            f"- {r['model']}: skill={r.get('future_skill')} parametric={r.get('future_parametric')} "
            f"H_carrier={r.get('H_carrier')} retention_d={r.get('retention_delta_parametric')}\n"
        )
    (out / "assimilation" / "CARRIER_HEADROOM.md").write_text("".join(md), encoding="utf-8")
    return decision


def analyze_acquisition(out: Path) -> dict:
    root = out / "acquisition"
    rows = []
    for summary in root.rglob("summary.json"):
        rec = loadj(summary) or {}
        parts = summary.parts
        try:
            i = parts.index("acquisition")
            rec.setdefault("model", parts[i + 1])
            rec.setdefault("method", parts[i + 2])
            rec.setdefault("seed", parts[i + 3].replace("seed_", ""))
        except Exception:
            pass
        rows.append(rec)
    write_csv(root / "R7_FINAL_METRICS.csv", rows)
    per_model = defaultdict(lambda: defaultdict(list))
    for rec in rows:
        per_model[rec.get("model")][rec.get("method")].append(float(rec.get("macro") or 0.0))
    head = []
    for model, methods in per_model.items():
        means = {m: mean(vs) for m, vs in methods.items()}
        best_dep = max((means[m] for m in means if m not in {"exact_utility", "exact_ava"}), default=0.0)
        oracle = means.get("exact_utility", 0.0)
        h_policy = oracle - best_dep
        head.append({"model": model, **means, "H_policy": h_policy, "best_deployable": best_dep})
    write_csv(root / "R7_POLICY_HEADROOM.csv", head)
    write_csv(root / "R7_PER_MODEL.csv", head)
    ds_rows = []
    cap_rows = []
    curve = []
    for rec in rows:
        for ds, v in (rec.get("per_dataset") or {}).items():
            ds_rows.append({"model": rec.get("model"), "method": rec.get("method"), "seed": rec.get("seed"), "dataset": ds, "em": v})
        for cap, v in (rec.get("per_capability") or {}).items():
            cap_rows.append({"model": rec.get("model"), "method": rec.get("method"), "seed": rec.get("seed"), "capability": cap, "em": v})
        curve.append({"model": rec.get("model"), "method": rec.get("method"), "seed": rec.get("seed"), "n_acquired": rec.get("n_acquired"), "macro": rec.get("macro")})
    write_csv(root / "R7_PER_DATASET.csv", ds_rows)
    write_csv(root / "R7_PER_CAPABILITY.csv", cap_rows)
    write_csv(root / "R7_LEARNING_CURVES.csv", curve)
    write_csv(root / "R7_COST.csv", [{"model": r.get("model"), "method": r.get("method"), "seed": r.get("seed"), "n": r.get("n")} for r in rows])
    return {"headroom": head, "n": len(rows)}


def decide(out: Path) -> dict:
    own = loadj(out / "audit" / "SKILL_OWNERSHIP_AUDIT.json") or {}
    assim = loadj(out / "assimilation" / "ASSIMILATOR_DECISION.json") or {}
    carrier = analyze_parametric(out) if (out / "assimilation" / "parametric").exists() else {}
    acq = analyze_acquisition(out) if (out / "acquisition").exists() else {"headroom": []}
    head = acq.get("headroom") or []
    n_front_win = 0
    n_oracle = 0
    h_policy_vals = []
    for row in head:
        dep = [row.get(m) for m in ("passive", "ace_skill", "coverage") if row.get(m) is not None]
        best_dep = max([float(x) for x in dep], default=0.0)
        fr = float(row.get("legacy_frontier") or 0.0)
        if fr > best_dep + 1e-9:
            n_front_win += 1
        hp = float(row.get("H_policy") or 0.0)
        h_policy_vals.append(hp)
        if hp >= ORACLE_HEADROOM_GATE:
            n_oracle += 1
    old_front = n_front_win >= 3 and bool(head)
    oracle_hr = (mean(h_policy_vals) if h_policy_vals else 0.0)
    h_carrier = carrier.get("H_carrier_mean")
    n_carrier_pass = int(carrier.get("n_carrier_pass") or 0)
    h_carrier_max = float(carrier.get("H_carrier_max") or 0.0)
    content_ok = bool(assim.get("content_sensitive_assimilator_valid"))
    parametric_done = (out / "PHASE_PARAMETRIC.DONE").exists() or bool(carrier.get("n_models"))
    if old_front:
        case = "A"
        nxt = "OLD_FRONTIER_GENERALIZES; enter learned policy but do not write only frontier acquisition"
    elif n_oracle >= 3:
        case = "B"
        nxt = "heuristic failed, experience-policy headroom exists; enter R8 learned Experience Policy"
    elif parametric_done and (oracle_hr < H_POLICY_GATE) and h_carrier_max >= 0.05:
        case = "C"
        nxt = "discard acquisition-centric ExPolicy; switch to Experience Lifecycle / Cross-Substrate Assimilation Policy"
    elif parametric_done and (not head or oracle_hr < H_POLICY_GATE) and h_carrier_max < 0.05:
        case = "D"
        nxt = "DISCARD_EXPOLICY_METHOD; pivot to Experience → Capability Construction"
    elif not content_ok and not parametric_done:
        case = "INCOMPLETE"
        nxt = "content-sensitive skill backend FAIL; Backend 3 parametric carrier measurement still running"
    else:
        case = "INCOMPLETE"
        nxt = "wait for remaining GPU jobs"
    decision = {
        "old_skill_backend_is_replacement_executor": own.get("OLD_SKILL_BACKEND") == "REPLACEMENT_EXECUTOR",
        "interventional_skill_valid": (out / "PHASE_INTERVENTIONAL.DONE").exists(),
        "content_sensitive_assimilator_valid": content_ok,
        "old_frontier_cross_model_generalizes": old_front,
        "oracle_experience_policy_headroom": oracle_hr,
        "carrier_headroom": h_carrier,
        "n_carrier_pass": n_carrier_pass,
        "pathbench_ready": (out / "audit" / "PATHBENCH_AUDIT.md").exists()
        and "pathbench_ready = false" not in (out / "audit" / "PATHBENCH_AUDIT.md").read_text(encoding="utf-8"),
        "recommended_case": case,
        "next_action": nxt,
        "n_frontier_model_wins": n_front_win,
        "n_oracle_headroom_models": n_oracle,
    }
    write_json(out / "R7_DECISION.json", decision)
    md = ["# R7_MASTER_SUMMARY\n", "```json\n" + json.dumps(decision, indent=2) + "\n```\n"]
    md.append("\nSee `audit/`, `model_skill_matrix/`, `assimilation/`, `acquisition/`.\n")
    if case == "D":
        md.append("\nExperience-policy headroom is empirically absent under current agent-learning substrates.\n")
    (out / "R7_MASTER_SUMMARY.md").write_text("".join(md), encoding="utf-8")
    return decision


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="final")
    p.add_argument("--out", default=str(R7_ROOT))
    args = p.parse_args()
    out = Path(args.out)
    if args.stage in {"content", "final"}:
        if (out / "assimilation" / "content_batches").exists():
            analyze_content(out)
    if args.stage in {"matrix", "final"}:
        if (out / "model_skill_matrix").exists():
            analyze_matrix(out)
    if args.stage in {"parametric", "final"}:
        if (out / "assimilation" / "parametric").exists():
            analyze_parametric(out)
    if args.stage in {"final", "acquisition"}:
        decide(out)
    print("analyze done", flush=True)


if __name__ == "__main__":
    main()
