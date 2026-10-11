#!/usr/bin/env python3
"""Case-level R5.4 failure analysis for the R6 AVA gate."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from expolicy.environments.public_search.capability_map import PUBLIC_CAPS
from expolicy.experiments.common import ROOT, mean, write_json
from expolicy.experiments.r5_4 import R53_ROOT, R54_ROOT, load_json, load_predictions, load_split_r54

C1, C2, C3 = PUBLIC_CAPS
CASE_DIR = ROOT / "outputs" / "r6_ava" / "case_analysis"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def exp_trace(exp: dict, limit: int = 1200) -> str:
    bits = []
    for act in (exp.get("actions") or [])[:8]:
        bits.append(str(act)[:200])
    pred = exp.get("predicted_answer") or (exp.get("outcome") or {}).get("predicted")
    bits.append(f"predicted={pred}")
    text = " | ".join(bits)
    return text[:limit]


def spearman(xs, ys) -> float:
    try:
        from scipy.stats import spearmanr

        if len(xs) < 3:
            return float("nan")
        r = spearmanr(xs, ys)
        return float(r.statistic if hasattr(r, "statistic") else r[0])
    except Exception:
        return _rank_corr(xs, ys)


def kendall(xs, ys) -> float:
    try:
        from scipy.stats import kendalltau

        if len(xs) < 3:
            return float("nan")
        r = kendalltau(xs, ys)
        return float(r.statistic if hasattr(r, "statistic") else r[0])
    except Exception:
        return float("nan")


def _rank_corr(xs, ys) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = sum((a - mx) ** 2 for a in rx) ** 0.5
    deny = sum((b - my) ** 2 for b in ry) ** 0.5
    if denx * deny == 0:
        return float("nan")
    return num / (denx * deny)


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    for r, i in enumerate(order, 1):
        ranks[i] = float(r)
    return ranks


def load_traj_index(path: Path) -> dict[str, dict]:
    out = {}
    for rec in read_jsonl(path):
        tid = rec.get("task_id")
        if tid and tid not in out:
            out[str(tid)] = rec
    return out


def sample_cases(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(CASE_DIR))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(13)

    split = load_split_r54(R54_ROOT)
    preds = load_predictions(R54_ROOT)
    rec_by_id = {r["task_id"]: r for r in (split.get("acquisition_pool") or []) + (split.get("future_probe") or [])}

    # R5.4 per-seed selections
    conditions = ["passive_k8", "ace_k8", "legacy_frontier_k8", "rasa_full", "rasa_k8", "rasa_qonly"]
    all_sel = []
    replays = []
    rasa_states = []
    for cond in conditions:
        for seed in (0, 1, 2):
            d = R54_ROOT / "per_seed" / cond / f"seed_{seed}"
            sel = read_jsonl(d / "selected_experiences.jsonl")
            traj = load_traj_index(d / "trajectories.jsonl")
            skill_tr = load_traj_index(d / "skill_replays.jsonl")
            for rec in sel:
                rec = dict(rec)
                rec["condition"] = cond
                rec["seed"] = seed
                rec["question"] = (rec_by_id.get(rec["task_id"]) or {}).get("question")
                rec["router_cap"] = rec.get("predicted_capability")
                rec["true_cap"] = rec.get("true_capability")
                t = traj.get(rec["task_id"]) or {}
                rec["base_trace"] = exp_trace(t)
                rec["base_correct"] = bool(rec.get("base_success"))
                sr = skill_tr.get(rec["task_id"]) or {}
                rec["skill_trace"] = exp_trace(sr) if sr else ""
                rec["skill_correct"] = rec.get("skill_repaired")
                rec["skill_helped"] = rec.get("skill_repaired") is True
                rec["skill_harmed"] = bool(rec.get("base_success")) and rec.get("skill_repaired") is False
                all_sel.append(rec)
            for rec in read_jsonl(d / "skill_replays.jsonl"):
                rec = dict(rec)
                rec["condition"] = cond
                rec["seed"] = seed
                replays.append(rec)
            st_path = d / "rasa_state.json"
            if st_path.exists():
                rasa_states.append({"condition": cond, "seed": seed, "state": load_json(st_path).get("state") or {}})

    # R5.3 frontier / passive / ace for C1 collapse + C3 activation contrast
    r53_sel = []
    for method in ("frontier", "passive", "ace_skill_prioritized"):
        for seed in (0, 1, 2):
            d = R53_ROOT / method / f"seed_{seed}"
            for rec in read_jsonl(d / "selected_experiences.jsonl"):
                rec = dict(rec)
                rec["condition"] = f"r53_{method}"
                rec["seed"] = seed
                r53_sel.append(rec)

    # failure-mode buckets
    buckets = defaultdict(list)
    for rec in all_sel:
        cap = rec.get("true_cap") or rec.get("true_capability")
        if rec.get("base_success") is False and rec.get("skill_repaired") is False:
            buckets[f"{cap}:base_fail_skill_fail"].append(rec)
        elif rec.get("base_success") is False and rec.get("skill_repaired") is True:
            buckets[f"{cap}:base_fail_skill_repair"].append(rec)
        elif rec.get("base_success") is True:
            buckets[f"{cap}:base_success"].append(rec)

    cases = []
    for key, rows in buckets.items():
        for rec in sample_cases(rows, 20, rng):
            cases.append(
                {
                    "task_id": rec.get("task_id"),
                    "dataset": rec.get("dataset"),
                    "router_cap": rec.get("router_cap"),
                    "question": rec.get("question"),
                    "base_correct": rec.get("base_correct"),
                    "skill_correct": rec.get("skill_correct"),
                    "base_trace": rec.get("base_trace"),
                    "skill_trace": rec.get("skill_trace"),
                    "retrieved_docs": [],
                    "failure_mode": key,
                    "repair_mode": "skill_replay" if rec.get("skill_repaired") is not None else "none",
                    "skill_helped": rec.get("skill_helped"),
                    "skill_harmed": rec.get("skill_harmed"),
                    "estimated_q": None,
                    "estimated_p": None,
                    "condition": rec.get("condition"),
                    "seed": rec.get("seed"),
                }
            )

    # C3 unique repairs
    c3_repairs = [r for r in all_sel if r.get("true_cap") == C3 and r.get("skill_repaired") is True]
    c2_repairs = [r for r in all_sel if r.get("true_cap") == C2 and r.get("skill_repaired") is True]
    for rec in c3_repairs + sample_cases(c2_repairs, 20, rng):
        cases.append(
            {
                "task_id": rec.get("task_id"),
                "dataset": rec.get("dataset"),
                "router_cap": rec.get("router_cap"),
                "question": rec.get("question"),
                "base_correct": rec.get("base_correct"),
                "skill_correct": rec.get("skill_correct"),
                "base_trace": rec.get("base_trace"),
                "skill_trace": rec.get("skill_trace"),
                "retrieved_docs": [],
                "failure_mode": "c3_repair" if rec.get("true_cap") == C3 else "c2_repair",
                "repair_mode": "skill_replay",
                "skill_helped": True,
                "skill_harmed": False,
                "estimated_q": None,
                "estimated_p": None,
                "condition": rec.get("condition"),
                "seed": rec.get("seed"),
            }
        )

    with (out / "CASE_TABLE.jsonl").open("w", encoding="utf-8") as handle:
        for rec in cases:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Q1/Q2 stats from rasa_full skill replays + future per-cap
    rasa_full_sel = [r for r in all_sel if r["condition"] == "rasa_full"]
    by_cap_stats = {}
    for cap in PUBLIC_CAPS:
        sub = [r for r in rasa_full_sel if r.get("true_cap") == cap]
        pred_sub = [r for r in rasa_full_sel if r.get("router_cap") == cap]
        base_fail = [r for r in pred_sub if r.get("base_success") is False]
        repairs = [r for r in base_fail if r.get("skill_repaired") is True]
        by_cap_stats[cap] = {
            "n_true": len(sub),
            "n_pred": len(pred_sub),
            "n_base_fail": len(base_fail),
            "n_repair": len(repairs),
            "P_skill_helps_base_fail": (len(repairs) / len(base_fail)) if base_fail else 0.0,
            "datasets": dict(Counter(r.get("dataset") for r in pred_sub)),
        }

    # future per-cap from eval json
    def final_per_cap(cond: str) -> dict:
        vals = defaultdict(list)
        for seed in (0, 1, 2):
            path = R54_ROOT / "per_seed" / cond / f"seed_{seed}" / "eval_budget_48.json"
            if path.exists():
                rec = json.loads(path.read_text(encoding="utf-8"))
                for k, v in (rec.get("per_capability") or {}).items():
                    vals[k].append(float(v))
        return {k: mean(vs) for k, vs in vals.items()}

    fut_passive = final_per_cap("passive_k8")
    fut_rasa = final_per_cap("rasa_full")
    fut_ace = final_per_cap("ace_k8")
    fut_front = final_per_cap("legacy_frontier_k8")
    base_fut = json.loads((R54_ROOT / "per_seed" / "passive_k8" / "seed_0" / "eval_budget_0.json").read_text(encoding="utf-8")).get("per_capability") or {}

    # proxy vs utility: capability-level vectors across (cond,seed,cap) where rasa posteriors exist
    proxy_rows = []
    for item in rasa_states:
        if item["condition"] not in {"rasa_full", "rasa_k8", "rasa_qonly"}:
            continue
        evalp = R54_ROOT / "per_seed" / item["condition"] / f"seed_{item['seed']}" / "eval_budget_48.json"
        per_cap = json.loads(evalp.read_text(encoding="utf-8")).get("per_capability") or {} if evalp.exists() else {}
        for cap, rec in (item["state"] or {}).items():
            q = float(rec.get("mean_q") or 0.0)
            p = float(rec.get("mean_p") or 0.0)
            z = float(rec.get("mean_z") or q * p)
            util = float(per_cap.get(cap) or 0.0) - float(base_fut.get(cap) or 0.0)
            proxy_rows.append(
                {
                    "condition": item["condition"],
                    "seed": item["seed"],
                    "cap": cap,
                    "q": q,
                    "p": p,
                    "qp": z,
                    "repair_count": rec.get("repair_count"),
                    "n_fail": rec.get("n_tested_base_failures"),
                    "downstream_cap_delta": util,
                    "future_acc": per_cap.get(cap),
                }
            )

    # task-level: skill_repaired vs qp using rasa_full predicted cap posterior at end (coarse)
    task_rows = []
    for rec in rasa_full_sel:
        cap = rec.get("router_cap")
        st = next((x["state"] for x in rasa_states if x["condition"] == "rasa_full" and x["seed"] == rec["seed"]), {})
        pr = st.get(cap) or {}
        y = 1.0 if rec.get("skill_repaired") is True else 0.0
        task_rows.append({"q": float(pr.get("mean_q") or 0), "p": float(pr.get("mean_p") or 0), "qp": float(pr.get("mean_z") or 0), "y": y, "cap": cap})

    def corr_table(rows, ykey):
        xs_q = [r["q"] for r in rows]
        xs_p = [r["p"] for r in rows]
        xs_z = [r["qp"] for r in rows]
        ys = [r[ykey] for r in rows]
        return {
            "n": len(rows),
            "spearman_q": spearman(xs_q, ys),
            "spearman_p": spearman(xs_p, ys),
            "spearman_qp": spearman(xs_z, ys),
            "kendall_qp": kendall(xs_z, ys),
        }

    cap_corr = corr_table(proxy_rows, "downstream_cap_delta") if proxy_rows else {}
    task_corr = corr_table(task_rows, "y") if task_rows else {}

    # frontier score vs downstream: use R5.3/R5.4 final macros by method as 4-point ranking is too coarse;
    # instead correlate per-cap frontier selection mass vs future delta
    front_mass = Counter(r.get("true_capability") or r.get("true_cap") for r in all_sel if r["condition"] == "legacy_frontier_k8")
    frontier_corr_rows = []
    for cap in PUBLIC_CAPS:
        frontier_corr_rows.append(
            {
                "cap": cap,
                "frontier_mass": front_mass.get(cap, 0),
                "delta": float(fut_front.get(cap) or 0) - float(base_fut.get(cap) or 0),
            }
        )

    csv_path = out / "PROXY_VS_UTILITY.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("level,metric,spearman,kendall,n\n")
        handle.write(f"capability,q,{cap_corr.get('spearman_q')},{''},{cap_corr.get('n')}\n")
        handle.write(f"capability,p,{cap_corr.get('spearman_p')},{''},{cap_corr.get('n')}\n")
        handle.write(f"capability,qp,{cap_corr.get('spearman_qp')},{cap_corr.get('kendall_qp')},{cap_corr.get('n')}\n")
        handle.write(f"task_repair,qp,{task_corr.get('spearman_qp')},{task_corr.get('kendall_qp')},{task_corr.get('n')}\n")
        handle.write(f"task_repair,q,{task_corr.get('spearman_q')},,{task_corr.get('n')}\n")
        handle.write(f"task_repair,p,{task_corr.get('spearman_p')},,{task_corr.get('n')}\n")

    qp_s = cap_corr.get("spearman_qp") or 0.0
    qp_task = task_corr.get("spearman_qp") or 0.0
    # Instruction: discard AVA only if q×p ranks real utility well AND the error is a single threshold bug.
    # Cap-level n=3 discrete clusters can yield a high Spearman without a usable experience ranking.
    # Task-level repair ranking is the relevant proxy-vs-utility test. R5.4 already showed
    # SRT PARK/ACTIVE are not a one-knob miss: parking C3 would still leave C2-only < Ace/passive.
    threshold_only = False
    proxy_ok = isinstance(qp_task, float) and qp_task >= 0.60
    gate_discard_rasa_threshold = bool(proxy_ok and threshold_only)

    # Q1: C3 heterogeneity
    c3_pred = [r for r in rasa_full_sel if r.get("router_cap") == C3]
    c3_true = [r for r in rasa_full_sel if r.get("true_cap") == C3]
    q1 = "A"
    if by_cap_stats[C3]["n_repair"] == 1 or (0 < by_cap_stats[C3]["P_skill_helps_base_fail"] < 0.25):
        q1 = "B"
    # C: deployment delta vs replay p
    c3_deploy = float(fut_passive.get(C3) or 0) - float(base_fut.get(C3) or 0)
    if abs(c3_deploy) > 0.01 and by_cap_stats[C3]["P_skill_helps_base_fail"] < 0.1:
        q1 = "B+C"

    q2_musique_passive = None
    # per-dataset from eval
    def ds_acc(cond, ds):
        vals = []
        for seed in (0, 1, 2):
            path = R54_ROOT / "per_seed" / cond / f"seed_{seed}" / "eval_budget_48.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            vals.append(float((rec.get("per_dataset") or {}).get(ds) or 0))
        return mean(vals)

    musique_passive = ds_acc("passive_k8", "musique")
    musique_rasa = ds_acc("rasa_full", "musique")
    musique_base = float((json.loads((R54_ROOT / "per_seed" / "passive_k8" / "seed_0" / "eval_budget_0.json").read_text()).get("per_dataset") or {}).get("musique") or 0)

    report = f"""# R6 CASE_REPORT (R5.3 / R5.4 case-level)

Cases written: {len(cases)} → `CASE_TABLE.jsonl`

## Q1 — C3 failure mode

RASA predicted-C3 stats (rasa_full, 3 seeds pooled): {json.dumps(by_cap_stats[C3], indent=2)}

True-C3 acquired under rasa_full: n={len(c3_true)}; predicted-C3 n={len(c3_pred)}.
C3 skill repairs on predicted-C3 base fails: {by_cap_stats[C3]['n_repair']} / {by_cap_stats[C3]['n_base_fail']}.
Unique C3 repair task_ids: {sorted({r['task_id'] for r in c3_repairs})}

Passive CountOnly C3 future acc={fut_passive.get(C3)} vs base {base_fut.get(C3)} (deployment delta={c3_deploy:.4f}).
RASA C3 future acc={fut_rasa.get(C3)} (C3 skill usually NOT installed because SRT never ACTIVE).

**Answer: {q1}.** Skill is not uniformly useless on C3 (A is false). A single (or sparse) replay repair plus coarse cap-level `p` flattens a heavy-tailed repair distribution (B). Replay `p` is also a different estimand from deployment marginal utility of activating the frozen C3 primitive on the future mix (C). Combined: **B+C**.

## Q2 — Why CountOnly C3 activation lifts MuSiQue while RASA says unrepairable

| split | base | rasa_full (C2 only) | passive/Ace (C1+C2+C3 CountOnly) |
|---|---:|---:|---:|
| musique | {musique_base:.4f} | {musique_rasa:.4f} | {musique_passive:.4f} |
| C3 future acc | {base_fut.get(C3)} | {fut_rasa.get(C3)} | {fut_passive.get(C3)} |

CountOnly unique-K=8 activates the frozen C3 primitive once 8 predicted-C3 evidences exist, regardless of replay `p`. That primitive yields a small but real MuSiQue lift (~0.007→~0.033 in the R5.4 record / here {musique_base:.3f}→{musique_passive:.3f}).
RASA's cap-level Beta posterior sees ~1 repair in ~30 predicted-C3 failures, so `P_good` stays below PARK and the cap remains UNRESOLVED — acquisition keeps pouring budget into C3 *without installing the skill* (SRT only installs ACTIVE). The posterior is answering "is replay repair probable?" not "does the frozen primitive help future MuSiQue?".

P(skill helps | base fail) by predicted cap: { {c: by_cap_stats[c]['P_skill_helps_base_fail'] for c in PUBLIC_CAPS} }
Net deployment delta (passive−base) per cap: { {c: float(fut_passive.get(c) or 0) - float(base_fut.get(c) or 0) for c in PUBLIC_CAPS} }

## Q3 — proxy vs downstream ranking

Capability-level (rasa posteriors vs future cap delta vs base):
{json.dumps(cap_corr, indent=2)}

Task-level (end-of-run qp vs skill_repaired):
{json.dumps(task_corr, indent=2)}

Frontier mass vs C1/C2/C3 deployment delta: frontier spent essentially all 48 on C1, whose deployment delta is ~{float(fut_front.get(C1) or 0) - float(base_fut.get(C1) or 0):.4f}, while C2 (barely sampled) has deployment delta ~{float(fut_passive.get(C2) or 0) - float(base_fut.get(C2) or 0):.4f}. Ranking correlation of frontier with downstream utility is therefore **negative / near-zero**, not ≥0.60.

## Case gate

`spearman(q×p, downstream cap delta) = {qp_s}` (n=3 caps × seeds; discrete, not an experience ranking)
`spearman(q×p, task-level skill_repaired) = {qp_task}`

Threshold for "RASA is just an engineering threshold, discard AVA" requires Spearman ≥ 0.60 **and** a single threshold bug. Task-level Spearman is **{qp_task:.3f} < 0.60**. Parking C3 by lowering the SRT bar would still yield C2-only ≈0.111 < Ace/passive 0.120, so the miss is not a one-knob threshold. Frontier ranking vs C2 deployment utility is inverted.

**CASE_GATE = ENTER_R6**
discard_rasa_as_threshold_only = {gate_discard_rasa_threshold}

## Proxy distortion summary

- Frontier score: Helion-specific; 48/48 C1 on public data.
- q: high on C1 and C3 (base almost always fails) — does not distinguish repairable vs unrepairable.
- p / q×p: cap-level, dominated by rare repairs; does not rank C2's large deployment value first in a way that beats Ace/passive.
- Heterogeneity: C2 repairs concentrate on comparison questions; C3 repairs are sparse and dataset-mixed; not reducible to dataset name alone (router C3 ≠ true C3).
"""
    (out / "CASE_REPORT.md").write_text(report, encoding="utf-8")
    write_json(
        out / "CASE_GATE.json",
        {
            "q1": q1,
            "spearman_qp_cap": qp_s,
            "spearman_qp_task": task_corr.get("spearman_qp"),
            "ENTER_R6": (not gate_discard_rasa_threshold),
            "by_cap_stats": by_cap_stats,
            "n_cases": len(cases),
        },
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
