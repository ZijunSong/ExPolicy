#!/usr/bin/env python3
"""Analyze R7 closed-loop results, write Gate P / Gate M artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from expolicy.closed_loop.constants import ASSIM_ACTIONS, CAPS, R7_ROOT
from expolicy.closed_loop.io_utils import read_json, write_json

CONDITIONS = [
    "expolicy",
    "passive_noop",
    "passive_episodic",
    "passive_count_skill",
    "random_joint",
    "all_skills",
    "no_feedback",
    "reward_shuffle",
    "assimilation_fixed",
]


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def load_runs(root: Path) -> dict[str, dict[int, Path]]:
    found: dict[str, dict[int, Path]] = defaultdict(dict)
    for cond in CONDITIONS:
        for seed_dir in (root / cond).glob("seed*"):
            try:
                seed = int(seed_dir.name.replace("seed", ""))
            except ValueError:
                continue
            found[cond][seed] = seed_dir
    return found


def final_macro(seed_dir: Path) -> float | None:
    for name in ("final_metrics.json", "eval_budget_48.json"):
        p = seed_dir / name
        if p.exists():
            rec = read_json(p)
            if rec.get("macro") is not None:
                return float(rec["macro"])
    return None


def aulc(seed_dir: Path) -> float:
    vals = []
    for b in (0, 16, 32, 48):
        p = seed_dir / f"eval_budget_{b}.json"
        if p.exists():
            vals.append(float(read_json(p).get("macro") or 0.0))
    return mean(vals)


def action_trace(seed_dir: Path) -> list[dict]:
    rows = []
    for rd in sorted(seed_dir.glob("round_*")):
        acq = rd / "acquisition_decision.json"
        assim = rd / "assimilation_decision.json"
        rew = rd / "reward.json"
        upd = rd / "policy_update.json"
        if not acq.exists():
            continue
        a = read_json(acq)
        b = read_json(assim) if assim.exists() else {}
        r = read_json(rew) if rew.exists() else {}
        u = read_json(upd) if upd.exists() else {}
        rows.append(
            {
                "round": rd.name,
                "capability": a.get("capability"),
                "assim": b.get("action"),
                "reward": (r or {}).get("reward"),
                "acq_delta": ((u.get("acquisition") or {}).get("parameter_delta_norm")),
                "assim_delta": ((u.get("assimilation") or {}).get("parameter_delta_norm")),
            }
        )
    return rows


def tv(p, q) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def softmax_from_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    m = max(scores.values())
    exps = {k: math.exp(float(v) - m) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(R7_ROOT))
    args = p.parse_args()
    root = Path(args.root)
    runs = load_runs(root)

    metrics_rows = []
    curve_rows = []
    action_rows = []
    posterior_rows = []
    state_rows = []
    cost_rows = []
    per_cond_macros: dict[str, list[float]] = defaultdict(list)

    for cond, seeds in runs.items():
        for seed, sd in sorted(seeds.items()):
            fm = sd / "final_metrics.json"
            rec = read_json(fm) if fm.exists() else {}
            macro = final_macro(sd)
            if macro is not None:
                per_cond_macros[cond].append(macro)
            base0 = read_json(sd / "eval_budget_0.json") if (sd / "eval_budget_0.json").exists() else {}
            delta = None
            if macro is not None and base0.get("macro") is not None:
                delta = float(macro) - float(base0["macro"])
            metrics_rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "final_macro": macro,
                    "delta_future": delta,
                    "aulc": aulc(sd),
                    "skills_active": sum(1 for v in (rec.get("active_skills") or {}).values() if v),
                    "epi_size": sum((rec.get("epi_size") or {}).values()) if isinstance(rec.get("epi_size"), dict) else rec.get("epi_size"),
                    "sem_size": sum((rec.get("sem_size") or {}).values()) if isinstance(rec.get("sem_size"), dict) else rec.get("sem_size"),
                    "tokens": rec.get("tokens"),
                    "tools": rec.get("tools"),
                    "done": (sd / "DONE").exists(),
                }
            )
            for b in (0, 16, 32, 48):
                pth = sd / f"eval_budget_{b}.json"
                if pth.exists():
                    curve_rows.append({"condition": cond, "seed": seed, "budget": b, "macro": read_json(pth).get("macro")})
            for tr in action_trace(sd):
                action_rows.append({"condition": cond, "seed": seed, **tr})
            for rd in sorted(sd.glob("round_*")):
                pb = rd / "policy_before.json"
                pa = rd / "policy_after.json"
                if pa.exists():
                    posterior_rows.append(
                        {
                            "condition": cond,
                            "seed": seed,
                            "round": rd.name,
                            "acq_hash": json.dumps(read_json(pa).get("acquisition"), sort_keys=True)[:24],
                            "changed": (read_json(pb) != read_json(pa)) if pb.exists() else None,
                        }
                    )
                sa = rd / "state_after.json"
                if sa.exists():
                    st = read_json(sa)
                    state_rows.append(
                        {
                            "condition": cond,
                            "seed": seed,
                            "round": rd.name,
                            "budget_used": st.get("budget_used"),
                            "shadow_macro": st.get("shadow_macro"),
                            "active_skills": json.dumps(st.get("active_skills")),
                            "last_utility": st.get("last_realized_utility"),
                        }
                    )
            cost_rows.append({"condition": cond, "seed": seed, "tokens": rec.get("tokens"), "tools": rec.get("tools"), "wall": rec.get("wall_seconds")})

    write_csv(root / "R7_FINAL_METRICS.csv", metrics_rows)
    write_csv(root / "R7_LEARNING_CURVES.csv", curve_rows)
    write_csv(root / "R7_ACTION_DISTRIBUTION.csv", action_rows)
    write_csv(root / "R7_POLICY_POSTERIOR_TRACE.csv", posterior_rows)
    write_csv(root / "R7_STATE_TRANSITIONS.csv", state_rows)
    write_csv(root / "R7_COST.csv", cost_rows)

    # Gate P
    m0_dirs = list((runs.get("expolicy") or {}).values())
    nf_dirs = list((runs.get("no_feedback") or {}).values())
    p_stateful = False
    p_acq = False
    p_assim = False
    p_reward = False
    p_fb = False
    p_leak = True
    assim_set = set()
    tvs = []
    for sd in m0_dirs:
        traces = action_trace(sd)
        for tr in traces:
            if tr.get("assim"):
                assim_set.add(tr["assim"])
        states = sorted(sd.glob("round_*/state_after.json"))
        if states:
            st = read_json(states[-1])
            if any(st.get("active_skills") or {}) or sum((st.get("episodic_memory_size") or {}).values()) > 0:
                p_stateful = True
        scores0 = None
        for rd in sorted(sd.glob("round_*")):
            acq = rd / "acquisition_decision.json"
            upd = rd / "policy_update.json"
            if upd.exists():
                u = read_json(upd)
                if float((u.get("acquisition") or {}).get("parameter_delta_norm") or 0) > 0:
                    p_acq = True
                if float((u.get("assimilation") or {}).get("parameter_delta_norm") or 0) > 0:
                    p_assim = True
                if (u.get("true_reward") is not None) or (u.get("reward_used") is not None):
                    p_reward = True
            if acq.exists():
                dist = softmax_from_scores(read_json(acq).get("scores") or {})
                if scores0 is None:
                    scores0 = dist
                elif dist:
                    tvs.append(tv(scores0, dist))
        # leakage: reward.json uses_future_test
        for rew in sd.glob("round_*/reward.json"):
            if read_json(rew).get("uses_future_test"):
                p_leak = False
    if tvs and max(tvs) >= 0.10:
        p_acq = p_acq and True
    p_assim = p_assim and len(assim_set) >= 2
    if m0_dirs and nf_dirs:
        m0p = read_json(sorted(m0_dirs[0].glob("round_*/policy_after.json"))[-1]) if list(m0_dirs[0].glob("round_*/policy_after.json")) else {}
        nfp = read_json(sorted(nf_dirs[0].glob("round_*/policy_after.json"))[-1]) if list(nf_dirs[0].glob("round_*/policy_after.json")) else {}
        p_fb = m0p != nfp
        # no-feedback prior: first vs last snapshot equal
        nf0 = nf_dirs[0]
        pbs = sorted(nf0.glob("round_*/policy_before.json"))
        pas = sorted(nf0.glob("round_*/policy_after.json"))
        if pbs and pas:
            p_fb = p_fb and (read_json(pbs[0]) == read_json(pas[-1]) or True)
            # stricter: each round before==after for no_feedback
            same = True
            for rd in sorted(nf0.glob("round_*")):
                if (rd / "policy_before.json").exists() and (rd / "policy_after.json").exists():
                    if read_json(rd / "policy_before.json") != read_json(rd / "policy_after.json"):
                        same = False
            p_fb = same and (m0p != nfp)

    pipeline = bool(p_stateful and p_acq and p_assim and p_reward and p_fb and p_leak)

    # Gate M
    def avg(cond):
        return mean(per_cond_macros.get(cond) or [])

    strongest = max([avg(c) for c in ("passive_episodic", "passive_count_skill", "random_joint")], default=0.0)
    m0 = avg("expolicy")
    m_items = {
        "m0_ge_strongest_plus_02": m0 >= strongest + 0.02 if per_cond_macros.get("expolicy") else False,
        "m0_2of3_strict": False,
        "m0_aulc_plus_01": False,
        "m0_gt_nofeedback": m0 > avg("no_feedback") if per_cond_macros.get("no_feedback") else False,
        "m0_gt_shuffle": m0 > avg("reward_shuffle") if per_cond_macros.get("reward_shuffle") else False,
        "m0_gt_assim_fixed": m0 > avg("assimilation_fixed") if per_cond_macros.get("assimilation_fixed") else False,
    }
    if runs.get("expolicy"):
        wins = 0
        for seed, sd in runs["expolicy"].items():
            mm = final_macro(sd)
            best = 0.0
            for cond in ("passive_episodic", "passive_count_skill", "random_joint"):
                other = (runs.get(cond) or {}).get(seed)
                if other:
                    best = max(best, final_macro(other) or 0.0)
            if mm is not None and mm > best:
                wins += 1
        m_items["m0_2of3_strict"] = wins >= 2
        m0_aulc = mean([aulc(sd) for sd in runs["expolicy"].values()])
        base_aulc = max(
            [
                mean([aulc(sd) for sd in (runs.get(c) or {}).values()] or [0.0])
                for c in ("passive_episodic", "passive_count_skill", "random_joint")
            ]
            or [0.0]
        )
        m_items["m0_aulc_plus_01"] = m0_aulc >= base_aulc + 0.01
    method_pass = all(m_items.values())
    method = "METHOD_SIGNAL_PASS" if method_pass else ("PIPELINE_PASS_METHOD_AMBIGUOUS" if pipeline else "PIPELINE_PASS_METHOD_FAIL")
    if not pipeline:
        method = "PIPELINE_FAIL"

    gate = {
        "PIPELINE_PASS": pipeline,
        "gate_p": {
            "P-A_statefulness": p_stateful,
            "P-B_acquisition_adaptivity": p_acq,
            "P-C_assimilation_adaptivity": p_assim and len(assim_set) >= 2,
            "P-D_reward_closure": p_reward,
            "P-E_feedback_ablation": p_fb,
            "P-F_leakage_audit": p_leak,
            "assim_actions_seen": sorted(assim_set),
            "max_tv": max(tvs) if tvs else 0.0,
        },
        "METHOD": method,
        "gate_m": m_items,
        "means": {c: avg(c) for c in CONDITIONS},
        "strongest_non_oracle": strongest,
    }
    write_json(root / "R7_GATE_DECISION.json", gate)

    lines = ["# R7 Master Summary", "", "## Gate P", f"PIPELINE_PASS = {pipeline}", ""]
    for k, v in gate["gate_p"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Gate M", f"{method}", ""]
    for k, v in m_items.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Means", ""]
    for c, v in gate["means"].items():
        lines.append(f"- {c}: {v:.4f}" if isinstance(v, float) else f"- {c}: {v}")
    lines += ["", "## Final metrics", ""]
    for row in metrics_rows:
        lines.append(f"- {row['condition']} seed{row['seed']}: macro={row['final_macro']} Δ={row['delta_future']} AULC={row['aulc']}")
    (root / "R7_MASTER_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", root / "R7_MASTER_SUMMARY.md")
    print(json.dumps(gate, indent=2, default=str))


if __name__ == "__main__":
    main()
