#!/usr/bin/env python3
"""Analyze R5 Gate B, run CPU ablations, write reports. No learned router."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from expolicy.assimilation.carriers.skill_store import (
    Skill,
    SkillStore,
    builtin_executable_skills,
    run_skill_on_task,
)
from expolicy.assimilation.skill_assimilator import SkillAssimilator
from expolicy.core.experience import Experience
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.experiments.common import ROOT, load_world, mean, score_with_retrieval, write_json
from expolicy.experiments.r_common import add_common_args, apply_gpu

DEFAULT_OUT = str(ROOT / "outputs" / "r5_acquisition_skill")
METHODS = ("passive", "reward_seeking", "uncertainty", "frontier", "failure_replay")
PRIMARY = ("frontier", "uncertainty", "failure_replay")
CONTROLS = ("passive", "reward_seeking")
TARGET_CAPS = {
    "s1_multi_hop": "multi_hop",
    "s2_temporal": "temporal_reasoning",
    "s4_cross_doc": "cross_document_synthesis",
}
SEEDS = (0, 1, 2)


def _load_jsonl_exp(path: Path) -> list[Experience]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("run_id") not in {None, "acquire"} and "experience_id" not in payload:
            continue
        try:
            exp = Experience.model_validate(payload)
        except Exception:
            continue
        if payload.get("run_id") == "acquire" or (exp.task_id and exp.capability and "split" not in payload):
            rows.append(exp)
    # selected_experiences.jsonl is not Experience
    return rows


def _acquired(run_dir: Path) -> list[Experience]:
    selected = run_dir / "selected_experiences.jsonl"
    traj = run_dir / "trajectories.jsonl"
    if traj.exists():
        rows = []
        for line in traj.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("run_id") != "acquire":
                continue
            rows.append(Experience.model_validate(payload))
        if rows:
            return rows
    # fallback from selected ids + empty experiences (should not happen)
    return _load_jsonl_exp(selected) if False else []


def _metrics(root: Path, method: str, seed: int) -> dict | None:
    path = root / method / f"seed_{seed}" / "final_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoints(root: Path, method: str, seed: int) -> list[dict]:
    path = root / method / f"seed_{seed}" / "checkpoint_metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _eval_future_with_store(store: SkillStore | None, world, future_by_cap: dict[str, list], base_per_cap: dict) -> dict:
    env = SearchEnvironment(world)
    per_cap = {}
    ems = []
    for cap, tasks in future_by_cap.items():
        skill = store.executable_for(cap) if store is not None else None
        if skill is None:
            em = float(base_per_cap.get(cap, {}).get("em") or 0.0)
            per_cap[cap] = {"em": em, "n": len(tasks), "source": "base_fallback"}
            ems.extend([em] * len(tasks))  # keep macro as mean of task-level using cap em
            continue
        rows = [score_with_retrieval(run_skill_on_task(skill, env, t), t) for t in tasks]
        em = mean(r["em_strict"] for r in rows)
        per_cap[cap] = {"em": em, "n": len(rows), "source": "skill"}
        ems.extend(r["em_strict"] for r in rows)
    return {"em": mean(ems) if ems else 0.0, "per_cap": per_cap}


def _future_by_cap(world, split: dict) -> dict[str, list]:
    return {cap: [world.tasks[i] for i in split["future_probe"][bucket] if i in world.tasks] for bucket, cap in TARGET_CAPS.items()}


def _run_ablations(root: Path, world) -> dict:
    assimilator = SkillAssimilator(min_support=8)
    out = {"A_skill_disabled": {}, "B_shuffled": {}, "C_fixed_r2": {}}
    for seed in SEEDS:
        split = json.loads((root / "splits" / f"seed_{seed}.json").read_text(encoding="utf-8"))
        future_by_cap = _future_by_cap(world, split)
        base = json.loads((root / "base" / f"seed_{seed}" / "final_metrics.json").read_text(encoding="utf-8"))
        base_em = float(base["future"]["em"])
        base_per_cap = base.get("future_per_cap") or {}
        # C: fixed pre-installed R2 skills, once per seed
        store_c = SkillStore(root / "ablations" / f"fixed_r2_seed_{seed}")
        for skill in builtin_executable_skills():
            store_c.install(skill)
        c_eval = _eval_future_with_store(store_c, world, future_by_cap, base_per_cap)
        out["C_fixed_r2"][str(seed)] = {"future_em": c_eval["em"], "delta_future": c_eval["em"] - base_em, "per_cap": c_eval["per_cap"]}
        for method in METHODS:
            run_dir = root / method / f"seed_{seed}"
            metrics = _metrics(root, method, seed)
            if metrics is None:
                continue
            # A: skill disabled == base, because R5 writes no other carrier
            out["A_skill_disabled"].setdefault(method, {})[str(seed)] = {
                "future_em_noskill": base_em,
                "future_em_skill": metrics.get("future_em"),
                "delta_skill_minus_noskill": (metrics.get("future_em") or 0.0) - base_em,
            }
            acquired = _acquired(run_dir)
            store_b = SkillStore(root / "ablations" / method / f"seed_{seed}_shuffled")
            rng = random.Random(100 + seed)
            shuffled = []
            for exp in acquired:
                clone = exp.model_copy(deep=True)
                shuffled.append(clone)
            rng.shuffle(shuffled)
            # assign mixed experiences to each cap without capability filter
            for cap in TARGET_CAPS.values():
                skill = assimilator.fit(shuffled, cap, require_capability_match=False, version=1)
                if skill is not None:
                    store_b.install(skill)
            b_eval = _eval_future_with_store(store_b, world, future_by_cap, base_per_cap)
            out["B_shuffled"].setdefault(method, {})[str(seed)] = {
                "future_em_shuffled": b_eval["em"],
                "future_em_correct": metrics.get("future_em"),
                "delta_correct_minus_shuffled": (metrics.get("future_em") or 0.0) - b_eval["em"],
                "per_cap": b_eval["per_cap"],
            }
    write_json(root / "ablations" / "ablation_metrics.json", out)
    return out


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.pstdev(xs)


def _bootstrap_ci(xs: list[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    if not xs:
        return float("nan"), float("nan")
    stats = []
    for _ in range(n):
        sample = [xs[rng.randrange(len(xs))] for _ in xs]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    lo = stats[int(alpha / 2 * n)]
    hi = stats[min(n - 1, int((1 - alpha / 2) * n))]
    return lo, hi


def analyze(root: Path, world) -> dict:
    table = []
    by_method = defaultdict(list)
    missing = []
    for method in METHODS:
        for seed in SEEDS:
            row = _metrics(root, method, seed)
            if row is None:
                missing.append(f"{method}/seed_{seed}")
                continue
            table.append(row)
            by_method[method].append(row)
    per_seed_rows = []
    for row in table:
        per_seed_rows.append(
            {
                "method": row["method"],
                "seed": row["seed"],
                "delta_future": row.get("delta_future"),
                "future_em": row.get("future_em"),
                "retention_em": row.get("retention_em"),
                "delta_retention": row.get("delta_retention"),
                "immediate_reward": row.get("immediate_reward"),
                "n_active_skills": row.get("n_active_skills"),
                "caps_chosen": json.dumps(row.get("caps_chosen") or {}),
                "first_skill_activation": json.dumps(row.get("first_skill_activation") or {}),
            }
        )

    summary_methods = {}
    for method in METHODS:
        rows = by_method[method]
        deltas = [float(r["delta_future"]) for r in rows if r.get("delta_future") is not None]
        mu, sd = _mean_std(deltas)
        lo, hi = _bootstrap_ci(deltas, seed=hash(method) % 10_000)
        cap_gain = defaultdict(list)
        for r in rows:
            base_caps = None
            # per-cap gain vs this seed's base
            seed = r["seed"]
            base = json.loads((root / "base" / f"seed_{seed}" / "final_metrics.json").read_text(encoding="utf-8"))
            for cap, payload in (r.get("future_per_cap") or {}).items():
                b = float((base.get("future_per_cap") or {}).get(cap, {}).get("em") or 0.0)
                cap_gain[cap].append(float(payload.get("em") or 0.0) - b)
        summary_methods[method] = {
            "n_seeds": len(rows),
            "mean_delta_future": mu,
            "std_delta_future": sd,
            "ci95": [lo, hi],
            "per_seed_delta_future": deltas,
            "mean_immediate_reward": mean(float(r.get("immediate_reward") or 0.0) for r in rows),
            "mean_n_active_skills": mean(float(r.get("n_active_skills") or 0.0) for r in rows),
            "per_cap_mean_gain": {k: mean(v) for k, v in cap_gain.items()},
        }

    passive_mu = summary_methods.get("passive", {}).get("mean_delta_future", float("nan"))
    grain = 1.0 / 36.0
    primary_better = []
    for name in PRIMARY:
        mu = summary_methods.get(name, {}).get("mean_delta_future", float("-inf"))
        primary_better.append(mu > passive_mu)
    cond1 = sum(primary_better) >= 2

    cond2_hits = []
    for name in PRIMARY:
        rows_a = by_method.get(name) or []
        rows_p = {r["seed"]: r for r in by_method.get("passive") or []}
        paired = []
        seed_wins = 0
        for r in rows_a:
            p = rows_p.get(r["seed"])
            if p is None or r.get("delta_future") is None or p.get("delta_future") is None:
                continue
            d = float(r["delta_future"]) - float(p["delta_future"])
            paired.append(d)
            if float(r["delta_future"]) > float(p["delta_future"]):
                seed_wins += 1
        mu_d = mean(paired) if paired else float("-inf")
        cond2_hits.append(
            {
                "method": name,
                "mean_gap_vs_passive": mu_d,
                "seed_wins": seed_wins,
                "n": len(paired),
                "pass": mu_d >= 0.05 and seed_wins >= math.ceil(2 * max(len(paired), 1) / 3),
            }
        )
    cond2 = any(x["pass"] for x in cond2_hits)
    winner = next((x["method"] for x in cond2_hits if x["pass"]), None)
    cond3 = bool(winner and summary_methods[winner]["mean_delta_future"] > 0)
    cond4 = False
    if winner:
        gains = summary_methods[winner]["per_cap_mean_gain"]
        cond4 = sum(1 for v in gains.values() if v >= 0) >= 2
    # amplitude
    cond5_ambiguous = True
    if winner:
        gap = next(x["mean_gap_vs_passive"] for x in cond2_hits if x["method"] == winner)
        cond5_ambiguous = abs(gap) < grain - 1e-12

    if missing:
        decision = "INCOMPLETE"
        reason = f"missing runs: {missing}"
    elif cond1 and cond2 and cond3 and cond4 and not cond5_ambiguous:
        decision = "PASS"
        reason = "primary active methods beat passive with non-trivial, multi-capability gain"
    elif cond1 and cond2 and cond3 and cond4 and cond5_ambiguous:
        decision = "AMBIGUOUS"
        reason = "gap is within single-task probe grain (1/36)"
    else:
        decision = "FAIL"
        reason = "active acquisition did not reliably beat passive under executable assimilation"

    ablations = _run_ablations(root, world)

    curves = []
    for method in METHODS:
        for seed in SEEDS:
            for row in _checkpoints(root, method, seed):
                curves.append(
                    {
                        "method": method,
                        "seed": seed,
                        "budget": row.get("budget"),
                        "future_em": (row.get("future") or {}).get("em"),
                        "delta_future": row.get("delta_future"),
                        "n_active_skills": row.get("n_active_skills"),
                    }
                )

    activations = []
    for method in METHODS:
        for seed in SEEDS:
            path = root / method / f"seed_{seed}" / "skill_activation.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                activations.append({"method": method, "seed": seed, **{k: ev.get(k) for k in ("budget", "capability", "n_support", "version", "skill_id")}})

    payload = {
        "decision": decision,
        "reason": reason,
        "missing": missing,
        "conditions": {
            "1_two_of_three_primary_mean_gt_passive": cond1,
            "2_one_primary_gap_ge_0.05_and_2of3_seeds": cond2,
            "3_winner_delta_future_gt_0": cond3,
            "4_two_of_three_caps_nonneg": cond4,
            "5_not_single_task_grain": not cond5_ambiguous,
        },
        "grain": grain,
        "passive_mean_delta_future": passive_mu,
        "primary_vs_passive": cond2_hits,
        "methods": summary_methods,
        "note": "R5 tests acquisition under validated executable assimilation; it does not prove deployable learned routing.",
    }
    write_json(root / "R5_SUMMARY.json", payload)

    with (root / "R5_PER_SEED.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed_rows[0].keys()) if per_seed_rows else ["method", "seed"])
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with (root / "R5_LEARNING_CURVE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", "budget", "future_em", "delta_future", "n_active_skills"])
        writer.writeheader()
        writer.writerows(curves)
    with (root / "R5_SKILL_ACTIVATION.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", "budget", "capability", "n_support", "version", "skill_id"])
        writer.writeheader()
        writer.writerows(activations)

    gate_md = [
        "# R5 Gate B",
        "",
        f"**Decision: {decision}**",
        "",
        reason,
        "",
        "R5 证明/否定的是 acquisition under validated assimilation，不证明 deployable learned routing.",
        "",
        "## Frozen criteria",
        "",
        "1. 2/3 primary (frontier, uncertainty, failure_replay) 3-seed mean ΔFuture > passive",
        "2. ≥1 primary: mean(Δ_active−Δ_passive) ≥ 0.05 and better on ≥2/3 seeds",
        "3. that method mean ΔFuture > 0 vs clean base",
        "4. ≥2/3 target capabilities have nonnegative mean gain",
        "5. gap not within single-task grain 1/36",
        "",
        "## Condition results",
        "",
    ]
    for k, v in payload["conditions"].items():
        gate_md.append(f"- `{k}`: **{'PASS' if v else 'FAIL'}**")
    gate_md += ["", "## Method means", ""]
    for method, row in summary_methods.items():
        gate_md.append(
            f"- `{method}`: mean ΔFuture={row['mean_delta_future']:.4f} ± {row['std_delta_future']:.4f} "
            f"CI95={row['ci95']} immediate={row['mean_immediate_reward']:.3f} active_skills={row['mean_n_active_skills']:.2f} "
            f"per-cap={ {k: round(v, 3) for k, v in row['per_cap_mean_gain'].items()} }"
        )
    gate_md += ["", "## Primary vs passive", ""]
    for row in cond2_hits:
        gate_md.append(f"- {row}")
    (root / "R5_GATE_B.md").write_text("\n".join(gate_md) + "\n", encoding="utf-8")

    report = [
        "# R5 REPORT",
        "",
        "R5 repairs the acquisition test behind a validated executable assimilation path.",
        f"Gate B is **{decision}** under this controlled setting.",
        "",
        payload["note"],
        "",
        "## Causal wiring",
        "",
        "clean agent → acquire under π → SkillAssimilator.fit(real experiences) → capability-conditioned executable skill → future eval.",
        "Skills are frozen R2 primitives with experience-conditioned activation/config. Empty/shuffled batches cannot load unmodified r2_spec skills.",
        "Main eval uses oracle capability dispatch (S0 retention never gets S1/S2/S4 procedures).",
        "",
        f"## Gate B: {decision}",
        "",
        reason,
        "",
        "## Ablations",
        "",
        "A. Skill disabled equals clean base (R5 writes no other carrier). Active+Skill − Active+NoSkill = ΔFuture.",
        "B. Shuffled unfiltered fit vs correct experience-conditioned fit.",
        "C. Fixed pre-installed R2 skills (diagnostic only; not in Gate B).",
        "",
        json.dumps({k: "see ablations/ablation_metrics.json" for k in ablations}, indent=2),
        "",
        "Learned carrier routing and weight internalization remain unsupported (R4).",
        "",
    ]
    (root / "R5_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(root / "ablations" / "summary.json", {k: v for k, v in ablations.items()})
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    args = p.parse_args()
    apply_gpu(args.gpu)
    root = Path(args.output_dir)
    world = load_world(Path(args.world))
    payload = analyze(root, world)
    print(json.dumps({"decision": payload["decision"], "reason": payload["reason"], "conditions": payload["conditions"]}, indent=2))


if __name__ == "__main__":
    main()
