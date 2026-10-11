#!/usr/bin/env python3
"""R5: active acquisition behind experience-conditioned executable skill assimilation."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from filelock import FileLock

from expolicy.acquisition.base import make_acquisition
from expolicy.assimilation.carriers.skill_store import Skill, SkillStore
from expolicy.assimilation.skill_assimilator import SkillAssimilator, maps_to_skill_capability
from expolicy.core.experience import Experience
from expolicy.core.state import empty_agent
from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
from expolicy.experiments.common import ROOT, build_static_agent, load_world, mean, write_json
from expolicy.experiments.r_common import (
    add_common_args,
    apply_gpu,
    compact_metrics,
    load_e0_experiences,
    metrics_without_rows,
    write_run_header,
)
from expolicy.logging.event_logger import append_jsonl

DEFAULT_OUT = str(ROOT / "outputs" / "r5_acquisition_skill")
METHODS = ("passive", "reward_seeking", "uncertainty", "frontier", "failure_replay", "frontier_capnorm")
REFIT_AT = (8, 16, 24, 32, 40, 48)
EVAL_AT = (0, 16, 32, 48)
TARGET_CAPS = {
    "s1_multi_hop": "multi_hop",
    "s2_temporal": "temporal_reasoning",
    "s4_cross_doc": "cross_document_synthesis",
}


def _heartbeat(out: Path, **payload) -> None:
    payload = {"updated_at": time.time(), "pid": os.getpid(), **payload}
    write_json(out / "STATUS_LIVE.json", payload)


def _load_jsonl_exp(path: Path) -> list[Experience]:
    rows: list[Experience] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(Experience.model_validate(json.loads(line)))
    return rows


def _tasks(world, ids: list[str]):
    return [world.tasks[i] for i in ids if i in world.tasks]


def _eval_tasks(agent, state, tasks, *, out: Path, phase: str, acquired: int, budget: int) -> dict:
    from expolicy.experiments.common import score_with_retrieval

    rows = []
    for i, task in enumerate(tasks, 1):
        _heartbeat(out, phase=phase, acquired=acquired, budget=budget, eval_i=i, eval_n=len(tasks), task_id=task.task_id)
        exp = agent.run_task(task, state=state)
        row = score_with_retrieval(exp, task)
        row["experience_id"] = exp.experience_id
        row["predicted"] = exp.predicted_answer
        row["gold"] = task.gold_answer
        row["task_id"] = task.task_id
        rows.append(row)
    return compact_metrics(rows)


def _eval_agent(
    agent,
    state,
    future_by_cap: dict[str, list],
    retention: list,
    *,
    condition: str,
    out: Path,
    acquired: int,
    budget: int,
) -> dict:
    per_cap = {}
    future_rows = []
    for cap, tasks in future_by_cap.items():
        payload = _eval_tasks(
            agent, state, tasks, out=out, phase=f"eval_future_{cap}", acquired=acquired, budget=budget
        )
        per_cap[cap] = metrics_without_rows(payload)
        for row in payload["rows"]:
            row = dict(row)
            row["capability"] = cap
            row["condition"] = condition
            future_rows.append(row)
    future = compact_metrics(future_rows)
    ret = _eval_tasks(agent, state, retention, out=out, phase="eval_retention", acquired=acquired, budget=budget)
    return {
        "future": metrics_without_rows(future),
        "future_per_cap": per_cap,
        "retention": metrics_without_rows(ret),
        "future_rows": future_rows,
        "retention_rows": ret["rows"],
        "n_active_skills": sum(
            1
            for s in (agent.skill_store.skills.values() if agent.skill_store else [])
            if s.status == "active" and s.executable
        ),
        "active_skills": [
            {
                "capability": s.capability,
                "skill_id": s.skill_id,
                "version": s.version,
                "n_support": s.n_support,
                "activation_reason": s.activation_reason,
            }
            for s in (agent.skill_store.skills.values() if agent.skill_store else [])
            if s.status == "active" and s.executable
        ],
    }


def _save_eval_rows(path: Path, rows: list[dict], *, split: str, budget: int) -> None:
    for row in rows:
        append_jsonl(path, {"split": split, "budget": budget, **row})


def _configure_dispatch(agent, store: SkillStore | None, mode: str) -> None:
    if mode == "disabled" or store is None:
        agent.skill_store = None
        agent.use_executable_skill = False
        return
    agent.skill_store = store
    agent.use_executable_skill = True


def _refit_and_install(
    assimilator: SkillAssimilator,
    store: SkillStore,
    acquired: list[Experience],
    *,
    versions: dict[str, int],
    skill_dir: Path,
    probe_ids: list[str],
    gold_docs: list[str],
    budget: int,
) -> list[dict]:
    events = []
    for cap in TARGET_CAPS.values():
        matched_n = sum(1 for e in acquired if maps_to_skill_capability(e.capability) == cap)
        if matched_n < assimilator.min_support:
            continue
        versions[cap] = versions.get(cap, 0) + 1
        skill = assimilator.fit(
            acquired,
            cap,
            version=versions[cap],
            probe_ids=probe_ids,
            gold_doc_ids=gold_docs,
        )
        if skill is None:
            continue
        store.install(skill)
        path = skill_dir / f"skill_v{versions[cap]:03d}_{cap}.json"
        write_json(path, skill.model_dump(mode="json"))
        events.append(
            {
                "budget": budget,
                "capability": cap,
                "skill_id": skill.skill_id,
                "version": skill.version,
                "n_support": skill.n_support,
                "support_success_rate": skill.support_success_rate,
                "support_failure_rate": skill.support_failure_rate,
                "activation_reason": skill.activation_reason,
                "runtime_config": skill.runtime_config,
                "source_experience_ids": skill.source_experience_ids,
                "path": str(path),
            }
        )
    return events


def _replay_policy(policy, pool, acquired: list[Experience], prior, dummy_state, diagnoser):
    remaining = list(pool)
    recent = list(prior) + list(acquired)
    # Advance policy RNG the same way as the original sequential loop.
    profile = diagnoser.diagnose(prior) if prior else diagnoser.diagnose(acquired or prior)
    recent_stream: list[Experience] = list(prior)
    remaining_live = list(pool)
    for exp in acquired:
        profile = diagnoser.diagnose(recent_stream) if recent_stream else profile
        policy.propose(dummy_state, profile, remaining_live, 1, recent=recent_stream)
        remaining_live = [t for t in remaining_live if t.task_id != exp.task_id]
        recent_stream.append(exp)
    return remaining_live, diagnoser.diagnose(recent)


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    p.add_argument("--method", required=True, choices=[*METHODS, "base"])
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--splits", default=None)
    p.add_argument("--base_dir", default=None)
    p.add_argument("--dispatch", default="oracle_capability", choices=("oracle_capability", "disabled", "blind_global"))
    p.add_argument("--min_support", type=int, default=8)
    args = p.parse_args()
    apply_gpu(args.gpu)
    root_out = Path(args.output_dir)
    if args.method == "base":
        out = root_out / "base" / f"seed_{args.seed}"
    else:
        out = root_out / args.method / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    skill_dir = out / "skill_versions"
    skill_dir.mkdir(parents=True, exist_ok=True)
    write_run_header(
        out,
        config={
            "script": "run_r5_acquisition_skill.py",
            "method": args.method,
            "seed": args.seed,
            "model": args.model,
            "gpu": args.gpu,
            "budget": args.budget,
            "dispatch": args.dispatch,
            "min_support": args.min_support,
            "world_seed": 13,
        },
    )
    write_json(out / "config.json", json.loads((out / "config.yaml").read_text()) if False else {
        "script": "run_r5_acquisition_skill.py",
        "method": args.method,
        "seed": args.seed,
        "model": args.model,
        "gpu": args.gpu,
        "budget": args.budget,
        "dispatch": args.dispatch,
        "min_support": args.min_support,
        "world_seed": 13,
        "refit_at": list(REFIT_AT),
        "eval_at": list(EVAL_AT),
    })
    if (out / "DONE").exists() and args.resume:
        print(json.dumps({"status": "already_done", "out": str(out)}))
        return

    world = load_world(Path(args.world))
    split_path = Path(args.splits) if args.splits else root_out / "splits" / f"seed_{args.seed}.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    future_by_cap = {cap: _tasks(world, split["future_probe"][bucket]) for bucket, cap in TARGET_CAPS.items()}
    retention = _tasks(world, split["retention_probe"])
    pool = _tasks(world, split["acquisition_pool"])
    probe_ids = list(split["future_probe_ids"]) + list(split["retention_probe"])
    gold_docs = [d for tasks in future_by_cap.values() for t in tasks for d in t.gold_doc_ids]

    blocked = set(probe_ids)
    e0 = [e for e in load_e0_experiences() if e.task_id not in blocked]
    diagnoser = CapabilityDiagnoser()
    prior_profile = diagnoser.diagnose(e0)

    _heartbeat(out, phase="load_model", acquired=0, budget=args.budget)
    agent, base_state = build_static_agent(world, model_path=args.model, output_dir=out / "base_agent")
    store = SkillStore(out / "skills")
    assimilator = SkillAssimilator(min_support=args.min_support)
    dummy_state = empty_agent(out / "dummy", model_name=args.model)

    base_dir = Path(args.base_dir) if args.base_dir else root_out / "base" / f"seed_{args.seed}"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_metrics_path = base_dir / "final_metrics.json"

    def run_base_eval() -> dict:
        _configure_dispatch(agent, None, "disabled")
        _heartbeat(out, phase="base_eval", acquired=0, budget=args.budget)
        payload = _eval_agent(
            agent,
            base_state,
            future_by_cap,
            retention,
            condition="base",
            out=out,
            acquired=0,
            budget=args.budget,
        )
        slim = {k: v for k, v in payload.items() if k not in {"future_rows", "retention_rows"}}
        write_json(base_metrics_path, slim)
        write_json(base_dir / "token_stats.json", {"future_tokens": payload["future"]["tokens"], "retention_tokens": payload["retention"]["tokens"]})
        _save_eval_rows(base_dir / "trajectories.jsonl", payload["future_rows"], split="future", budget=0)
        _save_eval_rows(base_dir / "trajectories.jsonl", payload["retention_rows"], split="retention", budget=0)
        (base_dir / "DONE").write_text("ok\n", encoding="utf-8")
        return slim

    if not base_metrics_path.exists():
        lock = FileLock(str(base_dir / "eval.lock"))
        with lock:
            if not base_metrics_path.exists():
                run_base_eval()
    base_metrics = json.loads(base_metrics_path.read_text(encoding="utf-8"))
    write_json(out / "base_metrics.json", base_metrics)

    if args.method == "base":
        (out / "DONE").write_text("ok\n", encoding="utf-8")
        print(json.dumps({"method": "base", "seed": args.seed, "future_em": base_metrics["future"]["em"]}, indent=2))
        return

    acquired = _load_jsonl_exp(out / "trajectories.jsonl") if args.resume else []
    # trajectories.jsonl also stores eval rows; keep only acquisition experiences
    if acquired:
        acquired = [e for e in acquired if e.run_id == "acquire"]
    if not acquired and args.resume:
        acquired = _load_jsonl_exp(out / "selected_experiences.jsonl")

    versions: dict[str, int] = {}
    for path in sorted(skill_dir.glob("skill_v*.json")):
        skill = Skill.model_validate(json.loads(path.read_text(encoding="utf-8")))
        store.install(skill)
        versions[skill.capability or ""] = max(versions.get(skill.capability or "", 0), skill.version)

    policy = make_acquisition(args.method, seed=args.seed)
    if acquired:
        remaining, profile = _replay_policy(policy, pool, acquired, e0, dummy_state, diagnoser)
    else:
        remaining = list(pool)
        profile = prior_profile

    checkpoint_path = out / "checkpoint_metrics.jsonl"
    done_evals = set()
    if checkpoint_path.exists() and args.resume:
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_evals.add(int(json.loads(line)["budget"]))

    activation_log = out / "skill_activation.jsonl"
    t0 = time.time()
    first_activated: dict[str, int] = {}

    def maybe_eval(budget: int) -> dict | None:
        if budget not in EVAL_AT or budget in done_evals:
            return None
        _configure_dispatch(agent, store, args.dispatch)
        _heartbeat(out, phase="eval", acquired=budget, budget=args.budget)
        payload = _eval_agent(
            agent,
            base_state,
            future_by_cap,
            retention,
            condition=f"{args.method}@{budget}",
            out=out,
            acquired=budget,
            budget=args.budget,
        )
        slim = {
            "budget": budget,
            "method": args.method,
            "seed": args.seed,
            "future": payload["future"],
            "future_per_cap": payload["future_per_cap"],
            "retention": payload["retention"],
            "n_active_skills": payload["n_active_skills"],
            "active_skills": payload["active_skills"],
            "delta_future": payload["future"]["em"] - base_metrics["future"]["em"],
            "delta_retention": payload["retention"]["em"] - base_metrics["retention"]["em"],
        }
        append_jsonl(checkpoint_path, slim)
        _save_eval_rows(out / "eval_trajectories.jsonl", payload["future_rows"], split="future", budget=budget)
        _save_eval_rows(out / "eval_trajectories.jsonl", payload["retention_rows"], split="retention", budget=budget)
        done_evals.add(budget)
        return slim

    if 0 not in done_evals:
        # budget 0 equals shared base; don't rerun LLM
        append_jsonl(
            checkpoint_path,
            {
                "budget": 0,
                "method": args.method,
                "seed": args.seed,
                "future": base_metrics["future"],
                "future_per_cap": base_metrics["future_per_cap"],
                "retention": base_metrics["retention"],
                "n_active_skills": 0,
                "active_skills": [],
                "delta_future": 0.0,
                "delta_retention": 0.0,
            },
        )
        done_evals.add(0)

    recent = list(e0) + list(acquired)
    while len(acquired) < args.budget:
        n = len(acquired)
        profile = diagnoser.diagnose(recent) if recent else prior_profile
        props = policy.propose(dummy_state, profile, remaining, 1, recent=recent)
        if not props or not props[0].task_ids:
            break
        tid = props[0].task_ids[0]
        task = next((t for t in remaining if t.task_id == tid), None)
        if task is None:
            task = remaining[0]
        remaining = [t for t in remaining if t.task_id != task.task_id]
        _configure_dispatch(agent, None, "disabled")  # acquire with clean runtime, not the skill being learned
        _heartbeat(out, phase="acquire", acquired=n, budget=args.budget, task_id=task.task_id)
        exp = agent.run_task(task, state=base_state)
        exp.run_id = "acquire"
        acquired.append(exp)
        recent.append(exp)
        append_jsonl(out / "trajectories.jsonl", json.loads(exp.model_dump_json()))
        append_jsonl(
            out / "selected_experiences.jsonl",
            {
                "step": len(acquired),
                "task_id": task.task_id,
                "capability": task.capability,
                "stage": task.stage,
                "success": exp.success,
                "experience_id": exp.experience_id,
                "tokens": exp.cost.tokens,
                "tool_calls": exp.cost.tool_calls,
                "predicted": exp.predicted_answer,
            },
        )
        n = len(acquired)
        if n in REFIT_AT:
            events = _refit_and_install(
                assimilator,
                store,
                acquired,
                versions=versions,
                skill_dir=skill_dir,
                probe_ids=probe_ids,
                gold_docs=gold_docs,
                budget=n,
            )
            for ev in events:
                append_jsonl(activation_log, ev)
                cap = ev["capability"]
                if cap not in first_activated:
                    first_activated[cap] = n
        if n in EVAL_AT:
            maybe_eval(n)

    last = maybe_eval(len(acquired))
    if last is None and checkpoint_path.exists():
        rows = [json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        last = max(rows, key=lambda r: int(r.get("budget") or 0)) if rows else None
    cap_counts = dict(Counter((e.capability or "unknown") for e in acquired))
    final_future = last["future"]["em"] if last else base_metrics["future"]["em"]
    summary = {
        "method": args.method,
        "seed": args.seed,
        "n_acquired": len(acquired),
        "immediate_reward": mean(float(bool(e.success)) for e in acquired),
        "caps_chosen": cap_counts,
        "first_skill_activation": first_activated,
        "n_active_skills": last["n_active_skills"] if last else len([s for s in store.skills.values() if s.status == "active"]),
        "base_future_em": base_metrics["future"]["em"],
        "base_retention_em": base_metrics["retention"]["em"],
        "future_em": last["future"]["em"] if last else None,
        "future_per_cap": last["future_per_cap"] if last else None,
        "retention_em": last["retention"]["em"] if last else None,
        "delta_future": (last["future"]["em"] - base_metrics["future"]["em"]) if last else None,
        "delta_retention": (last["retention"]["em"] - base_metrics["retention"]["em"]) if last else None,
        "u_capability": (
            (last["future"]["em"] - base_metrics["future"]["em"])
            - 0.5 * max(0.0, -(last["retention"]["em"] - base_metrics["retention"]["em"]))
            if last
            else None
        ),
        "tokens": int(sum(e.cost.tokens for e in acquired)),
        "tool_calls": int(sum(e.cost.tool_calls for e in acquired)),
        "wall_seconds": time.time() - t0,
        "dispatch": args.dispatch,
        "note": "R5 tests acquisition under validated executable assimilation; it does not prove deployable learned routing.",
    }
    write_json(out / "final_metrics.json", summary)
    write_json(
        out / "token_stats.json",
        {"acquire_tokens": summary["tokens"], "acquire_tool_calls": summary["tool_calls"], "wall_seconds": summary["wall_seconds"]},
    )
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
