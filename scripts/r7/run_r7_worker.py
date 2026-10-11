#!/usr/bin/env python3
"""GPU worker for R7 phases. Resume-safe. One CUDA device per process."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from expolicy.assimilation.carriers.skill_store import SkillStore
from expolicy.environments.public_search.skills import count_only_public_skill, install_count_only
from expolicy.experiments.common import ROOT, write_json
from expolicy.experiments.r_common import apply_gpu
from expolicy.r7.constants import (
    CHECKPOINTS,
    MODEL_ROOT,
    PUBLIC_CAPS,
    R7_DATA,
    R7_ROOT,
    RETRIEVER_URL,
)
from expolicy.r7.interventional_skills import install_interventional
from expolicy.r7.runtime import (
    build_agent,
    eval_base,
    eval_with_backend,
    heartbeat,
    load_json,
    records_to_tasks,
    wait_retriever,
    world_from_tasks,
)
from expolicy.r7.skillopt import SkillOptAssimilator, skill_hash


def load_suite(root: Path) -> dict:
    return json.loads((root / "split.json").read_text(encoding="utf-8"))


def tasks_of(split: dict, key: str, per_dataset: int | None = None):
    recs = list(split.get(key) or [])
    if per_dataset is None:
        return records_to_tasks(recs, split=key)
    by = {}
    out = []
    for rec in recs:
        ds = rec.get("dataset")
        by.setdefault(ds, 0)
        if by[ds] >= per_dataset:
            continue
        by[ds] += 1
        out.append(rec)
    return records_to_tasks(out, split=key)


def done(path: Path) -> bool:
    return (path / "DONE").exists() and (path / "summary.json").exists()


def mark(path: Path, summary: dict) -> None:
    write_json(path / "summary.json", {k: v for k, v in summary.items() if k != "rows"})
    if "rows" in summary:
        write_json(path / "summary_full.json", summary)
    (path / "DONE").write_text("ok\n", encoding="utf-8")


def _copy_eval_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("rows.jsonl", "summary.json", "DONE"):
        sp = src / name
        if sp.exists():
            (dst / name).write_bytes(sp.read_bytes())


def _load_marked(path: Path) -> dict:
    rows = []
    rp = path / "rows.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    summary = load_json(path / "summary.json") if (path / "summary.json").exists() else {}
    summary["rows"] = rows
    return summary


def phase_matrix(args: argparse.Namespace) -> None:
    apply_gpu(args.gpu)
    wait_retriever(args.retriever_url)
    split = load_suite(Path(args.suite))
    tasks = tasks_of(split, "future_probe", per_dataset=args.per_dataset)
    val_tasks = tasks_of(split, "skill_validation", per_dataset=min(40, args.per_dataset))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if done(out):
        print(f"skip matrix job done {out}", flush=True)
        return
    shared_base = out.parent.parent / "_shared_base"
    model_path = str(MODEL_ROOT / args.model)
    agent_dir = out / "agent"
    world = world_from_tasks(tasks + val_tasks)
    agent, state, backend, env = build_agent(
        world, model_path=model_path, output_dir=agent_dir, retriever_url=args.retriever_url
    )
    base_dir = out / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    if not done(base_dir):
        if (shared_base / "rows.jsonl").exists():
            _copy_eval_dir(shared_base, base_dir)
            if not (base_dir / "DONE").exists():
                (base_dir / "DONE").write_text("ok\n", encoding="utf-8")
            base = _load_marked(base_dir)
        else:
            base = eval_base(agent, tasks, out=base_dir)
            mark(base_dir, base)
            if not (shared_base / "rows.jsonl").exists():
                _copy_eval_dir(base_dir, shared_base)
    else:
        base = _load_marked(base_dir)
        if not (shared_base / "rows.jsonl").exists():
            _copy_eval_dir(base_dir, shared_base)
    base_rows = {r["task_id"]: r for r in base["rows"]}

    cond = args.condition
    cond_dir = out / cond
    cond_dir.mkdir(parents=True, exist_ok=True)
    if cond == "base":
        summary = {k: v for k, v in base.items() if k != "rows"}
        summary["rows"] = base["rows"]
        summary["n_skill_invocations"] = 0
        summary["n_base_fallback"] = len(base["rows"])
        summary["condition"] = "base"
    elif done(cond_dir):
        summary = _load_marked(cond_dir)
    else:
        store = SkillStore(cond_dir / "store")
        if cond == "old_replacement_skill":
            for cap in PUBLIC_CAPS:
                store.install(count_only_public_skill(cap, n_support=64, min_support=8, version=1))
            summary = eval_with_backend(tasks, condition=cond, store=store, env=env, model=backend, base_rows=base_rows, out=cond_dir)
        elif cond == "interventional_skill":
            install_interventional(store)
            summary = eval_with_backend(tasks, condition=cond, store=store, env=env, model=backend, base_rows=base_rows, out=cond_dir)
        elif cond == "content_optimized_skill":
            art = Path(args.optimized_skills)
            if not art.exists():
                raise SystemExit(f"optimized skills missing: {art}")
            payload = json.loads(art.read_text())
            from expolicy.assimilation.carriers.skill_store import Skill

            for item in payload.get("skills") or []:
                store.install(Skill.model_validate(item))
            summary = eval_with_backend(tasks, condition="interventional_skill", store=store, env=env, model=backend, base_rows=base_rows, out=cond_dir)
        else:
            raise SystemExit(f"unknown condition {cond}")
        mark(cond_dir, summary)
    summary["model"] = args.model
    summary["seed"] = args.seed
    write_json(out / "summary.json", {k: v for k, v in summary.items() if k != "rows"})
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    heartbeat(out, phase="matrix_done", condition=cond)


def phase_interventional_validate(args: argparse.Namespace) -> None:
    apply_gpu(args.gpu)
    wait_retriever(args.retriever_url)
    split = load_suite(Path(args.suite))
    tasks = tasks_of(split, "skill_validation", per_dataset=40)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_path = str(MODEL_ROOT / args.model)
    world = world_from_tasks(tasks)
    agent, state, backend, env = build_agent(world, model_path=model_path, output_dir=out / "agent", retriever_url=args.retriever_url)
    base = eval_base(agent, tasks, out=out / "base")
    mark(out / "base", base)
    base_rows = {r["task_id"]: r for r in base["rows"]}
    old_store = SkillStore(out / "old_store")
    for cap in PUBLIC_CAPS:
        old_store.install(count_only_public_skill(cap, n_support=64, min_support=8, version=1))
    old = eval_with_backend(tasks, condition="old_replacement_skill", store=old_store, env=env, model=backend, base_rows=base_rows, out=out / "old")
    mark(out / "old", old)
    new_store = SkillStore(out / "new_store")
    install_interventional(new_store)
    new = eval_with_backend(tasks, condition="interventional_skill", store=new_store, env=env, model=backend, base_rows=base_rows, out=out / "new")
    mark(out / "new", new)
    tokens = [int(r.get("model_tokens_after_skill") or 0) for r in new["rows"] if r.get("dispatch_source", "").startswith("interventional")]
    report = {
        "model": args.model,
        "base_macro": base.get("macro"),
        "old_macro": old.get("macro"),
        "new_macro": new.get("macro"),
        "delta_old": (old.get("macro") or 0) - (base.get("macro") or 0),
        "delta_new": (new.get("macro") or 0) - (base.get("macro") or 0),
        "per_capability_base": base.get("per_capability"),
        "per_capability_new": new.get("per_capability"),
        "min_model_tokens_after_skill": min(tokens) if tokens else 0,
        "frac_tokens_gt0": (sum(1 for t in tokens if t > 0) / len(tokens)) if tokens else 0.0,
        "final_answer_source": "policy_model",
    }
    write_json(out / "VALIDATE.json", report)
    (out / "DONE").write_text("ok\n")


def phase_content_batch(args: argparse.Namespace) -> None:
    """Optimize skills on one experience batch; validate on frozen skill_validation."""
    apply_gpu(args.gpu)
    wait_retriever(args.retriever_url)
    split = load_suite(Path(args.suite))
    acq = tasks_of(split, "acquisition_pool")
    val = tasks_of(split, "skill_validation", per_dataset=40)
    fut = tasks_of(split, "future_probe", per_dataset=args.per_dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if done(out):
        return
    rng_seed = int(args.batch_id)
    import random

    rng = random.Random(20260815 + rng_seed)
    # 12 content-different batches: stratified slices of acquisition
    by_cap = {c: [t for t in acq if t.capability == c] for c in PUBLIC_CAPS}
    batch_tasks = []
    for cap, pool in by_cap.items():
        rng.shuffle(pool)
        chunk = max(8, len(pool) // 12)
        start = (int(args.batch_id) % 12) * chunk
        batch_tasks.extend(pool[start : start + chunk] or pool[:8])
    model_path = str(MODEL_ROOT / args.model)
    world = world_from_tasks(batch_tasks + val + fut)
    agent, state, backend, env = build_agent(world, model_path=model_path, output_dir=out / "agent", retriever_url=args.retriever_url)
    experiences = []
    for i, task in enumerate(batch_tasks):
        heartbeat(out, phase="acquire_batch", i=i, n=len(batch_tasks), task_id=task.task_id)
        experiences.append(agent.run_task(task))
    # validation closure: interventional eval on val split for this skill only (cheap proxy: n<=40/ds)
    val_base_path = out / "val_base"
    if not done(val_base_path):
        vb = eval_base(agent, val, out=val_base_path)
        mark(val_base_path, vb)
    else:
        rows = [json.loads(l) for l in (val_base_path / "rows.jsonl").read_text().splitlines() if l.strip()]
        vb = {"rows": rows, **load_json(val_base_path / "summary.json")}
    val_base_rows = {r["task_id"]: r for r in vb["rows"]}

    def validate(skill) -> float:
        store = SkillStore(out / f"tmp_{skill.skill_id}")
        store.install(skill)
        cap_tasks = [t for t in val if t.capability == skill.capability]
        if not cap_tasks:
            return 0.0
        summary = eval_with_backend(
            cap_tasks,
            condition="interventional_skill",
            store=store,
            env=env,
            model=backend,
            base_rows=val_base_rows,
            out=out / f"val_{skill.skill_id}",
        )
        return float(summary.get("macro") or 0.0)

    opt = SkillOptAssimilator()
    result = opt.optimize(experiences, validate=validate)
    skills_dump = [sk.model_dump(mode="json") for sk in result["skills"].values()]
    write_json(
        out / "optimized.json",
        {
            "batch_id": args.batch_id,
            "model": args.model,
            "hashes": result["hashes"],
            "accepted": result["accepted"],
            "rejected": result["rejected"],
            "analyses": result["analyses"],
            "skills": skills_dump,
        },
    )
    store = SkillStore(out / "store")
    for sk in result["skills"].values():
        store.install(sk)
    fut_base = eval_base(agent, fut, out=out / "future_base") if not done(out / "future_base") else {
        "rows": [json.loads(l) for l in (out / "future_base" / "rows.jsonl").read_text().splitlines() if l.strip()],
        **load_json(out / "future_base" / "summary.json"),
    }
    if not done(out / "future_base"):
        mark(out / "future_base", fut_base)
    fut_rows = {r["task_id"]: r for r in fut_base["rows"]}
    fut_skill = eval_with_backend(
        fut,
        condition="interventional_skill",
        store=store,
        env=env,
        model=backend,
        base_rows=fut_rows,
        out=out / "future_skill",
    )
    mark(out / "future_skill", fut_skill)
    gain = float(fut_skill.get("macro") or 0.0) - float(fut_base.get("macro") or 0.0)
    write_json(
        out / "summary.json",
        {
            "batch_id": args.batch_id,
            "model": args.model,
            "future_base": fut_base.get("macro"),
            "future_skill": fut_skill.get("macro"),
            "future_gain": gain,
            "hashes": result["hashes"],
            "per_capability_gain": {
                cap: float((fut_skill.get("per_capability") or {}).get(cap) or 0.0)
                - float((fut_base.get("per_capability") or {}).get(cap) or 0.0)
                for cap in PUBLIC_CAPS
            },
        },
    )
    (out / "DONE").write_text("ok\n")


def _load_experiences(path: Path) -> list:
    from expolicy.core.experience import Experience

    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(Experience.model_validate_json(line))
    return rows


def _strip_gold(exp):
    payload = exp.model_dump()
    payload["gold_answer"] = None
    return exp.model_copy(update={"gold_answer": None})


def phase_parametric(args: argparse.Namespace) -> None:
    """Same frozen experience batch → SkillOpt vs LoRA/OPD internalization."""
    import gc
    import random

    from expolicy.assimilation.carriers.weight_update import WeightUpdater
    from expolicy.core.errors import TrainingDivergenceError
    from expolicy.core.experience import Experience
    from expolicy.logging.event_logger import append_jsonl

    apply_gpu(args.gpu)
    wait_retriever(args.retriever_url)
    split = load_suite(Path(args.suite))
    pool = tasks_of(split, "acquisition_pool")
    val = tasks_of(split, "skill_validation", per_dataset=min(40, args.per_dataset))
    fut = tasks_of(split, "future_probe", per_dataset=args.per_dataset)
    cal = tasks_of(split, "calibration", per_dataset=32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if done(out):
        return
    rng = random.Random(20260815)
    batch_tasks = []
    for cap in PUBLIC_CAPS:
        cap_pool = [t for t in pool if t.capability == cap]
        rng.shuffle(cap_pool)
        batch_tasks.extend(cap_pool[:16])
    model_path = str(MODEL_ROOT / args.model)
    world = world_from_tasks(batch_tasks + val + fut + cal)
    agent, state, backend, env = build_agent(
        world, model_path=model_path, output_dir=out / "agent", retriever_url=args.retriever_url
    )
    exp_path = out / "experiences.jsonl"
    experiences = _load_experiences(exp_path)
    done_ids = {e.task_id for e in experiences}
    for i, task in enumerate(batch_tasks):
        if task.task_id in done_ids:
            continue
        heartbeat(out, phase="parametric_collect", i=i, n=len(batch_tasks), task_id=task.task_id)
        exp = agent.run_task(task)
        append_jsonl(exp_path, json.loads(exp.model_dump_json()))
        experiences.append(exp)
        done_ids.add(task.task_id)

    shared_base = R7_ROOT / "model_skill_matrix" / args.model / "_shared_base"
    fut_base_dir = out / "future_base"
    if not done(fut_base_dir):
        if (shared_base / "rows.jsonl").exists():
            _copy_eval_dir(shared_base, fut_base_dir)
            if not (fut_base_dir / "DONE").exists():
                (fut_base_dir / "DONE").write_text("ok\n", encoding="utf-8")
        else:
            fb = eval_base(agent, fut, out=fut_base_dir)
            mark(fut_base_dir, fb)
    fut_base = _load_marked(fut_base_dir)
    fut_rows = {r["task_id"]: r for r in fut_base["rows"]}

    cal_base_dir = out / "cal_base"
    if not done(cal_base_dir):
        cb = eval_base(agent, cal, out=cal_base_dir)
        mark(cal_base_dir, cb)
    cal_base = _load_marked(cal_base_dir)
    cal_rows = {r["task_id"]: r for r in cal_base["rows"]}
    val_base_dir = out / "val_base"
    if not done(val_base_dir):
        vb = eval_base(agent, val, out=val_base_dir)
        mark(val_base_dir, vb)
    else:
        vb = _load_marked(val_base_dir)
    val_rows = {r["task_id"]: r for r in vb["rows"]}

    def validate(skill) -> float:
        store = SkillStore(out / f"tmp_{skill.skill_id}")
        store.install(skill)
        cap_tasks = [t for t in val if t.capability == skill.capability]
        if not cap_tasks:
            return 0.0
        summary = eval_with_backend(
            cap_tasks,
            condition="interventional_skill",
            store=store,
            env=env,
            model=backend,
            base_rows=val_rows,
            out=out / f"val_{skill.skill_id}",
        )
        return float(summary.get("macro") or 0.0)

    skill_dir = out / "skill"
    if not done(skill_dir):
        heartbeat(out, phase="parametric_skillopt")
        opt = SkillOptAssimilator()
        result = opt.optimize(experiences, validate=validate)
        store = SkillStore(skill_dir / "store")
        for sk in result["skills"].values():
            store.install(sk)
        write_json(
            skill_dir / "optimized.json",
            {"hashes": result["hashes"], "accepted": result["accepted"], "rejected": result["rejected"]},
        )
        fut_skill = eval_with_backend(
            fut,
            condition="interventional_skill",
            store=store,
            env=env,
            model=backend,
            base_rows=fut_rows,
            out=skill_dir / "future",
        )
        cal_skill = eval_with_backend(
            cal,
            condition="interventional_skill",
            store=store,
            env=env,
            model=backend,
            base_rows=cal_rows,
            out=skill_dir / "retention",
        )
        mark(
            skill_dir,
            {
                "future_macro": fut_skill.get("macro"),
                "retention_macro": cal_skill.get("macro"),
                "per_dataset_future": fut_skill.get("per_dataset"),
                "per_capability_future": fut_skill.get("per_capability"),
                "hashes": result["hashes"],
                "n_future": fut_skill.get("n"),
            },
        )

    skill_sum = load_json(skill_dir / "summary.json")
    lora_dir = out / "parametric"
    train_info: dict = {}
    if not done(lora_dir):
        heartbeat(out, phase="parametric_lora_train")
        train_exps = []
        for exp in experiences:
            dumped = json.loads(exp.model_dump_json())
            dumped["gold_answer"] = None
            train_exps.append(Experience.model_validate(dumped))
        try:
            del agent
        except Exception:
            pass
        try:
            del backend
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        updater = WeightUpdater(model_path, lora_dir / "lora_train")
        try:
            train_info = updater.train(train_exps, epochs=2, lr=5e-5, lora_rank=8)
            trained = True
        except TrainingDivergenceError as exc:
            train_info = {"error": str(exc), "adapter_dir": None}
            trained = False
            (lora_dir / "TRAIN_FAILED.json").write_text(json.dumps(train_info, indent=2), encoding="utf-8")
        agent, state, backend, env = build_agent(
            world, model_path=model_path, output_dir=out / "agent_lora", retriever_url=args.retriever_url
        )
        if trained and train_info.get("adapter_dir"):
            backend.load(str(train_info["adapter_dir"]))
        heartbeat(out, phase="parametric_lora_eval")
        fut_lora = eval_base(agent, fut, out=lora_dir / "future")
        cal_lora = eval_base(agent, cal, out=lora_dir / "retention")
        mark(
            lora_dir,
            {
                "future_macro": fut_lora.get("macro"),
                "retention_macro": cal_lora.get("macro"),
                "per_dataset_future": fut_lora.get("per_dataset"),
                "per_capability_future": fut_lora.get("per_capability"),
                "trained": trained,
                "train": {k: v for k, v in train_info.items() if k != "training_data_ids"},
                "n_future": fut_lora.get("n"),
            },
        )
        try:
            backend.unload_adapter()
        except Exception:
            pass

    lora_sum = load_json(lora_dir / "summary.json") if (lora_dir / "summary.json").exists() else {}
    base_macro = float(fut_base.get("macro") or 0.0)
    skill_macro = float(skill_sum.get("future_macro") or 0.0)
    lora_macro = float(lora_sum.get("future_macro") or 0.0)
    best_fixed = max(skill_macro, lora_macro)
    # per-task routing oracle if both future row files exist
    skill_rows = {}
    lora_rows = {}
    sp = skill_dir / "future" / "rows.jsonl"
    lp = lora_dir / "future" / "rows.jsonl"
    if sp.exists():
        for line in sp.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                skill_rows[rec["task_id"]] = float(rec.get("em") or 0.0)
    if lp.exists():
        for line in lp.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                lora_rows[rec["task_id"]] = float(rec.get("em") or 0.0)
    ids = sorted(set(skill_rows) | set(lora_rows) | {r["task_id"] for r in fut_base["rows"]})
    oracle_hits = []
    for tid in ids:
        vals = []
        if tid in skill_rows:
            vals.append(skill_rows[tid])
        if tid in lora_rows:
            vals.append(lora_rows[tid])
        if vals:
            oracle_hits.append(max(vals))
    oracle_carrier = sum(oracle_hits) / len(oracle_hits) if oracle_hits else best_fixed
    h_carrier = oracle_carrier - best_fixed
    cal_base_m = float(cal_base.get("macro") or 0.0)
    summary = {
        "model": args.model,
        "n_experience": len(experiences),
        "n_success": sum(1 for e in experiences if e.success),
        "future_base": base_macro,
        "future_skill": skill_macro,
        "future_parametric": lora_macro,
        "oracle_carrier": oracle_carrier,
        "best_fixed_carrier": best_fixed,
        "H_carrier": h_carrier,
        "H_skill_from_this_batch": skill_macro - base_macro,
        "retention_base": cal_base_m,
        "retention_skill": skill_sum.get("retention_macro"),
        "retention_parametric": lora_sum.get("retention_macro"),
        "retention_delta_parametric": float(lora_sum.get("retention_macro") or 0.0) - cal_base_m,
        "retention_delta_skill": float(skill_sum.get("retention_macro") or 0.0) - cal_base_m,
    }
    write_json(out / "summary.json", summary)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    heartbeat(out, phase="parametric_done", **{k: summary[k] for k in ("H_carrier", "future_skill", "future_parametric")})


def phase_acquisition(args: argparse.Namespace) -> None:
    apply_gpu(args.gpu)
    wait_retriever(args.retriever_url)
    from expolicy.acquisition.base import make_acquisition

    split = load_suite(Path(args.suite))
    pool = tasks_of(split, "acquisition_pool")
    val = tasks_of(split, "skill_validation", per_dataset=40)
    fut = tasks_of(split, "future_probe", per_dataset=args.per_dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if done(out):
        return
    model_path = str(MODEL_ROOT / args.model)
    world = world_from_tasks(pool + val + fut)
    agent, state, backend, env = build_agent(world, model_path=model_path, output_dir=out / "agent", retriever_url=args.retriever_url)
    # reuse matrix base rows if provided
    fut_base = eval_base(agent, fut, out=out / "future_base")
    mark(out / "future_base", fut_base)
    val_base = eval_base(agent, val, out=out / "val_base")
    mark(out / "val_base", val_base)
    val_rows = {r["task_id"]: r for r in val_base["rows"]}
    fut_rows = {r["task_id"]: r for r in fut_base["rows"]}

    def validate(skill) -> float:
        store = SkillStore(out / f"tmp_{skill.skill_id}")
        store.install(skill)
        cap_tasks = [t for t in val if t.capability == skill.capability][:30]
        if not cap_tasks:
            return 0.0
        s = eval_with_backend(cap_tasks, condition="interventional_skill", store=store, env=env, model=backend, base_rows=val_rows, out=out / f"val_{skill.skill_id}")
        return float(s.get("macro") or 0.0)

    method = args.method
    budget = int(args.budget)
    acquired = []
    selected = []
    remaining = list(pool)
    from expolicy.core.capability import CapabilityProfile
    from expolicy.core.state import empty_agent

    dummy = empty_agent(out / "dummy", model_name=args.model)
    profile = CapabilityProfile()
    method = args.method
    if method == "legacy_frontier":
        acq = make_acquisition("frontier", seed=int(args.seed))
    elif method == "exact_utility":
        acq = make_acquisition("passive", seed=int(args.seed))
    else:
        acq = make_acquisition(method, seed=int(args.seed))

    checkpoints = [int(x) for x in args.checkpoints]
    opt = SkillOptAssimilator()
    for b in range(0, budget, 8):
        heartbeat(out, phase="acquire", budget=b)
        take = min(8, budget - b)
        if method == "exact_utility":
            # greedy: try a small candidate set, pick the 8 with best post-assim val (expensive)
            import random

            rng = random.Random(int(args.seed) * 1000 + b)
            cand = list(remaining)
            rng.shuffle(cand)
            cand = cand[: min(24, len(cand))]
            scored = []
            for task in cand:
                exp = agent.run_task(task)
                trial = acquired + [exp]
                result = opt.optimize(trial, validate=validate)
                store = SkillStore(out / f"oracle_try_{b}_{task.task_id.replace(':', '_')}")
                for sk in result["skills"].values():
                    store.install(sk)
                s = eval_with_backend(val[:80], condition="interventional_skill", store=store, env=env, model=backend, base_rows=val_rows, out=out / f"oracle_eval_{b}_{task.task_id.replace(':', '_')}")
                scored.append((float(s.get("macro") or 0.0), task, exp))
            scored.sort(key=lambda x: x[0], reverse=True)
            pick = scored[:take]
            for sc, task, exp in pick:
                acquired.append(exp)
                selected.append({"task_id": task.task_id, "capability": task.capability, "oracle_val": sc, "step": len(acquired)})
                remaining = [t for t in remaining if t.task_id != task.task_id]
        else:
            props = acq.propose(dummy, profile, remaining, take, recent=acquired)
            ids = []
            for prop in props:
                ids.extend(prop.task_ids)
            idset = []
            for i in ids:
                if i not in idset:
                    idset.append(i)
            chosen = [t for t in remaining if t.task_id in set(idset[:take])]
            if len(chosen) < take:
                extra = [t for t in remaining if t not in chosen][: take - len(chosen)]
                chosen.extend(extra)
            for task in chosen[:take]:
                heartbeat(out, phase="acquire_task", task_id=task.task_id, n=len(acquired))
                exp = agent.run_task(task)
                acquired.append(exp)
                selected.append({"task_id": task.task_id, "capability": task.capability, "dataset": (task.metadata or {}).get("dataset"), "success": exp.success, "step": len(acquired)})
                remaining = [t for t in remaining if t.task_id != task.task_id]
        n = len(acquired)
        if n in checkpoints:
            result = opt.optimize(acquired, validate=validate)
            store = SkillStore(out / f"store_b{n}")
            for sk in result["skills"].values():
                store.install(sk)
            fut_s = eval_with_backend(fut, condition="interventional_skill", store=store, env=env, model=backend, base_rows=fut_rows, out=out / f"eval_budget_{n}")
            write_json(out / f"eval_budget_{n}.json", {k: v for k, v in fut_s.items() if k != "rows"})
    from expolicy.logging.event_logger import append_jsonl

    for row in selected:
        append_jsonl(out / "selected_experiences.jsonl", row)
    last = max(checkpoints)
    summary = load_json(out / f"eval_budget_{last}.json") if (out / f"eval_budget_{last}.json").exists() else {}
    summary.update({"model": args.model, "method": method, "seed": args.seed, "n_acquired": len(acquired)})
    write_json(out / "summary.json", summary)
    (out / "DONE").write_text("ok\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="base")
    p.add_argument("--method", default="passive")
    p.add_argument("--batch-id", dest="batch_id", type=int, default=0)
    p.add_argument("--suite", default=str(R7_DATA))
    p.add_argument("--out", required=True)
    p.add_argument("--retriever-url", dest="retriever_url", default=os.environ.get("EXPOLICY_RETRIEVER_URL", RETRIEVER_URL))
    p.add_argument("--per-dataset", dest="per_dataset", type=int, default=80)
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--checkpoints", nargs="+", default=[str(x) for x in CHECKPOINTS])
    p.add_argument("--optimized-skills", dest="optimized_skills", default="")
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    heartbeat(Path(args.out), phase=args.phase, model=args.model)
    if args.phase == "matrix":
        phase_matrix(args)
    elif args.phase == "interventional_validate":
        phase_interventional_validate(args)
    elif args.phase == "content_batch":
        phase_content_batch(args)
    elif args.phase == "parametric":
        phase_parametric(args)
    elif args.phase == "acquisition":
        phase_acquisition(args)
    elif args.phase == "compat":
        import subprocess, sys

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "r7" / "audit_model_compatibility.py"),
            "--model",
            args.model,
            "--gpu",
            str(args.gpu),
            "--real-generation-smoke",
            "8",
            "--retriever-url",
            args.retriever_url,
            "--out",
            args.out,
        ]
        raise SystemExit(subprocess.call(cmd))
    else:
        raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    main()
