"""R6 Exact AVA helpers: frozen split, CountOnly assimilator, shadow-mask cache, seed loop."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import numpy as np

from expolicy.acquisition.ace_skill import AceSkillPrioritizedAcquisition
from expolicy.acquisition.ava import ExactAVAAcquisition, RandomClusterAcquisition
from expolicy.acquisition.base import make_acquisition
from expolicy.acquisition.rasa import RASAAcquisition
from expolicy.assimilation.readiness_gate import STATE_ACTIVE, STATE_PARKED
from expolicy.core.experience import Experience
from expolicy.core.state import empty_agent
from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
from expolicy.environments.public_search.capability_map import PUBLIC_CAPS
from expolicy.environments.public_search.environment import PublicSearchEnvironment
from expolicy.environments.public_search.evaluator import score_public
from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.environments.public_search.skills import PRIMITIVES, run_public_skill_on_task
from expolicy.evaluation.shadow_probe import ALL_MASKS, load_mask_summary, mask_eval_path, store_for_mask
from expolicy.experiments.common import ROOT, write_json
from expolicy.experiments.r5_3 import (
    DEFAULT_MODEL,
    acquisition_distribution,
    build_public_agent,
    eval_with_skills,
    heartbeat,
    world_from_tasks,
)
from expolicy.experiments.r5_4 import (
    E5_MODEL,
    MIN_SUPPORT,
    R54_ROOT,
    RETRIEVER_URL,
    count_only_store_predicted,
    load_base_rows as load_r54_base_rows,
    load_clusters,
    load_predictions,
    load_router,
    records_to_tasks,
    sha256_file,
    srt_store,
)
from expolicy.experiments.r_common import apply_gpu, write_run_header
from expolicy.logging.event_logger import append_jsonl
from expolicy.state.isolated_clone import EvidenceState

R6_ROOT = ROOT / "outputs" / "r6_ava"
SPLIT_SEED = 20260816
CANDIDATE_PER_DS = 240
SHADOW_PER_DS = 60
BUDGET_DEFAULT = 48
BATCH_SIZE = 8
NUM_CANDIDATES = 8
FULL_EVAL_AT = (0, 8, 16, 24, 32, 40, 48)
DATASETS = ("hotpotqa", "2wiki", "musique")

CONDITIONS: dict[str, dict[str, Any]] = {
    "exact_ava": {"acquisition": "exact_ava", "activation": "count_only", "qonly": False, "diversity": True},
    "passive": {"acquisition": "passive", "activation": "count_only", "qonly": False, "diversity": True},
    "ace_skill_prioritized": {"acquisition": "ace_skill_prioritized", "activation": "count_only", "qonly": False, "diversity": True},
    "legacy_frontier": {"acquisition": "frontier", "activation": "count_only", "qonly": False, "diversity": True},
    "rasa_full": {"acquisition": "rasa", "activation": "srt", "qonly": False, "diversity": True},
    "random_cluster": {"acquisition": "random_cluster", "activation": "count_only", "qonly": False, "diversity": True},
}

NEEDS_REPLAY = {"rasa_full"}
NEEDS_RASA_STATE = {"rasa_full"}


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_split_r6(root: Path | None = None) -> dict[str, Any]:
    root = Path(root or R6_ROOT)
    path = root / "split_r6_ava.json"
    if not path.exists():
        raise FileNotFoundError(f"R6 split missing: {path}")
    return load_json(path)


def merged_base_rows(root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for name in ("base_eval_future", "base_eval_shadow", "base_eval"):
        d = Path(root) / name
        path = d / "rows.jsonl"
        shards = sorted(d.glob("shard_*.jsonl"))
        files = [path] if path.exists() else shards
        for fp in files:
            if not fp.exists():
                continue
            for line in fp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                tid = rec.get("task_id")
                if tid:
                    rows[str(tid)] = rec
    if not rows:
        rows.update(load_r54_base_rows(R54_ROOT))
    return rows


def copy_future_base_eval(root: Path) -> None:
    dest = Path(root) / "base_eval_future"
    dest.mkdir(parents=True, exist_ok=True)
    src = R54_ROOT / "base_eval" / "rows.jsonl"
    if src.exists() and not (dest / "rows.jsonl").exists():
        shutil.copy2(src, dest / "rows.jsonl")
        (dest / "DONE").write_text("copied_from_r54\n", encoding="utf-8")


def load_r6_embeddings(root: Path, task_ids: Sequence[str]) -> dict[str, np.ndarray]:
    clusters, embeddings = load_clusters(R54_ROOT)
    out = {tid: embeddings[tid] for tid in task_ids if tid in embeddings}
    missing = [tid for tid in task_ids if tid not in out]
    if missing:
        # should not happen: R6 ids are a subset of R5.4 acquisition
        pass
    return out


def _load_jsonl_exp(path: Path) -> list[Experience]:
    rows: list[Experience] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(Experience.model_validate(json.loads(line)))
        except Exception:
            continue
    return rows


def build_traj_cache() -> dict[str, Experience]:
    cache: dict[str, Experience] = {}
    root = R54_ROOT / "per_seed"
    if not root.exists():
        return cache
    for path in sorted(root.glob("*/seed_*/trajectories.jsonl")):
        for exp in _load_jsonl_exp(path):
            if exp.task_id and str(exp.task_id) not in cache:
                cache[str(exp.task_id)] = exp
    return cache


def _seed_complete(out: Path, budget: int) -> bool:
    if not (out / "DONE").exists() or not (out / "final_metrics.json").exists():
        return False
    path = out / "checkpoint_metrics.jsonl"
    if not path.exists():
        return False
    budgets = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            budgets.add(int(json.loads(line)["budget"]))
    return 0 in budgets and budget in budgets


def _try_seed_lock(out: Path) -> TextIO | None:
    lock_path = out / ".run.lock"
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} ts={time.time()}\n")
    fh.flush()
    return fh


def _release_seed_lock(fh: TextIO | None) -> None:
    if fh is None:
        return
    try:
        fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        fh.close()
    except OSError:
        pass


def require_mask_cache(root: Path, split_name: str) -> dict[str, dict[str, Any]]:
    out = {}
    missing = []
    for mask in ALL_MASKS:
        rec = load_mask_summary(root, split_name, mask)
        if rec is None:
            missing.append(mask)
        else:
            out[mask] = rec
    if missing:
        raise FileNotFoundError(f"mask cache incomplete for {split_name}: {missing}")
    return out


def value_from_cache(cache: Mapping[str, Mapping[str, Any]], before: str, after: str) -> float:
    return float(cache[after]["macro"]) - float(cache[before]["macro"])


def run_one_seed(
    *,
    condition: str,
    seed: int,
    output_dir: Path,
    model: str,
    budget: int,
    gpu: str | None,
    resume: bool,
    retriever_url: str,
    batch_size: int = BATCH_SIZE,
    num_candidates: int = NUM_CANDIDATES,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    out = root / "per_seed" / condition / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    lock_f: TextIO | None = None
    while True:
        if resume and _seed_complete(out, budget):
            return {"status": "already_done", "out": str(out), "condition": condition, "seed": seed}
        lock_f = _try_seed_lock(out)
        if lock_f is not None:
            break
        print(f"[lock] {condition} seed={seed} held; waiting", flush=True)
        time.sleep(5)
    try:
        return _run_one_seed_unlocked(
            condition=condition,
            seed=seed,
            output_dir=output_dir,
            model=model,
            budget=budget,
            gpu=gpu,
            resume=resume,
            retriever_url=retriever_url,
            batch_size=batch_size,
            num_candidates=num_candidates,
            cache=cache,
        )
    finally:
        _release_seed_lock(lock_f)


def _run_one_seed_unlocked(
    *,
    condition: str,
    seed: int,
    output_dir: Path,
    model: str,
    budget: int,
    gpu: str | None,
    resume: bool,
    retriever_url: str,
    batch_size: int,
    num_candidates: int,
    cache: dict[str, Any] | None,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    spec = CONDITIONS[condition]
    apply_gpu(gpu)
    root = Path(output_dir)
    out = root / "per_seed" / condition / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if resume and _seed_complete(out, budget):
        return {"status": "already_done", "out": str(out), "condition": condition, "seed": seed}

    split = load_split_r6(root)
    predictions = load_predictions(R54_ROOT)
    clusters, _emb_all = load_clusters(R54_ROOT)
    router = load_router(R54_ROOT)
    pool = records_to_tasks(split.get("candidate_pool") or [], split="candidate_pool", predictions=predictions)
    shadow_tasks = records_to_tasks(split.get("shadow_probe") or [], split="shadow_probe", predictions=predictions)
    future = records_to_tasks(split.get("future_probe") or [], split="future_probe", predictions=predictions)
    future_ids = {t.task_id for t in future}
    shadow_ids = {t.task_id for t in shadow_tasks}
    assert not (set(t.task_id for t in pool) & future_ids)
    assert not (set(t.task_id for t in pool) & shadow_ids)
    by_id = {t.task_id: t for t in pool}
    world = world_from_tasks(list(pool) + list(shadow_tasks) + list(future))
    base_rows = merged_base_rows(root)
    missing_future = [t.task_id for t in future if t.task_id not in base_rows]
    if missing_future:
        raise RuntimeError(f"missing future base rows n={len(missing_future)}")
    embeddings = load_r6_embeddings(root, [t.task_id for t in pool])
    shadow_cache = require_mask_cache(root, "shadow_probe")
    future_cache = require_mask_cache(root, "future_probe")
    traj_cache = cache.get("traj") if cache and cache.get("traj") is not None else build_traj_cache()
    if cache is not None:
        cache["traj"] = traj_cache

    write_run_header(
        out,
        config={
            "script": "run_r6_ava_stage1.py",
            "condition": condition,
            "seed": seed,
            "model": model,
            "gpu": gpu,
            "budget": budget,
            "batch_size": batch_size,
            "num_candidates": num_candidates,
            "dispatch": "text_router",
            "activation": spec["activation"],
            "acquisition": spec["acquisition"],
            "retriever_url": retriever_url,
            "split_hash": sha256_file(root / "split_r6_ava.json") if (root / "split_r6_ava.json").exists() else "",
            "future_ids_in_selector": False,
            "assimilator": "count_only" if spec["activation"] == "count_only" else "srt",
        },
    )

    acquired = _load_jsonl_exp(out / "trajectories.jsonl") if resume else []
    dummy_state = empty_agent(out / "dummy", model_name=model)
    diagnoser = CapabilityDiagnoser()
    rasa: RASAAcquisition | None = None
    ava: ExactAVAAcquisition | None = None
    evidence = EvidenceState(predicted_cap=predictions)
    if acquired:
        evidence = evidence.apply_batch([str(e.task_id) for e in acquired if e.task_id], predictions)

    if spec["acquisition"] == "rasa" or condition in NEEDS_RASA_STATE:
        rasa = RASAAcquisition(
            seed=seed,
            predicted_cap=predictions,
            clusters=clusters,
            embeddings=embeddings,
            qonly=bool(spec["qonly"]),
            diversity=bool(spec["diversity"]),
        )
        policy: Any = rasa
    elif spec["acquisition"] == "exact_ava":
        ava = ExactAVAAcquisition(
            seed=seed,
            predicted_cap=predictions,
            embeddings=embeddings,
            n_batches=num_candidates,
            batch_size=batch_size,
        )
        policy = ava
    elif spec["acquisition"] == "random_cluster":
        policy = RandomClusterAcquisition(seed=seed, embeddings=embeddings, n_batches=num_candidates, batch_size=batch_size)
    elif spec["acquisition"] == "ace_skill_prioritized":
        policy = make_acquisition("ace_skill_prioritized", seed=seed)
    else:
        policy = make_acquisition(spec["acquisition"], seed=seed)

    pending: list[str] = []
    t_round = 0
    state_path = out / "sampler_state.json"
    rasa_path = out / "rasa_state.json"
    ev_path = out / "evidence_state.json"
    if isinstance(policy, AceSkillPrioritizedAcquisition):
        policy.bind([t.task_id for t in pool])
        if resume and state_path.exists():
            payload = load_json(state_path)
            policy.load_state(payload)
            pending = list(payload.get("pending_round") or [])
            t_round = int(payload.get("t", 0))
    if rasa is not None and resume and rasa_path.exists():
        rasa.load_snapshot(load_json(rasa_path))
    if resume and ev_path.exists():
        evidence = EvidenceState.from_snapshot(load_json(ev_path))
    if ava is not None and resume:
        ava.round_idx = t_round

    cache = cache if cache is not None else {}
    checkpoint_path = out / "checkpoint_metrics.jsonl"
    done_evals: set[int] = set()
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_evals.add(int(json.loads(line)["budget"]))

    unique_ids: list[str] = []
    seen_unique: set[str] = set()
    for exp in acquired:
        if exp.task_id and str(exp.task_id) not in seen_unique:
            seen_unique.add(str(exp.task_id))
            unique_ids.append(str(exp.task_id))

    need_llm = len(unique_ids) < budget
    agent = cache.get("agent")
    base_state = cache.get("base_state")
    backend = cache.get("backend")
    env = cache.get("env")
    if need_llm and agent is None:
        heartbeat(out, phase="load_model", acquired=len(unique_ids), budget=budget, condition=condition, seed=seed)
        agent, base_state, backend, env = build_public_agent(
            world, model_path=model, output_dir=out / "base_agent", retriever_url=retriever_url
        )
        cache.update({"agent": agent, "base_state": base_state, "backend": backend, "env": env})
    env = env or cache.get("env") or PublicSearchEnvironment(world, RetrieverClient(retriever_url))
    t0 = time.time()
    replay_count = 0
    tokens = int(sum(e.cost.tokens for e in acquired))
    tools = int(sum(e.cost.tool_calls for e in acquired))
    cost_path = out / "cost.jsonl"
    remaining = [t for t in pool if t.task_id not in seen_unique]
    recent = list(acquired)
    frozen_skills = {cap: fn() for cap, fn in PRIMITIVES.items()}
    for cap, skill in frozen_skills.items():
        skill.status = "active"
        skill.executable = True

    def save_side() -> None:
        write_json(ev_path, evidence.snapshot())
        if rasa is not None:
            write_json(rasa_path, rasa.snapshot())
        if isinstance(policy, AceSkillPrioritizedAcquisition):
            payload = policy.state_dict()
            payload["t"] = t_round
            payload["pending_round"] = pending
            write_json(state_path, payload)
        if ava is not None:
            write_json(out / "ava_round.json", {"round_idx": ava.round_idx})

    def lookup_eval(n_unique: int, mask: str, acquired_chunk: Sequence[Experience]) -> dict[str, Any]:
        fut = future_cache[mask]
        sh = shadow_cache[mask]
        dist = acquisition_distribution(acquired_chunk)
        row = {
            "budget": n_unique,
            "condition": condition,
            "seed": seed,
            "macro": fut["macro"],
            "per_dataset": fut.get("per_dataset") or {},
            "per_capability": fut.get("per_capability") or {},
            "n_active_skills": fut.get("n_active_skills") or len(mask.split("+")) if mask != "none" else 0,
            "n_skill_invocations": fut.get("n_skill_invocations") or 0,
            "activation": spec["activation"],
            "mask": mask,
            "shadow_macro": sh["macro"],
            "shadow_per_dataset": sh.get("per_dataset") or {},
            **dist,
        }
        return row

    def maybe_eval(n_unique: int, *, force: bool = False) -> dict[str, Any] | None:
        targets = [b for b in FULL_EVAL_AT if b <= n_unique]
        if force and n_unique not in targets:
            targets.append(n_unique)
        last_row = None
        id_order = unique_ids[:n_unique]
        chunk_by_id = {}
        for e in acquired:
            tid = str(e.task_id)
            if tid in id_order and tid not in chunk_by_id:
                chunk_by_id[tid] = e
        ordered = [chunk_by_id[i] for i in id_order if i in chunk_by_id]
        if spec["activation"] == "count_only":
            mask = evidence.mask_key()
            if n_unique == 0:
                mask = "none"
        else:
            assert rasa is not None
            active = tuple(c for c, rec in rasa.state.items() if rec.get("state") == STATE_ACTIVE)
            from expolicy.state.isolated_clone import mask_key_from_caps

            mask = mask_key_from_caps(active) if n_unique else "none"
        for n in sorted(set(targets)):
            if n in done_evals and not force:
                continue
            if n in done_evals:
                continue
            heartbeat(out, phase="eval", acquired=n, budget=budget, condition=condition, seed=seed, mask=mask)
            row = lookup_eval(n, mask if n else "none", ordered if n else [])
            if spec["activation"] != "count_only" or n != n_unique:
                # recompute mask at prefix n
                prefix_state = EvidenceState(predicted_cap=predictions).apply_batch(unique_ids[:n], predictions)
                use_mask = "none" if n == 0 else prefix_state.mask_key()
                if spec["activation"] == "srt" and rasa is not None:
                    # SRT mask is not a prefix CountOnly function; use live rasa state only at current n
                    use_mask = mask if n == n_unique else use_mask
                row = lookup_eval(n, use_mask, ordered[:n] if n else [])
                row["mask"] = use_mask
            append_jsonl(checkpoint_path, row)
            done_evals.add(n)
            write_json(out / f"eval_budget_{n}.json", row)
            last_row = row
        return last_row

    if 0 not in done_evals:
        maybe_eval(0)

    sel_path = out / "R6_CANDIDATE_SELECTIONS.jsonl"

    def acquire_one(task) -> Experience:
        nonlocal tokens, tools
        cached = traj_cache.get(str(task.task_id))
        if cached is not None:
            exp = cached.model_copy(deep=True)
            exp.run_id = "acquire_cached"
            return exp
        if agent is None:
            raise RuntimeError("LLM agent required for acquisition")
        agent.skill_store = None
        agent.use_executable_skill = False
        heartbeat(out, phase="acquire", acquired=len(unique_ids), budget=budget, condition=condition, seed=seed, task_id=task.task_id)
        exp = agent.run_task(task, state=base_state)
        exp.run_id = "acquire"
        return exp

    while len(unique_ids) < budget:
        if rasa is not None and spec["acquisition"] == "rasa" and rasa.all_terminal() and rasa.warm_done:
            break
        n = len(unique_ids)
        want = 1 if spec["acquisition"] == "rasa" else min(int(batch_size), budget - n)
        profile = diagnoser.diagnose(recent) if recent else diagnoser.diagnose([])
        scored_rows: list[dict[str, Any]] = []
        if pending:
            batch_ids = pending[:want]
            pending = pending[want:]
        elif ava is not None:
            ava.round_idx = t_round
            batches = ava.propose_candidates(remaining, round_idx=t_round)
            scored_rows = ava.score_candidates(
                evidence,
                batches,
                lambda b, a: value_from_cache(shadow_cache, b, a),
            )
            picked = ava.select(scored_rows)
            batch_ids = [tid for tid in picked.get("task_ids") or [] if tid not in seen_unique][:want]
            append_jsonl(
                sel_path,
                {
                    "round": t_round,
                    "seed": seed,
                    "picked": picked,
                    "candidates": scored_rows,
                    "committed_ids": batch_ids,
                },
            )
        else:
            cand_pool = remaining if spec["acquisition"] != "ace_skill_prioritized" else pool
            if spec["acquisition"] == "ace_skill_prioritized":
                props = policy.propose(dummy_state, profile, cand_pool, want, recent=recent, t=t_round)
            else:
                props = policy.propose(dummy_state, profile, cand_pool, want, recent=recent)
            batch_ids = [tid for tid in list(props[0].task_ids) if tid not in seen_unique][:want]
            if getattr(props[0], "metadata", {}) and props[0].metadata.get("early_stop") and not batch_ids:
                break
        if not batch_ids:
            if remaining:
                batch_ids = [remaining[0].task_id]
            else:
                break
        pending = list(batch_ids)
        save_side()
        for tid in list(batch_ids):
            if len(unique_ids) >= budget:
                break
            if tid in seen_unique:
                pending = [x for x in pending if x != tid]
                remaining = [t for t in remaining if t.task_id != tid]
                continue
            if tid in future_ids or tid in shadow_ids:
                raise RuntimeError(f"leakage: attempted to acquire {tid}")
            task = by_id.get(tid) or next((t for t in remaining if t.task_id == tid), None)
            if task is None:
                if remaining:
                    task = remaining[0]
                else:
                    break
            exp = acquire_one(task)
            pred_cap = predictions.get(task.task_id) or task.capability
            exp.capability = pred_cap
            row = score_public(exp, task)
            base_success = bool(row.get("em"))
            exp.success = base_success
            acquired.append(exp)
            recent.append(exp)
            tokens += int(exp.cost.tokens or 0)
            tools += int(exp.cost.tool_calls or 0)
            unique_ids.append(str(task.task_id))
            seen_unique.add(str(task.task_id))
            remaining = [t for t in remaining if t.task_id != task.task_id]
            evidence = evidence.apply_batch([task.task_id], predictions)
            if rasa is not None:
                rasa.observe_base(task.task_id, base_success, unique_budget=len(unique_ids))
            if isinstance(policy, AceSkillPrioritizedAcquisition):
                policy.observe(task.task_id, 1.0 if base_success else 0.0, t_round)
            repaired = None
            if (not base_success) and condition in NEEDS_REPLAY and rasa is not None:
                cap = str(pred_cap)
                rec = rasa.state.get(cap) or {}
                if rec.get("state") not in {STATE_ACTIVE, STATE_PARKED}:
                    skill = frozen_skills.get(cap)
                    if skill is not None:
                        heartbeat(out, phase="skill_replay", acquired=len(unique_ids), task_id=task.task_id)
                        replay = run_public_skill_on_task(skill, env, task, model=backend)
                        replay.run_id = "skill_replay"
                        replay_row = score_public(replay, task)
                        repaired = bool(replay_row.get("em"))
                        replay_count += 1
                        tokens += int(replay.cost.tokens or 0)
                        tools += int(replay.cost.tool_calls or 0)
                        rasa.observe_repair(task.task_id, repaired)
                        append_jsonl(out / "skill_replays.jsonl", json.loads(replay.model_dump_json()))
            if rasa is not None:
                rasa.apply_readiness(unique_budget=len(unique_ids), use_srt=spec["activation"] == "srt")
            append_jsonl(out / "trajectories.jsonl", json.loads(exp.model_dump_json()))
            append_jsonl(
                out / "selected_experiences.jsonl",
                {
                    "step": len(unique_ids),
                    "task_id": task.task_id,
                    "predicted_capability": pred_cap,
                    "true_capability": (task.metadata or {}).get("true_capability"),
                    "dataset": (task.metadata or {}).get("dataset"),
                    "base_success": base_success,
                    "skill_repaired": repaired,
                    "tokens": exp.cost.tokens,
                    "tool_calls": exp.cost.tool_calls,
                    "mask_after": evidence.mask_key(),
                },
            )
            append_jsonl(
                cost_path,
                {
                    "step": len(unique_ids),
                    "tokens": tokens,
                    "tool_calls": tools,
                    "diagnostic_replay_count": replay_count,
                    "wall_seconds": time.time() - t0,
                },
            )
            pending = [x for x in pending if x != tid]
            save_side()
            if len(unique_ids) in FULL_EVAL_AT:
                maybe_eval(len(unique_ids))
        t_round += 1
        if ava is not None:
            ava.round_idx = t_round
        pending = []
        save_side()

    for n in FULL_EVAL_AT:
        if n <= len(unique_ids) and n not in done_evals:
            maybe_eval(n)

    rows = [json.loads(l) for l in checkpoint_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    zero = next((r for r in rows if int(r["budget"]) == 0), None)
    last = next((r for r in rows if int(r["budget"]) == budget), rows[-1] if rows else None)
    if last is None or zero is None:
        raise RuntimeError(f"incomplete evals for {condition} seed={seed}")
    dist = acquisition_distribution(acquired)
    summary = {
        "condition": condition,
        "seed": seed,
        "n_acquired_unique": len(unique_ids),
        "n_acquired_rollouts": len(acquired),
        "base_future_macro": float(zero["macro"]),
        "future_macro": last["macro"],
        "delta_future": last["macro"] - float(zero["macro"]),
        "per_dataset": last.get("per_dataset") or {},
        "per_capability": last.get("per_capability") or {},
        "n_active_skills": last.get("n_active_skills"),
        "final_mask": last.get("mask"),
        "shadow_macro_final": last.get("shadow_macro"),
        "tokens": tokens,
        "tool_calls": tools,
        "diagnostic_replay_count": replay_count,
        "wall_seconds": time.time() - t0,
        "dispatch": "text_router",
        "activation": spec["activation"],
        "acquisition": spec["acquisition"],
        **dist,
        "protocol": "r6.1 exact AVA; frozen CountOnly+text-router; R5.4 future untouched",
    }
    write_json(out / "final_metrics.json", summary)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    heartbeat(out, phase="done", acquired=len(unique_ids), budget=budget, condition=condition, seed=seed)
    return summary
