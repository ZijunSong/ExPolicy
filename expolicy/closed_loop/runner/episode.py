"""R7 closed-loop episode: stateful acquire → assimilate → shadow reward → policy update."""

from __future__ import annotations

import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from expolicy.closed_loop.agent_state import ClosedLoopRuntime
from expolicy.closed_loop.constants import (
    ACTIVATE_EXISTING_SKILL,
    CAPS,
    CONDITION_SPECS,
    R7_ROOT,
)
from expolicy.closed_loop.experience import summarize_batch
from expolicy.closed_loop.io_utils import append_jsonl, sha256_json, write_json
from expolicy.closed_loop.policies import (
    LinearThompsonAcquisition,
    LinearThompsonAssimilation,
    RandomAcquisition,
    RandomAssimilation,
    always_discard,
    always_episodic,
    count_only,
)
from expolicy.closed_loop.reward import compute_realized_utility
from expolicy.closed_loop.runner.checkpoint import (
    ResumeHashError,
    latest_completed_round,
    load_hashes,
    round_dir,
    run_dir,
    save_round_artifacts,
    write_hashes,
)
from expolicy.closed_loop.runner.config import R7Config
from expolicy.closed_loop.runner.evaluator import evaluate_tasks
from expolicy.closed_loop.split_data import SPLIT_PATH, load_split, split_hash, split_tasks, tasks_by_cap
from expolicy.closed_loop.state import ExPolicyState
from expolicy.core.experience import Experience
from expolicy.environments.public_search.environment import PublicSearchEnvironment, PublicWorld
from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.environments.search.world import Task
from expolicy.logging.event_logger import EventLogger
from expolicy.models.factory import build_backend


def heartbeat(out: Path, **payload: Any) -> None:
    payload = {"updated_at": time.time(), "pid": os.getpid(), **payload}
    write_json(out / "STATUS_LIVE.json", payload)


def make_acquisition(cfg: R7Config):
    if cfg.acquisition == "linear_thompson":
        return LinearThompsonAcquisition(prior_precision=cfg.prior_precision, reward_noise=cfg.reward_noise)
    if cfg.acquisition == "random":
        return RandomAcquisition()
    raise ValueError(cfg.acquisition)


def make_assimilation(cfg: R7Config):
    if cfg.assimilation == "linear_thompson":
        return LinearThompsonAssimilation(prior_precision=cfg.prior_precision, reward_noise=cfg.reward_noise)
    if cfg.assimilation == "random":
        return RandomAssimilation()
    if cfg.assimilation == "fixed_discard":
        return always_discard()
    if cfg.assimilation == "fixed_episodic":
        return always_episodic()
    if cfg.assimilation == "count_only":
        return count_only()
    raise ValueError(cfg.assimilation)


def apply_condition_spec(cfg: R7Config) -> R7Config:
    spec = CONDITION_SPECS.get(cfg.condition)
    if not spec:
        raise ValueError(f"unknown condition {cfg.condition}")
    cfg.acquisition = spec["acquisition"]
    cfg.assimilation = spec["assimilation"]
    cfg.feedback = spec["feedback"]
    cfg.all_skills_upfront = bool(spec["all_skills_upfront"])
    return cfg


def _ema(old: float, value: float, alpha: float = 0.3) -> float:
    return (1.0 - alpha) * float(old) + alpha * float(value)


def _ingest_acquisition(state: ExPolicyState, experiences: Sequence[Experience], cap: str) -> None:
    for exp in experiences:
        c = exp.capability or cap
        state.capability_seen_count[c] = int(state.capability_seen_count.get(c) or 0) + 1
        if exp.success:
            state.capability_success_count[c] = int(state.capability_success_count.get(c) or 0) + 1
        else:
            state.capability_failure_count[c] = int(state.capability_failure_count.get(c) or 0) + 1
        state.total_tokens += int(exp.cost.tokens or 0)
        state.total_tool_calls += int(exp.cost.tool_calls or 0)
        if exp.task_id and exp.task_id not in state.used_task_ids:
            state.used_task_ids.append(str(exp.task_id))
    state.budget_used = len(state.used_task_ids)
    state.last_batch_capability = cap


def _calibration_stats(experiences: Sequence[Experience]) -> dict[str, Any]:
    from collections import defaultdict

    by_cap: dict[str, list[Experience]] = defaultdict(list)
    for e in experiences:
        by_cap[str(e.capability or "")].append(e)
    out: dict[str, Any] = {}
    for cap, rows in by_cap.items():
        n = len(rows)
        succ = sum(1 for e in rows if e.success)
        searches = []
        repeats = []
        premature = []
        for e in rows:
            qs = [str((c.get("arguments") or {}).get("query") or "") for c in e.tool_calls if c.get("tool") == "search"]
            searches.append(len(qs))
            qn = [q.lower() for q in qs if q]
            repeats.append(1.0 if len(qn) >= 2 and len(set(qn)) < len(qn) else 0.0)
            saw_open = False
            prem = 0.0
            for act in e.actions:
                if act.get("tool") == "open":
                    saw_open = True
                if act.get("tool") == "answer":
                    prem = 0.0 if saw_open else 1.0
                    break
            premature.append(prem)
        out[cap] = {
            "n": n,
            "em": succ / max(1, n),
            "failure_rate": 1.0 - succ / max(1, n),
            "mean_searches": sum(searches) / max(1, n),
            "repeat_query_rate": sum(repeats) / max(1, n),
            "premature_answer_rate": sum(premature) / max(1, n),
        }
    return out


def _sample(pool: list[Task], cap: str, used: set[str], n: int, rng: np.random.Generator) -> list[Task]:
    cand = [t for t in pool if t.capability == cap and t.task_id not in used]
    if len(cand) < n:
        raise RuntimeError(f"fail-closed: not enough unseen {cap} tasks ({len(cand)} < {n})")
    idx = rng.permutation(len(cand))
    return [cand[int(i)] for i in idx[:n]]


def _decision_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dict__"):
        d = dict(obj.__dict__)
        return d
    return dict(obj)


def build_runtime(
    cfg: R7Config,
    tasks: Sequence[Task],
    out: Path,
    state: ExPolicyState,
    *,
    model: Any | None = None,
    client: Any | None = None,
) -> tuple[ClosedLoopRuntime, Any]:
    world = PublicWorld()
    for t in tasks:
        world.add_task(t)
    client = client or RetrieverClient(cfg.retriever_url)
    env = PublicSearchEnvironment(world, client, top_k=5)
    if model is None:
        model = build_backend({"backend": "transformers", "path": cfg.model, "dtype": "bfloat16"})
    logger = EventLogger(out / "events.jsonl")
    runtime = ClosedLoopRuntime(env=env, model=model, root=out / "agent", state=state, logger=logger, max_steps=cfg.max_steps)
    return runtime, model


def run_r7_episode(
    cfg: R7Config,
    *,
    output_root: Path | None = None,
    resume: bool = True,
    model: Any | None = None,
    split: dict[str, Any] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    cfg = apply_condition_spec(cfg)
    root = Path(output_root or R7_ROOT)
    out = run_dir(root, cfg.condition, cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    split = split or load_split()
    if SPLIT_PATH.exists():
        shash = split_hash()
    else:
        from expolicy.closed_loop.io_utils import sha256_json

        shash = sha256_json({k: split.get(k) for k in ("calibration", "acquisition_pool", "shadow_reward_probe", "retention_probe", "future_test", "split_seed")})
    rng = np.random.default_rng(int(cfg.seed) + 1009)

    cal_tasks = split_tasks(split, "calibration")
    acq_tasks = split_tasks(split, "acquisition_pool")
    shadow_tasks = split_tasks(split, "shadow_reward_probe")
    retention_tasks = split_tasks(split, "retention_probe")
    future_tasks = split_tasks(split, "future_test")
    if cfg.smoke:
        cal_tasks = _cap_trim(cal_tasks, 2)
        acq_tasks = _cap_trim(acq_tasks, 8)
        shadow_tasks = _cap_trim(shadow_tasks, 2)
        retention_tasks = retention_tasks[:4]
        future_tasks = future_tasks[:6]

    all_tasks = cal_tasks + acq_tasks + shadow_tasks + retention_tasks + future_tasks
    pool_sizes = {c: len(v) for c, v in tasks_by_cap(acq_tasks).items()}

    write_json(out / "config.json", cfg.to_dict())
    heartbeat(out, phase="boot", condition=cfg.condition, seed=cfg.seed)

    state = ExPolicyState(budget_total=cfg.budget, capability_pool_size=pool_sizes)
    acq_policy = make_acquisition(cfg)
    assim_policy = make_assimilation(cfg)

    if resume and (out / "DONE").exists():
        heartbeat(out, phase="already_done")
        return read_done(out)

    last = latest_completed_round(out) if resume else None
    if resume and last is not None:
        prev_hashes = load_hashes(out)
        if prev_hashes and prev_hashes.get("config_hash") != cfg.content_hash():
            raise ResumeHashError("config hash mismatch; refuse resume")
        if prev_hashes and prev_hashes.get("split_hash") != shash:
            raise ResumeHashError("split hash mismatch; refuse resume")
        rd = round_dir(root, cfg.condition, cfg.seed, last)
        state = ExPolicyState.from_dict(_read(rd / "state_after.json"))
        acq_policy.load_snapshot(_read(rd / "policy_after.json")["acquisition"])
        assim_policy.load_snapshot(_read(rd / "policy_after.json")["assimilation"])
        start_round = last + 1
    else:
        start_round = 0
    incomplete = round_dir(root, cfg.condition, cfg.seed, start_round)
    if incomplete.exists() and not (incomplete / "state_after.json").exists():
        shutil.rmtree(incomplete)

    runtime, model = build_runtime(cfg, all_tasks, out, state, model=model, client=client)
    t_run = time.time()
    reward_history: list[float] = []

    if cfg.all_skills_upfront:
        for cap in CAPS:
            state.active_skills[cap] = True
        heartbeat(out, phase="all_skills_eval")
        fut = evaluate_tasks(runtime, future_tasks, out=out, phase="future_test_final", heartbeat=lambda **k: heartbeat(out, **k))
        for b in cfg.full_eval_at:
            write_json(out / f"eval_budget_{b}.json", fut)
        write_json(out / "final_metrics.json", {"macro": fut["macro"], "n": fut["n"], "condition": cfg.condition})
        (out / "DONE").write_text("ok\n", encoding="utf-8")
        return fut

    if start_round == 0 and not (out / "init" / "shadow_summary.json").exists():
        heartbeat(out, phase="calibration", n=len(cal_tasks))
        cal_exps: list[Experience] = []
        for i, task in enumerate(cal_tasks):
            heartbeat(out, phase="calibration", i=i, n=len(cal_tasks), task_id=task.task_id)
            cal_exps.append(runtime.run_task(task))
        cal_stats = _calibration_stats(cal_exps)
        state.extra["calibration"] = cal_stats
        for cap, blob in cal_stats.items():
            if cap in CAPS:
                state.shadow_em_by_capability[cap] = float(blob.get("em") or 0.0)
        init_dir = out / "init"
        init_dir.mkdir(parents=True, exist_ok=True)
        heartbeat(out, phase="init_shadow")
        shadow0 = evaluate_tasks(runtime, shadow_tasks, out=init_dir, phase="shadow", heartbeat=lambda **k: heartbeat(out, **k))
        ret0 = evaluate_tasks(runtime, retention_tasks, out=init_dir, phase="retention", heartbeat=lambda **k: heartbeat(out, **k))
        state.shadow_em_by_capability = dict(shadow0["em_by_capability"])
        state.shadow_macro = float(shadow0["macro"])
        state.retention_score = float(ret0["macro"])
        if 0 in cfg.full_eval_at:
            heartbeat(out, phase="future_test", budget=0)
            fut0 = evaluate_tasks(runtime, future_tasks, out=init_dir, phase="future_test", heartbeat=lambda **k: heartbeat(out, **k))
            write_json(out / "eval_budget_0.json", fut0)
            assert "future_test" not in (acq_policy.snapshot().get("heads") or {}), "future leaked into policy"
        write_json(init_dir / "state.json", state.to_dict())

    for t in range(start_round, cfg.rounds):
        try:
            _run_one_round(
                cfg=cfg,
                root=root,
                out=out,
                t=t,
                state=state,
                runtime=runtime,
                acq_policy=acq_policy,
                assim_policy=assim_policy,
                acq_tasks=acq_tasks,
                shadow_tasks=shadow_tasks,
                retention_tasks=retention_tasks,
                future_tasks=future_tasks,
                rng=rng,
                shash=shash,
                reward_history=reward_history,
            )
        except Exception as exc:
            heartbeat(out, phase="fail_closed", round=t, error=str(exc), tb=traceback.format_exc()[-2000:])
            raise

    heartbeat(out, phase="final_future")
    fut = evaluate_tasks(runtime, future_tasks, out=out, phase="future_test_final", heartbeat=lambda **k: heartbeat(out, **k))
    write_json(out / f"eval_budget_{cfg.budget}.json", fut)
    write_json(
        out / "final_metrics.json",
        {
            "macro": fut["macro"],
            "em_by_capability": fut["em_by_capability"],
            "em_by_dataset": fut["em_by_dataset"],
            "condition": cfg.condition,
            "seed": cfg.seed,
            "budget": cfg.budget,
            "wall_seconds": time.time() - t_run,
            "active_skills": dict(state.active_skills),
            "epi_size": dict(state.episodic_memory_size),
            "sem_size": dict(state.semantic_memory_size),
            "tokens": state.total_tokens,
            "tools": state.total_tool_calls,
        },
    )
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    heartbeat(out, phase="done", macro=fut["macro"])
    return fut


def _run_one_round(
    *,
    cfg: R7Config,
    root: Path,
    out: Path,
    t: int,
    state: ExPolicyState,
    runtime: ClosedLoopRuntime,
    acq_policy,
    assim_policy,
    acq_tasks: list[Task],
    shadow_tasks: list[Task],
    retention_tasks: list[Task],
    future_tasks: list[Task],
    rng: np.random.Generator,
    shash: str,
    reward_history: list[float],
) -> None:
    rd = round_dir(root, cfg.condition, cfg.seed, t)
    rd.mkdir(parents=True, exist_ok=True)
    state.round_id = t
    runtime.state = state
    used = set(state.used_task_ids)
    available = [c for c in CAPS if any(x.capability == c and x.task_id not in used for x in acq_tasks)]
    if not available:
        raise RuntimeError("fail-closed: acquisition pool exhausted")

    state_before = state.copy()
    policy_before = {"acquisition": acq_policy.snapshot(), "assimilation": assim_policy.snapshot()}
    save_round_artifacts(rd, **{"state_before.json": state_before, "policy_before.json": policy_before})

    acq_dec = acq_policy.select(state, available, rng, n_rounds=cfg.rounds)
    cap = acq_dec.capability
    batch_tasks = _sample(acq_tasks, cap, used, cfg.batch_size, rng)
    save_round_artifacts(rd, **{"acquisition_decision.json": _decision_dict(acq_dec)})

    experiences: list[Experience] = []
    for i, task in enumerate(batch_tasks):
        heartbeat(out, phase="acquisition", round=t, i=i, n=len(batch_tasks), cap=cap, task_id=task.task_id)
        exp = runtime.run_task(task)
        experiences.append(exp)
        append_jsonl(rd / "experience_batch.jsonl", exp.model_dump(mode="json"))

    summary = summarize_batch(experiences, capability=cap)
    write_json(rd / "experience_summary.json", summary.to_dict())
    state.last_batch_success_rate = summary.success_rate
    state.last_batch_mean_searches = summary.mean_searches
    state.last_batch_repeat_rate = summary.repeat_query_rate
    state.last_batch_premature_answer_rate = summary.premature_answer_rate

    valid = runtime.valid_assim_actions(cap)
    assim_dec = assim_policy.select(state, summary, valid, rng)
    if assim_dec.action not in valid:
        raise RuntimeError(f"fail-closed: invalid action {assim_dec.action}")
    save_round_artifacts(rd, **{"assimilation_decision.json": _decision_dict(assim_dec)})

    apply_info = runtime.apply_assimilation(assim_dec.action, summary, experiences)
    write_json(rd / "apply.json", apply_info)
    _ingest_acquisition(state, experiences, cap)

    heartbeat(out, phase="shadow", round=t)
    shadow = evaluate_tasks(runtime, shadow_tasks, out=rd, phase="shadow", heartbeat=lambda **k: heartbeat(out, round=t, **k))
    heartbeat(out, phase="retention", round=t)
    retention = evaluate_tasks(runtime, retention_tasks, out=rd, phase="retention", heartbeat=lambda **k: heartbeat(out, round=t, **k))

    reward = compute_realized_utility(
        shadow_macro_before=state_before.shadow_macro,
        shadow_macro_after=float(shadow["macro"]),
        retention_before=state_before.retention_score,
        retention_after=float(retention["macro"]),
        token_cost=summary.token_cost,
        tool_cost=summary.tool_cost,
        retention_weight=cfg.retention_weight,
        token_cost_per_million=cfg.token_cost_per_million,
        tool_cost_per_100=cfg.tool_cost_per_100,
    )
    if reward.get("uses_future_test"):
        raise RuntimeError("fail-closed: reward used future_test")
    write_json(rd / "reward.json", reward)

    state.shadow_em_by_capability = dict(shadow["em_by_capability"])
    state.shadow_macro = float(shadow["macro"])
    state.retention_score = float(retention["macro"])
    state.last_realized_utility = float(reward["reward"])
    state.utility_ema_by_capability[cap] = _ema(float(state.utility_ema_by_capability.get(cap) or 0.0), float(reward["reward"]))
    state.utility_ema_by_assim_action[assim_dec.action] = _ema(
        float(state.utility_ema_by_assim_action.get(assim_dec.action) or 0.0), float(reward["reward"])
    )
    state.total_wall_seconds += float(shadow.get("wall_seconds") or 0.0) + float(retention.get("wall_seconds") or 0.0)

    true_r = float(reward["reward"])
    used_r = true_r
    if cfg.feedback == "shuffle":
        if reward_history:
            used_r = float(rng.choice(np.array(reward_history, dtype=np.float64)))
            if abs(used_r - true_r) < 1e-12:
                used_r = -true_r
        else:
            used_r = -true_r
        write_json(rd / "reward_shuffle.json", {"true": true_r, "shuffled": used_r})
    reward_history.append(true_r)

    acq_update = {"parameter_delta_norm": 0.0, "skipped": True, "policy": "acquisition"}
    assim_update = {"parameter_delta_norm": 0.0, "skipped": True, "policy": "assimilation"}
    if cfg.feedback != "none":
        acq_update = acq_policy.update(state_before, cap, used_r, state, n_rounds=cfg.rounds)
        assim_update = assim_policy.update(state_before, summary, assim_dec.action, used_r, state)
        if cfg.feedback == "true" and float(acq_update.get("parameter_delta_norm") or 0) <= 0 and cfg.acquisition == "linear_thompson":
            raise RuntimeError("fail-closed: acquisition posterior did not change")
        if cfg.feedback == "true" and float(assim_update.get("parameter_delta_norm") or 0) <= 0 and cfg.assimilation == "linear_thompson":
            raise RuntimeError("fail-closed: assimilation posterior did not change")

    policy_after = {"acquisition": acq_policy.snapshot(), "assimilation": assim_policy.snapshot()}
    write_json(
        rd / "policy_update.json",
        {
            "acquisition": acq_update,
            "assimilation": assim_update,
            "reward_used": used_r,
            "true_reward": true_r,
            "feedback": cfg.feedback,
        },
    )
    save_round_artifacts(rd, **{"state_after.json": state, "policy_after.json": policy_after})
    write_hashes(
        out,
        {
            "config_hash": cfg.content_hash(),
            "split_hash": shash,
            "state_hash": state.content_hash(),
            "policy_checkpoint_hash": sha256_json(policy_after),
        },
    )

    if state.budget_used in cfg.full_eval_at and state.budget_used != 0:
        heartbeat(out, phase="future_test", budget=state.budget_used)
        fut = evaluate_tasks(
            runtime,
            future_tasks,
            out=rd,
            phase=f"future_test_b{state.budget_used}",
            heartbeat=lambda **k: heartbeat(out, **k),
        )
        write_json(out / f"eval_budget_{state.budget_used}.json", fut)
        # future_test must not enter policy snapshot
        blob = sha256_json(policy_after)
        if "future_test" in blob:
            raise RuntimeError("fail-closed: future_test leaked into policy checkpoint")

    heartbeat(out, phase="round_done", round=t, cap=cap, action=assim_dec.action, reward=true_r, budget=state.budget_used)


def _cap_trim(tasks: list[Task], per_cap: int) -> list[Task]:
    out: list[Task] = []
    counts = {c: 0 for c in CAPS}
    for t in tasks:
        c = t.capability
        if c in counts and counts[c] < per_cap:
            out.append(t)
            counts[c] += 1
    return out


def _read(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def read_done(out: Path) -> dict[str, Any]:
    p = out / "final_metrics.json"
    if p.exists():
        return _read(p)
    return {"status": "done"}
