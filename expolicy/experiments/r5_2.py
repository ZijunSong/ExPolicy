"""R5.2 helpers: unique-task CountOnly gating and Ace-Skill collision protocol."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from expolicy.acquisition.ace_skill import AceSkillPrioritizedAcquisition
from expolicy.acquisition.base import make_acquisition
from expolicy.assimilation.carriers.skill_store import SkillStore
from expolicy.assimilation.skill_assimilator import maps_to_skill_capability
from expolicy.core.experience import Experience
from expolicy.core.state import empty_agent
from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
from expolicy.experiments.common import ROOT, build_static_agent, load_world, mean, write_json
from expolicy.experiments.r5_1 import (
    EVAL_AT,
    R51_ROOT,
    R5_ROOT,
    REFIT_AT,
    TARGET_CAP_LIST,
    TARGET_CAPS,
    count_only_skill,
    ensure_seed_splits,
    eval_with_dispatch,
    first_activation,
    future_by_cap,
    load_acquired,
    load_json,
    load_split,
    tasks_by_ids,
    train_router_for_split,
)
from expolicy.experiments.r_common import apply_gpu, load_e0_experiences, write_run_header
from expolicy.logging.event_logger import append_jsonl

R52_ROOT = ROOT / "outputs" / "r5_2_baseline_collision"
PRIMARY_METHODS = ("frontier", "ace_skill_prioritized", "passive")
ROUND_SIZE = 8
FULL_EVAL_AT = (0, 16, 32, 48)
MIN_SUPPORT = 8
DEFAULT_MODEL = "/data/ppnm/models/Qwen2.5-3B-Instruct"


def heartbeat(out: Path, **payload: Any) -> None:
    payload = {"updated_at": time.time(), "pid": os.getpid(), **payload}
    write_json(out / "STATUS_LIVE.json", payload)


def unique_task_ids(acquired: Sequence[Experience], cap: str) -> set[str]:
    ids: set[str] = set()
    for exp in acquired:
        if maps_to_skill_capability(exp.capability) == cap and exp.task_id:
            ids.add(str(exp.task_id))
    return ids


def unique_counts(acquired: Sequence[Experience]) -> dict[str, int]:
    return {cap: len(unique_task_ids(acquired, cap)) for cap in TARGET_CAP_LIST}


def acquisition_distribution(acquired: Sequence[Experience]) -> dict[str, Any]:
    ids = [str(e.task_id) for e in acquired if e.task_id]
    counts = Counter(ids)
    n = len(ids)
    unique = len(counts)
    dup = n - unique
    per_cap_rollouts = Counter()
    per_cap_unique: dict[str, set[str]] = {cap: set() for cap in TARGET_CAP_LIST}
    other_unique: set[str] = set()
    for exp in acquired:
        cap = maps_to_skill_capability(exp.capability)
        tid = str(exp.task_id) if exp.task_id else ""
        if cap in per_cap_unique:
            per_cap_rollouts[cap] += 1
            if tid:
                per_cap_unique[cap].add(tid)
        elif tid:
            other_unique.add(tid)
            per_cap_rollouts[exp.capability or "other"] += 1
    return {
        "total_rollouts": n,
        "unique_tasks": unique,
        "duplicate_rollouts": dup,
        "unique_tasks_per_capability": {cap: len(per_cap_unique[cap]) for cap in TARGET_CAP_LIST},
        "rollouts_per_capability": dict(per_cap_rollouts),
        "n_revisit_tasks": int(sum(1 for c in counts.values() if c > 1)),
        "max_visits": int(max(counts.values()) if counts else 0),
    }


def activation_times_from_acquired(acquired: Sequence[Experience], *, k: int = MIN_SUPPORT) -> dict[str, int]:
    first: dict[str, int] = {}
    seen: dict[str, set[str]] = {cap: set() for cap in TARGET_CAP_LIST}
    for i, exp in enumerate(acquired, 1):
        cap = maps_to_skill_capability(exp.capability)
        if cap not in seen or not exp.task_id:
            continue
        seen[cap].add(str(exp.task_id))
        if cap not in first and len(seen[cap]) >= k:
            first[cap] = i
    return first


def target_skill_times(first: dict[str, int], *, budget: int = 48) -> dict[str, Any]:
    ordered = sorted(int(v) for v in first.values())
    def fmt(val: int | None) -> int | str:
        if val is None:
            return f">{budget}"
        return val

    return {
        "budget_to_first_target_skill": fmt(ordered[0] if ordered else None),
        "budget_to_second_target_skill": fmt(ordered[1] if len(ordered) > 1 else None),
        "n_activated_target_skills": len(ordered),
        "first_by_cap": dict(first),
    }


def count_only_store(
    acquired: Sequence[Experience],
    *,
    min_support: int,
    probe_ids: Sequence[str],
    gold_docs: Sequence[str],
    skill_dir: Path,
    budget: int | None = None,
) -> tuple[SkillStore, list[dict[str, Any]], dict[str, int]]:
    skill_dir.mkdir(parents=True, exist_ok=True)
    store = SkillStore(skill_dir / "store")
    subset = list(acquired if budget is None else acquired[:budget])
    events: list[dict[str, Any]] = []
    first = activation_times_from_acquired(subset, k=min_support)
    versions: dict[str, int] = {}
    for cap in TARGET_CAP_LIST:
        n_unique = len(unique_task_ids(subset, cap))
        if n_unique < min_support:
            continue
        versions[cap] = versions.get(cap, 0) + 1
        skill = count_only_skill(
            cap,
            n_support=n_unique,
            min_support=min_support,
            version=versions[cap],
            probe_ids=probe_ids,
            gold_docs=gold_docs,
        )
        skill.activation_reason = (
            f"count_only unique_task_ids={n_unique}>={min_support}; capability={cap}; "
            "duplicates do not count toward K"
        )
        store.install(skill)
        path = skill_dir / f"count_only_v{versions[cap]:03d}_{cap}.json"
        write_json(path, skill.model_dump(mode="json"))
        events.append(
            {
                "budget": len(subset),
                "capability": cap,
                "n_support": n_unique,
                "n_support_unique_tasks": n_unique,
                "version": skill.version,
                "skill_id": skill.skill_id,
                "min_support": min_support,
                "activation_reason": skill.activation_reason,
                "path": str(path),
            }
        )
    return store, events, first


def resolve_reused_run(method: str, seed: int, *, r5: Path, r51: Path) -> Path | None:
    if method not in {"frontier", "passive", "uncertainty", "failure_replay", "reward_seeking"}:
        return None
    if seed <= 2:
        cand = r5 / method / f"seed_{seed}"
    else:
        cand = r51 / "seed_extension" / method / f"seed_{seed}"
    if (cand / "DONE").exists() and (cand / "trajectories.jsonl").exists():
        return cand
    return None


def resolve_base_dir(seed: int, *, r52: Path, r5: Path, r51: Path) -> Path:
    for cand in (
        r52 / "base" / f"seed_{seed}",
        r51 / "seed_extension" / "base" / f"seed_{seed}",
        r5 / "base" / f"seed_{seed}",
    ):
        if (cand / "final_metrics.json").exists() and (cand / "trajectories.jsonl").exists():
            return cand
    raise FileNotFoundError(f"base eval missing for seed {seed}")


def load_base_bundle(seed: int, *, r52: Path, r5: Path, r51: Path) -> tuple[dict, dict[str, dict], Path]:
    base_dir = resolve_base_dir(seed, r52=r52, r5=r5, r51=r51)
    metrics = load_json(base_dir / "final_metrics.json")
    rows: dict[str, dict] = {}
    for line in (base_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        tid = payload.get("task_id")
        if tid:
            rows[str(tid)] = payload
    if not rows:
        raise FileNotFoundError(f"empty base trajectories in {base_dir}")
    return metrics, rows, base_dir


def _load_jsonl_exp(path: Path) -> list[Experience]:
    rows: list[Experience] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("run_id") and payload.get("run_id") != "acquire":
            continue
        try:
            rows.append(Experience.model_validate(payload))
        except Exception:
            continue
    return rows


def _task_map(pool: Sequence) -> dict[str, Any]:
    return {t.task_id: t for t in pool}


def write_audit(out: Path) -> None:
    text = """# ACE_SKILL_IMPLEMENTATION_AUDIT

R5.2 ports **only** the Ace-Skill Prioritized Sampler. The clustered organizer,
experience merger, and skill document writer are not used. Backend is frozen
CountOnly unique-task gating (K=8 distinct task IDs per capability).

## Sources

- Paper: arXiv:2605.08887 (Ace-Skill: Bootstrapping Multimodal Agents with Prioritized and Clustered Evolution)
- Official code: https://github.com/AMAP-ML/Ace-Skill
  - `eval/ace_skill/weighted_sampler.py` (`SqrtBiasSampler`)
  - `configs/prioritize.yaml`

## Hyperparameters (paper + official yaml; not code-class defaults)

| name | paper / yaml | `SqrtBiasSampler.__init__` default | R5.2 |
|---|---|---|---|
| rho | 0.95 | 0.9 | **0.95** |
| gamma | 0.4 | 0.2 | **0.4** |
| epsilon | 0.1 | 0.1 | **0.1** |

Class defaults in official code are overridden by `configs/prioritize.yaml`.
R5.2 follows the yaml / paper values, not the class defaults.

## Initialization: paper text vs official code

Paper §2.2 text says `(α_i, β_i)` initialized to `(0.5, 0.5)`.

Algorithm 1 and official `SqrtBiasSampler` initialize:

```
I_alpha = zeros(n)
I_beta = zeros(n)
T_last = zeros(n)
```

Both give `v = (1+α)/(2+α+β) = 0.5` before any observation. After the first
reward they diverge:

- zeros + success: `α=1, β=0` → `v=2/3`
- (0.5,0.5) + success: `α=1.5, β=0.5` → `v=0.625`

**R5.2 follows official code (zeros).** We do not pick the paper-text init
because it would be a silent, favorable or unfavorable, hyperparameter choice.

## Sampling

Official `sample_batch` uses `numpy.random.RandomState.choice(..., replace=False)`
on the **full pool** — no within-batch duplicates, but **revisit across steps
is allowed** (`UniformSampler` tracks `_used_in_epoch`; `SqrtBiasSampler` does not).

R5.2 matches that: 6 rounds × batch 8, with-replacement across rounds.

`t` is the round index (official batch path: `update(..., sampling_step)` for
every item in the batch, then `sampling_step += 1` once per batch).

Reward `r_i` is binary EM in `[0,1]` (`1` if `experience.success` else `0`).
Official `compute_sample_reward` averages accuracy scores and returns `0.5` if
missing; we always have a scored rollout, so binary EM is the natural analogue.

## Skill gating (not in Ace-Skill)

K=8 eligibility uses **distinct acquired task IDs** per capability. Repeating
the same task does not accelerate activation. Frontier/passive almost never
revisit, so this constraint is Ace-Skill-specific in practice.

## Frozen ExPolicy contract

- Base model Qwen2.5-3B-Instruct
- Helion world seed 13
- R5/R5.1 frozen splits
- Future 36 / retention 24 / budget 48
- No future/test gold into acquisition scoring or gating
- No learned carrier router, no LoRA assimilation sweep
"""
    (out / "ACE_SKILL_IMPLEMENTATION_AUDIT.md").write_text(text, encoding="utf-8")


def _checkpoint_row(
    *,
    method: str,
    seed: int,
    budget: int,
    payload: dict[str, Any],
    base: dict[str, Any],
    acquired: Sequence[Experience],
    first: dict[str, int],
) -> dict[str, Any]:
    dist = acquisition_distribution(acquired[:budget] if budget else [])
    return {
        "budget": budget,
        "method": method,
        "seed": seed,
        "future": {"em": payload["future_em"], "n": 36},
        "future_per_cap": payload["future_per_cap"],
        "retention": {"em": payload["retention_em"], "n": 24},
        "n_active_skills": payload["n_active_skills"],
        "delta_future": payload["future_em"] - float(base["future"]["em"]),
        "delta_retention": payload["retention_em"] - float(base["retention"]["em"]),
        "unique_tasks_per_capability": dist["unique_tasks_per_capability"],
        "total_rollouts": dist["total_rollouts"],
        "unique_tasks": dist["unique_tasks"],
        "duplicate_rollouts": dist["duplicate_rollouts"],
        "first_skill_activation": first,
        "n_skill_invocations": payload.get("n_skill_invocations"),
        "s0_fp_rate": payload.get("s0_fp_rate"),
    }


def replay_count_only_eval(
    *,
    method: str,
    seed: int,
    acquired: list[Experience],
    out: Path,
    world,
    split: dict,
    base: dict,
    base_rows: dict[str, dict],
    budget: int,
    min_support: int,
    dispatch: str,
    router=None,
) -> dict[str, Any]:
    probe_ids = list(split["future_probe_ids"]) + list(split["retention_probe"])
    future_map = future_by_cap(world, split)
    gold_docs = [d for tasks in future_map.values() for t in tasks for d in t.gold_doc_ids]
    retention = tasks_by_ids(world, split["retention_probe"])
    checkpoint_path = out / "checkpoint_metrics.jsonl"
    done_evals: set[int] = set()
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_evals.add(int(json.loads(line)["budget"]))
    last = None
    skill_dir = out / "skill_versions"
    activation_log = out / "skill_activation.jsonl"
    first_all: dict[str, int] = {}
    for n in (0, *REFIT_AT):
        if n > budget:
            break
        chunk = acquired[:n]
        if n == 0:
            first = {}
            store = SkillStore(skill_dir / "store_empty")
            events = []
        else:
            store, events, first = count_only_store(
                chunk,
                min_support=min_support,
                probe_ids=probe_ids,
                gold_docs=gold_docs,
                skill_dir=skill_dir,
                budget=n,
            )
            for cap, b in first.items():
                first_all.setdefault(cap, b)
            for ev in events:
                append_jsonl(activation_log, ev)
        if n in FULL_EVAL_AT and n not in done_evals:
            if n == 0:
                payload = {
                    "future_em": float(base["future"]["em"]),
                    "retention_em": float(base["retention"]["em"]),
                    "future_per_cap": base.get("future_per_cap") or {},
                    "n_active_skills": 0,
                    "n_skill_invocations": 0,
                    "s0_fp_rate": 0.0,
                }
            else:
                heartbeat(out, phase="eval", acquired=n, budget=budget, method=method, seed=seed)
                payload = eval_with_dispatch(
                    world,
                    future_map,
                    retention,
                    store=store,
                    base_rows=base_rows,
                    dispatch=dispatch,
                    router=router,
                    out=out,
                )
            row = _checkpoint_row(
                method=method,
                seed=seed,
                budget=n,
                payload=payload,
                base=base,
                acquired=chunk,
                first=dict(first_all),
            )
            append_jsonl(checkpoint_path, row)
            done_evals.add(n)
            last = row
        elif n in FULL_EVAL_AT:
            rows = [json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            last = next((r for r in rows if int(r["budget"]) == n), last)
    if last is None:
        raise RuntimeError(f"no eval rows for {method} seed={seed}")
    dist = acquisition_distribution(acquired)
    times = target_skill_times(first_all, budget=budget)
    summary = {
        "method": method,
        "seed": seed,
        "n_acquired": len(acquired),
        "immediate_reward": mean(float(bool(e.success)) for e in acquired) if acquired else 0.0,
        "caps_chosen": dict(Counter((maps_to_skill_capability(e.capability) or e.capability or "unknown") for e in acquired)),
        "first_skill_activation": first_all,
        "n_active_skills": last["n_active_skills"],
        "base_future_em": float(base["future"]["em"]),
        "base_retention_em": float(base["retention"]["em"]),
        "future_em": last["future"]["em"],
        "future_per_cap": last["future_per_cap"],
        "retention_em": last["retention"]["em"],
        "delta_future": last["delta_future"],
        "delta_retention": last["delta_retention"],
        "tokens": int(sum(e.cost.tokens for e in acquired)),
        "tool_calls": int(sum(e.cost.tool_calls for e in acquired)),
        "dispatch": dispatch,
        "fit_mode": "count_only_unique_task",
        "min_support": min_support,
        **dist,
        **times,
        "protocol": "r5.2 unique-task CountOnly; Ace-Skill sampler allows revisits",
    }
    write_json(out / "final_metrics.json", summary)
    write_json(out / "token_stats.json", {"acquire_tokens": summary["tokens"], "acquire_tool_calls": summary["tool_calls"]})
    write_json(out / "acquisition_distribution.json", dist)
    return summary


def run_one_seed(
    *,
    method: str,
    seed: int,
    output_dir: Path,
    model: str,
    budget: int,
    gpu: str | None,
    resume: bool,
    r5_root: Path,
    r51_root: Path,
    min_support: int = MIN_SUPPORT,
    dispatch: str = "oracle",
    round_size: int = ROUND_SIZE,
    agent=None,
    base_state=None,
    world=None,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    apply_gpu(gpu)
    out = output_dir / method / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists() and resume:
        return {"status": "already_done", "out": str(out), "method": method, "seed": seed}

    cache = cache if cache is not None else {}
    world = world or cache.get("world") or load_world()
    cache["world"] = world
    split = load_split(output_dir, seed, r5_root=r5_root)
    pool = tasks_by_ids(world, split["acquisition_pool"])
    by_id = _task_map(pool)
    base, base_rows, base_dir = load_base_bundle(seed, r52=output_dir, r5=r5_root, r51=r51_root)
    write_json(out / "base_metrics.json", {k: v for k, v in base.items() if k not in {"future_rows", "retention_rows"}})
    write_run_header(
        out,
        config={
            "script": "run_r5_2_baselines.py",
            "method": method,
            "seed": seed,
            "model": model,
            "gpu": gpu,
            "budget": budget,
            "dispatch": dispatch,
            "min_support": min_support,
            "round_size": round_size,
            "fit_mode": "count_only_unique_task",
            "world_seed": 13,
            "base_dir": str(base_dir),
        },
    )

    reused = resolve_reused_run(method, seed, r5=r5_root, r51=r51_root)
    if method in {"frontier", "passive"} and reused is not None:
        heartbeat(out, phase="reuse_count_only_eval", method=method, seed=seed, source=str(reused))
        acquired = load_acquired(reused)
        write_json(out / "REUSED.json", {"source": str(reused), "n_acquired": len(acquired)})
        summary = replay_count_only_eval(
            method=method,
            seed=seed,
            acquired=acquired,
            out=out,
            world=world,
            split=split,
            base=base,
            base_rows=base_rows,
            budget=min(budget, len(acquired)),
            min_support=min_support,
            dispatch=dispatch,
        )
        summary["reused_from"] = str(reused)
        write_json(out / "final_metrics.json", summary)
        (out / "DONE").write_text("ok\n", encoding="utf-8")
        return summary

    # Ace-Skill (or any method that must actually roll out)
    acquired = _load_jsonl_exp(out / "trajectories.jsonl") if resume else []
    if acquired:
        acquired = [e for e in acquired if e.run_id == "acquire"]
    if not acquired and resume:
        acquired = _load_jsonl_exp(out / "selected_experiences.jsonl")

    dummy_state = empty_agent(out / "dummy", model_name=model)
    diagnoser = CapabilityDiagnoser()
    e0 = [e for e in load_e0_experiences() if e.task_id not in set(split["future_probe_ids"]) | set(split["retention_probe"])]
    policy = make_acquisition(method, seed=seed)
    pending_round: list[str] = []
    t_round = len(acquired) // round_size
    state_path = out / "sampler_state.json"
    if isinstance(policy, AceSkillPrioritizedAcquisition):
        policy.bind([t.task_id for t in pool])
        if resume and state_path.exists():
            payload = load_json(state_path)
            policy.load_state(payload)
            pending_round = list(payload.get("pending_round") or [])
            t_round = int(payload.get("t", t_round))
        elif acquired:
            profile = diagnoser.diagnose(e0) if e0 else diagnoser.diagnose(acquired)
            recent_replay: list[Experience] = list(e0)
            t_round = 0
            for ridx in range(0, len(acquired), round_size):
                chunk = acquired[ridx : ridx + round_size]
                if len(chunk) == round_size:
                    policy.propose(dummy_state, profile, pool, len(chunk), recent=recent_replay, t=t_round)
                for exp in chunk:
                    policy.observe(exp.task_id, 1.0 if exp.success else 0.0, t_round)
                    recent_replay.append(exp)
                    profile = diagnoser.diagnose(recent_replay)
                if len(chunk) == round_size:
                    t_round += 1
            pending_round = []
    need_llm = len(acquired) < budget
    agent = agent or cache.get("agent")
    base_state = base_state or cache.get("base_state")
    if need_llm and agent is None:
        heartbeat(out, phase="load_model", acquired=len(acquired), budget=budget, method=method, seed=seed)
        agent, base_state = build_static_agent(world, model_path=model, output_dir=out / "base_agent")
        cache["agent"] = agent
        cache["base_state"] = base_state
    recent = list(e0) + list(acquired)
    t0 = time.time()

    def save_sampler(pending: list[str]) -> None:
        if isinstance(policy, AceSkillPrioritizedAcquisition):
            payload = policy.state_dict()
            payload["t"] = t_round
            payload["pending_round"] = pending
            write_json(state_path, payload)

    remaining = list(pool)
    if method not in {"ace_skill_prioritized", "ace_skill"}:
        seen = {e.task_id for e in acquired}
        remaining = [t for t in remaining if t.task_id not in seen]

    while len(acquired) < budget:
        n = len(acquired)
        want = min(round_size, budget - n)
        profile = diagnoser.diagnose(recent) if recent else diagnoser.diagnose(e0)
        if pending_round:
            batch_ids = pending_round[:want]
            pending_round = pending_round[want:]
        else:
            if method in {"ace_skill_prioritized", "ace_skill"}:
                props = policy.propose(dummy_state, profile, pool, want, recent=recent, t=t_round)  # type: ignore[call-arg]
            else:
                props = policy.propose(dummy_state, profile, remaining, want, recent=recent)
            batch_ids = list(props[0].task_ids)[:want]
        save_sampler(batch_ids)
        for tid in list(batch_ids):
            task = by_id.get(tid) or next((t for t in remaining if t.task_id == tid), None)
            if task is None:
                task = remaining[0] if remaining else pool[n % len(pool)]
            heartbeat(
                out,
                phase="acquire",
                acquired=len(acquired),
                budget=budget,
                method=method,
                seed=seed,
                task_id=task.task_id,
                round=t_round,
            )
            if agent is None:
                raise RuntimeError("LLM agent required for acquisition")
            agent.skill_store = None
            agent.use_executable_skill = False
            exp = agent.run_task(task, state=base_state)
            exp.run_id = "acquire"
            acquired.append(exp)
            recent.append(exp)
            if method not in {"ace_skill_prioritized", "ace_skill"}:
                remaining = [t for t in remaining if t.task_id != task.task_id]
            reward = 1.0 if exp.success else 0.0
            if isinstance(policy, AceSkillPrioritizedAcquisition):
                policy.observe(task.task_id, reward, t_round)
            append_jsonl(out / "trajectories.jsonl", json.loads(exp.model_dump_json()))
            append_jsonl(
                out / "selected_experiences.jsonl",
                {
                    "step": len(acquired),
                    "task_id": task.task_id,
                    "capability": task.capability,
                    "stage": task.stage,
                    "success": exp.success,
                    "reward": reward,
                    "experience_id": exp.experience_id,
                    "tokens": exp.cost.tokens,
                    "tool_calls": exp.cost.tool_calls,
                    "round": t_round,
                    "predicted": exp.predicted_answer,
                },
            )
            batch_ids.pop(0)
            save_sampler(batch_ids)
            n = len(acquired)
            if n in REFIT_AT or n >= budget:
                pass
        t_round += 1
        pending_round = []
        save_sampler([])

    summary = replay_count_only_eval(
        method=method,
        seed=seed,
        acquired=acquired,
        out=out,
        world=world,
        split=split,
        base=base,
        base_rows=base_rows,
        budget=len(acquired),
        min_support=min_support,
        dispatch=dispatch,
    )
    summary["wall_seconds"] = time.time() - t0
    write_json(out / "final_metrics.json", summary)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    heartbeat(out, phase="done", acquired=len(acquired), budget=budget, method=method, seed=seed)
    return summary


def maybe_text_router_eval(
    *,
    method: str,
    seed: int,
    output_dir: Path,
    r5_root: Path,
    r51_root: Path,
    world=None,
) -> dict[str, Any] | None:
    out = output_dir / method / f"seed_{seed}" / "text_router"
    if (out / "DONE").exists():
        return load_json(out / "metrics.json")
    run = output_dir / method / f"seed_{seed}"
    reused = resolve_reused_run(method, seed, r5=r5_root, r51=r51_root)
    src = run
    if not (run / "trajectories.jsonl").exists() and reused is not None:
        src = reused
    acquired = load_acquired(src)
    if not acquired and reused is not None:
        acquired = load_acquired(reused)
    if not acquired:
        return None
    world = world or load_world()
    split = load_split(output_dir, seed, r5_root=r5_root)
    base, base_rows, _ = load_base_bundle(seed, r52=output_dir, r5=r5_root, r51=r51_root)
    router, router_metrics = train_router_for_split(world, split, seed=seed)
    probe_ids = list(split["future_probe_ids"]) + list(split["retention_probe"])
    future_map = future_by_cap(world, split)
    gold_docs = [d for tasks in future_map.values() for t in tasks for d in t.gold_doc_ids]
    retention = tasks_by_ids(world, split["retention_probe"])
    store, _events, _first = count_only_store(
        acquired,
        min_support=MIN_SUPPORT,
        probe_ids=probe_ids,
        gold_docs=gold_docs,
        skill_dir=run / "skill_versions_router",
        budget=len(acquired),
    )
    out.mkdir(parents=True, exist_ok=True)
    heartbeat(out, phase="text_router", method=method, seed=seed)
    payload = eval_with_dispatch(
        world,
        future_map,
        retention,
        store=store,
        base_rows=base_rows,
        dispatch="text_router",
        router=router,
        out=out,
    )
    rec = {
        "method": method,
        "seed": seed,
        "dispatch": "text_router",
        "future_em": payload["future_em"],
        "retention_em": payload["retention_em"],
        "delta_future": payload["future_em"] - float(base["future"]["em"]),
        "delta_retention": payload["retention_em"] - float(base["retention"]["em"]),
        "router_metrics": {k: v for k, v in router_metrics.items() if k != "confusion"},
    }
    write_json(out / "metrics.json", rec)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    return rec
