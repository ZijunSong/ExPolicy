"""GPU entry points. Importing this module does not start a model or an environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from expolicy.paper_a.audit import AuditTask, audit_noop
from expolicy.paper_a.config import PaperAConfig, cache_key, verify_assets
from expolicy.paper_a.constants import ACTIONS, RETENTION, SHADOW, STATUS_INFRA, STATUS_MISSING, STATUS_OK
from expolicy.paper_a.features import predecision_features
from expolicy.paper_a.matrix import build_utility_matrix
from expolicy.r9_lab.assimilators.discard import apply_discard
from expolicy.r9_lab.assimilators.episodic import apply_episodic, retrieve_episodic
from expolicy.r9_lab.assimilators.semantic import apply_semantic, retrieve_semantic
from expolicy.r9_lab.benchmark.split import load_split, records_in_bucket
from expolicy.r9_lab.constants import LAB_SYSTEM_PROMPT
from expolicy.r9_lab.io_util import read_json, sha256_json, write_json
from expolicy.r9_lab.matrix_runtime import apply_action
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState


class OfficialRunnerUnavailable(RuntimeError):
    pass


def probe_tasks(split: Mapping[str, Any], env: str) -> list[dict[str, Any]]:
    tasks = []
    for bucket in (SHADOW, RETENTION):
        for record in records_in_bucket(split, env, bucket):
            tasks.append(
                {
                    "probe_task_id": record.task_id,
                    "eval_split": bucket,
                    "instruction": record.instruction,
                    "env": env,
                    "sample_index": record.sample_index,
                }
            )
    return tasks


def load_capability(cfg: PaperAConfig) -> dict[str, dict[str, Any]]:
    path = cfg.path("capability.path")
    by_env: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return by_env
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            env = str(row.get("env") or "")
            if not env:
                continue
            by_env[env] = {
                "source": str(cfg.raw["capability"]["source"]),
                "shadow": float(row["success"]),
                "retention": None,
                "n": int(float(row["n"])),
                "file": str(path),
            }
    return by_env


def _tracing_client(inner: Any, *, seed: int, max_tokens: int) -> Any:
    class _Client:
        thread_safe = False

        def __init__(self) -> None:
            self.calls: list[list[dict[str, Any]]] = []

        def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Any:
            self.calls.append([dict(message) for message in messages])
            return inner.chat(messages, seed=seed, max_tokens=max_tokens, temperature=0.0)

        def reset(self) -> None:
            self.calls = []

    return _Client()


def run_probes(lab: Any, client: Any, state: PersistentState, probes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not hasattr(client, "reset"):
        client = _tracing_client(client, seed=0, max_tokens=1024)
    items = []
    for probe in probes:
        client.reset()
        instruction = str(probe.get("instruction") or "")
        episodic = [rec.task_id for rec in retrieve_episodic(state, instruction, trace=False)]
        semantic = [entry.dedup_key() for entry in retrieve_semantic(state, instruction, trace=False)]
        try:
            result = lab.run_episode(
                probe.get("sample_index"),
                client,
                state=state,
                task_id=str(probe.get("probe_task_id") or ""),
                instruction_hint=instruction,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"probe_missing {probe.get('probe_task_id')} {type(exc).__name__}: {exc}",
                flush=True,
            )
            items.append(
                {
                    "probe_task_id": probe.get("probe_task_id"),
                    "eval_split": probe.get("eval_split"),
                    "y_before": None,
                    "y_after": None,
                    "y": None,
                    "tokens": None,
                    "tool_calls": None,
                    "elapsed_s": None,
                    "infra_status": STATUS_MISSING,
                    "error": f"{type(exc).__name__}: {exc}",
                    "prompt_hash": None,
                    "retrieval_order": {"episodic": episodic, "semantic": semantic},
                }
            )
            continue
        infra = bool(getattr(result, "infra_failure", False))
        prompt_hash = sha256_json(client.calls)
        blob = "\n".join(message.get("content") or "" for call in client.calls for message in call)
        items.append(
            {
                "probe_task_id": probe.get("probe_task_id"),
                "eval_split": probe.get("eval_split"),
                "y": None if infra else (1.0 if result.success else 0.0),
                "tokens": int(getattr(result, "tokens", 0) or 0),
                "tool_calls": int(getattr(result, "steps", 0) or 0),
                "elapsed_s": float(getattr(result, "wall_time", 0.0) or 0.0),
                "infra_status": STATUS_INFRA if infra else STATUS_OK,
                "prompt_hash": prompt_hash,
                "retrieval_order": {"episodic": episodic, "semantic": semantic},
                "branch_name_in_prompt": "A_V1_BRANCH::" in blob,
                "env_reset_id": str(probe.get("sample_index") or ""),
            }
        )
        if len(items) % 10 == 0:
            print(
                f"probe_progress n={len(items)} last={probe.get('probe_task_id')} ok={items[-1]['infra_status']}",
                flush=True,
            )
    return items


def make_audit_evaluator(lab: Any, client: Any):
    def evaluate(state, tasks, *, mode, repeat_id, concurrency, branch):  # noqa: ARG001
        if int(concurrency) > 1:
            raise OfficialRunnerUnavailable("official LAB task is not thread-safe; refusing to fake a concurrent replay")
        probes = [
            {
                "probe_task_id": task.probe_task_id,
                "eval_split": task.eval_split,
                "instruction": task.instruction,
                "sample_index": getattr(task, "sample_index", None) or task.probe_task_id.split(":", 1)[-1],
            }
            for task in tasks
        ]
        rows = run_probes(lab, client, state, probes)
        for row in rows:
            row["branch"] = branch
            row["mode"] = mode
            row["repeat_id"] = repeat_id
        return rows

    evaluate.thread_safe = False  # type: ignore[attr-defined]
    return evaluate


def run_measurement_audit(
    cfg: PaperAConfig,
    *,
    env: str,
    experiences: Sequence[Mapping[str, Any]],
    lab: Any,
    client: Any,
    work_dir: Path,
    model_identity: Mapping[str, Any],
    s0: PersistentState,
) -> dict[str, Any]:
    split = load_split(cfg.path("split.path"))
    tasks = [
        AuditTask(
            probe_task_id=row["probe_task_id"],
            eval_split=row["eval_split"],
            instruction=row["instruction"],
            env=env,
        )
        for row in probe_tasks(split, env)
    ]
    report = audit_noop(
        s0,
        tasks,
        make_audit_evaluator(lab, client),
        experiences=experiences,
        work_dir=work_dir,
        model_identity=model_identity,
        system_prompt=LAB_SYSTEM_PROMPT,
        tool_config={"env": env, "max_round_locked": True},
        env_snapshot_id=env,
        repeats=int(cfg.raw["measurement"]["noise_repeats"]),
        equivalence=float(cfg.raw["measurement"]["equivalence_success"]),
        ci_z=float(cfg.raw["measurement"]["ci_z"]),
        formal_concurrency=int(cfg.raw["measurement"]["formal_concurrency"]),
        require_formal_concurrency_replay=bool(cfg.raw["measurement"]["require_formal_concurrency_replay"]),
    )
    return report.to_dict()


def _baseline_path(cache_dir: Path, config_hash: str, env: str) -> Path:
    return cache_dir / config_hash / "_baseline" / f"{env}.json"


def evaluate_baseline(lab: Any, client: Any, s0: PersistentState, probes: Sequence[Mapping[str, Any]], path: Path) -> dict[str, float]:
    if path.exists():
        stored = read_json(path)
        if stored.get("state_hash") == s0.content_hash() and stored.get("complete") is True:
            return {str(k): float(v) for k, v in stored["y"].items()}
    branch = clone_state(s0, path.parent / "state")
    items = run_probes(lab, client, branch, probes)
    scores = {}
    for item in items:
        if item["infra_status"] != STATUS_OK or item["y"] is None:
            write_json(path, {"complete": False, "state_hash": s0.content_hash(), "items": items})
            raise OfficialRunnerUnavailable(f"baseline probe missing: {item['probe_task_id']}")
        scores[str(item["probe_task_id"])] = float(item["y"])
    write_json(path, {"complete": True, "state_hash": s0.content_hash(), "y": scores})
    return scores


def make_cell_evaluator(
    *,
    lab: Any,
    client: Any,
    s0: PersistentState,
    experiences: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, float],
    capability: Mapping[str, Any] | None,
    work_dir: Path,
):
    def evaluate_cell(batch: Mapping[str, Any], action: str, probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if action not in ACTIONS:
            raise ValueError(action)
        branch_dir = work_dir / str(batch["batch_id"]) / action
        state = clone_state(s0, branch_dir / "state")
        before = state.content_hash()
        features = predecision_features(state, experiences, env=str(batch["env"]), capability=capability)
        applied = apply_action(action, state, experiences, client=client)
        items = run_probes(lab, client, state, probes)
        for item in items:
            task_id = str(item["probe_task_id"])
            item["y_before"] = baseline.get(task_id)
            item["y_after"] = None if item["infra_status"] != STATUS_OK else item["y"]
            if item["y_before"] is None:
                item["infra_status"] = STATUS_MISSING
                item["y_after"] = None
        return {
            "state_hash_before": before,
            "state_hash_after": state.content_hash(),
            "changed": applied.get("changed"),
            "accepted": applied.get("accepted", applied.get("changed")),
            "items": items,
            "features": features,
            "operator_tokens": sum(int(item.get("tokens") or 0) for item in items),
            "operator_elapsed_s": sum(float(item.get("elapsed_s") or 0.0) for item in items),
        }

    return evaluate_cell


def load_acquired_experiences(matrix_root: Path, batch_id: str) -> list[dict[str, Any]] | None:
    path = matrix_root / batch_id / "_acquire" / "DONE.json"
    if not path.exists():
        return None
    payload = read_json(path)
    experiences = payload.get("experiences")
    if not experiences:
        return None
    return list(experiences)


def build_live_matrix(
    cfg: PaperAConfig,
    batches: Sequence[Mapping[str, Any]],
    *,
    labs: Mapping[str, Any],
    client: Any,
    s0_by_env: Mapping[str, PersistentState],
    capability_by_env: Mapping[str, Mapping[str, Any]],
    cache_dir: Path,
    config_hash: str,
    model_hash: str,
) -> dict[str, Any]:
    """One evaluator per environment, because each batch keeps its own acquired experiences."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    features: dict[str, Any] = {}
    legacy: list[dict[str, Any]] = []
    split = load_split(cfg.path("split.path"))
    matrix_root = cfg.path("legacy.r9_matrix")
    by_env: dict[str, list[Mapping[str, Any]]] = {}
    for batch in batches:
        by_env.setdefault(str(batch["env"]), []).append(batch)
    for env, env_batches in by_env.items():
        probes = probe_tasks(split, env)
        baseline = evaluate_baseline(
            labs[env],
            client,
            s0_by_env[env],
            probes,
            _baseline_path(cache_dir, config_hash, env),
        )
        for batch in env_batches:
            experiences = load_acquired_experiences(matrix_root, str(batch["batch_id"]))
            if experiences is None:
                failures.append({"batch_id": batch["batch_id"], "infra_status": STATUS_MISSING, "error": "missing acquisition"})
                continue
            evaluator = make_cell_evaluator(
                lab=labs[env],
                client=client,
                s0=s0_by_env[env],
                experiences=experiences,
                baseline=baseline,
                capability=capability_by_env.get(env),
                work_dir=cache_dir / "states" / env,
            )

            def _call(one_batch, action, one_probes, evaluator=evaluator):
                return evaluator(one_batch, action, one_probes)

            built = build_utility_matrix(
                [batch],
                ACTIONS,
                {env: probes},
                _call,
                cache_dir=cache_dir,
                config_hash=config_hash,
                model_hash=model_hash,
            )
            rows.extend(built["rows"])
            failures.extend(built["failures"])
            costs.extend(built["costs"])
            legacy.extend(built["legacy"])
            features.update(built.get("features") or {})
    return {"rows": rows, "failures": failures, "costs": costs, "legacy": legacy, "features": features, "config_hash": config_hash}


def asset_cache_key(cfg: PaperAConfig) -> tuple[str, dict[str, str]]:
    hashes = verify_assets(cfg)
    return cache_key(cfg, hashes), hashes


# Re-export the assimilators the sequence runner should call, so execute stays the only GPU-facing module.
APPLY = {
    "DISCARD": apply_discard,
    "EPISODIC_REPLAY": apply_episodic,
    "SEMANTIC_MEMORY": apply_semantic,
}


def plan_episode_counts(cfg: PaperAConfig) -> dict[str, Any]:
    """Count the episodes a full run would score. This does not start the model."""
    split = load_split(cfg.path("split.path"))
    frozen = read_json(cfg.path("frozen_batches.path"))
    repeats = int(cfg.raw["measurement"]["noise_repeats"])
    modes = 2 if cfg.raw["measurement"]["require_formal_concurrency_replay"] else 1
    batches_per_env = int(cfg.raw["measurement"]["audit_batches_per_env"])
    by_env = {}
    audit = 0
    matrix = 0
    for env in cfg.environments:
        n_probe = len(probe_tasks(split, env))
        n_future = len(records_in_bucket(split, env, "future_test"))
        n_batches = sum(1 for batch in frozen["batches"] if batch["env"] == env)
        # BASE, DISCARD, and rejected-edit, on the shared probes, for each sampled batch.
        audit_env = n_probe * 3 * repeats * modes * batches_per_env
        matrix_env = n_probe * len(cfg.actions) * n_batches + n_probe
        by_env[env] = {"probes": n_probe, "future_test": n_future, "batches": n_batches, "audit": audit_env, "matrix": matrix_env}
        audit += audit_env
        matrix += matrix_env
    rounds = int(cfg.raw["decision"]["sequence_rounds"])
    eval_points = len(cfg.raw["decision"]["eval_rounds"])
    policies = 4
    sequence = 0
    for env, info in by_env.items():
        sequence += (info["probes"] + info["future_test"]) * eval_points * policies
    return {
        "audit_episodes": audit,
        "matrix_episodes": matrix,
        "sequence_episodes": sequence,
        "by_env": by_env,
        "probe_before_approval": int(cfg.raw["budget"]["probe_episodes_before_remaining_approval"]),
        "note": "future_test is counted only in the sequence stage. The matrix does not score it.",
    }


def run_sequences(
    cfg: PaperAConfig,
    *,
    labs: Mapping[str, Any],
    client: Any,
    s0_by_env: Mapping[str, PersistentState],
    selectors: Mapping[str, Any],
    capability_by_env: Mapping[str, Mapping[str, Any]],
    work_dir: Path,
) -> dict[str, Any]:
    """Four shared rounds per environment. Future scores stay inside the runner result."""
    from expolicy.paper_a.sequence import run_fixed_acquisition_sequence

    split = load_split(cfg.path("split.path"))
    frozen = read_json(cfg.path("frozen_batches.path"))
    n_rounds = int(cfg.raw["decision"]["sequence_rounds"])
    per_round = int(cfg.raw["decision"]["experiences_per_round"])
    summaries = {}
    for env in cfg.environments:
        batches = [batch for batch in frozen["batches"] if batch["env"] == env][:n_rounds]
        if len(batches) < n_rounds:
            raise OfficialRunnerUnavailable(f"{env} does not have {n_rounds} frozen batches")
        rounds = []
        for batch in batches:
            experiences = load_acquired_experiences(cfg.path("legacy.r9_matrix"), str(batch["batch_id"]))
            if not experiences or len(experiences) < per_round:
                raise OfficialRunnerUnavailable(f"missing acquired experiences for {batch['batch_id']}")
            rounds.append(experiences[:per_round])
        probes = probe_tasks(split, env)
        for record in records_in_bucket(split, env, "future_test"):
            probes.append(
                {
                    "probe_task_id": record.task_id,
                    "eval_split": "future_test",
                    "instruction": record.instruction,
                    "env": env,
                    "sample_index": record.sample_index,
                }
            )

        def evaluate(state: PersistentState, round_index: int, env: str = env, probes: list[dict[str, Any]] = probes) -> dict[str, Any]:
            _ = round_index
            items = run_probes(labs[env], client, state, probes)

            def _mean(split_name: str) -> float | None:
                chosen = [item for item in items if item.get("eval_split") == split_name]
                if not chosen or any(item.get("infra_status") != STATUS_OK or item.get("y") is None for item in chosen):
                    return None
                return float(sum(float(item["y"]) for item in chosen) / len(chosen))

            return {
                "shadow": _mean("shadow_reward_probe"),
                "retention": _mean("retention_probe"),
                "future": _mean("future_test"),
                "tokens": sum(int(item.get("tokens") or 0) for item in items),
                "tool_calls": sum(int(item.get("tool_calls") or 0) for item in items),
                "elapsed_s": sum(float(item.get("elapsed_s") or 0.0) for item in items),
                "infra_status": STATUS_OK if all(item.get("infra_status") == STATUS_OK for item in items) else STATUS_MISSING,
            }

        def apply_fn(state: PersistentState, action: str, experiences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            if action not in APPLY:
                raise ValueError(action)
            payload = APPLY[action](state, experiences) if action != "SEMANTIC_MEMORY" else APPLY[action](state, experiences, client)
            payload["accepted"] = payload.get("accepted", payload.get("changed"))
            return payload

        summaries[env] = run_fixed_acquisition_sequence(
            s0_by_env[env],
            rounds,
            selectors,
            apply_fn,
            evaluate,
            work_dir=work_dir / env,
            env=env,
            eval_at=tuple(cfg.raw["decision"]["eval_rounds"]),
            capability=capability_by_env.get(env),
            trained_on_states=["S0"],
        )
    return {"envs": summaries, "trained_on_states": ["S0"], "ood_state": True, "teacher_forced_gain_copied": False}


def measure_probe_cost(lab: Any, client: Any, state: PersistentState, probes: Sequence[Mapping[str, Any]], *, n: int, work_dir: Path) -> dict[str, Any]:
    sample = list(probes)[: int(n)]
    branch = clone_state(state, work_dir / "probe_state")
    items = run_probes(lab, client, branch, sample)
    attempted = len(items) or 1
    tokens = sum(int(item.get("tokens") or 0) for item in items)
    tool_calls = sum(int(item.get("tool_calls") or 0) for item in items)
    elapsed = sum(float(item.get("elapsed_s") or 0.0) for item in items)
    return {
        "n_attempted": len(items),
        "n_ok": sum(1 for item in items if item.get("infra_status") == STATUS_OK),
        "tokens": tokens,
        "tool_calls": tool_calls,
        "elapsed_s": elapsed,
        "tokens_per_episode": tokens / attempted,
        "tool_calls_per_episode": tool_calls / attempted,
        "elapsed_s_per_episode": elapsed / attempted,
        "items": items,
    }
