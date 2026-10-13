"""One learning sequence. Future scores are not computed here, and trajectories are cached by state hash."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from expolicy.paper_c.assimilator import FrozenAssimilator, assert_same_algorithm
from expolicy.paper_c.constants import ACCOUNT_COMPUTE, ACCOUNT_INTERACTION, STATUS_OK
from expolicy.paper_c.features import pre_acquisition_features
from expolicy.paper_c.ledger import BudgetLedger
from expolicy.paper_c.policies import AcquisitionView
from expolicy.paper_c.schema import empty_log
from expolicy.paper_c.utility import acquisition_gain
from expolicy.paper_c.world import SyntheticWorld, Task
from expolicy.r9_lab.io_util import sha256_json
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState


class CacheReuseError(RuntimeError):
    pass


class TrajectoryCache:
    """Reuse a trajectory only when the state hash matches. A task id is not a key by itself."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def get(self, state_hash: str, task_id: str) -> dict[str, Any] | None:
        row = self._rows.get(task_id)
        if row is None:
            return None
        if row["state_hash"] != state_hash:
            raise CacheReuseError(f"refusing to reuse {task_id} from state {row['state_hash']} at {state_hash}")
        return dict(row["episode"])

    def put(self, state_hash: str, task_id: str, episode: Mapping[str, Any]) -> None:
        existing = self._rows.get(task_id)
        if existing is not None and existing["state_hash"] != state_hash:
            raise CacheReuseError(f"refusing to store {task_id} under a second state")
        self._rows[task_id] = {"state_hash": state_hash, "episode": dict(episode)}


class Checkpoint:
    def __init__(
        self,
        *,
        policy: str,
        sequence_seed: int,
        round_index: int,
        state: PersistentState,
        interaction_cost: float,
        episode_cost: float,
        token_cost: float,
        probe_calls: int,
    ):
        self.policy = policy
        self.sequence_seed = sequence_seed
        self.round_index = round_index
        self.state = state
        self.interaction_cost = interaction_cost
        self.episode_cost = episode_cost
        self.token_cost = token_cost
        self.probe_calls = probe_calls


def _copy_pool(pool: Mapping[str, Sequence[Task]]) -> dict[str, list[Task]]:
    return {family: list(tasks) for family, tasks in pool.items()}


def _draw(pool: dict[str, list[Task]], family: str, count: int) -> list[Task]:
    if len(pool[family]) < count:
        raise RuntimeError(f"task pool for {family} is exhausted")
    chosen = pool[family][:count]
    pool[family] = pool[family][count:]
    return chosen


def run_learning_sequence(
    start: PersistentState,
    policy: Any,
    world: SyntheticWorld,
    assimilator: FrozenAssimilator,
    pool: Mapping[str, Sequence[Task]],
    *,
    work_dir: Path,
    ledger: BudgetLedger,
    run_id: str,
    sequence_seed: int,
    families: Sequence[str],
    declared_envs: Sequence[str],
    n_rounds: int,
    tasks_per_round: int,
    feedback_rounds: Sequence[int],
    eval_rounds: Sequence[int],
    retention_weight: float = 0.5,
    cache: TrajectoryCache | None = None,
) -> dict[str, Any]:
    """Run one policy from its own clone. Held-out future scores are not read in this function."""
    work_dir.mkdir(parents=True, exist_ok=True)
    state = clone_state(start, work_dir / "state")
    local_pool = _copy_pool(pool)
    cache = cache or TrajectoryCache()
    expected = assimilator.algorithm_hash()
    pool_hash = sha256_json({family: [task.task_id for task in tasks] for family, tasks in local_pool.items()})
    logs: list[dict[str, Any]] = []
    checkpoints: list[Checkpoint] = []
    episode_cost = 0.0
    token_cost = 0.0
    feedback = {int(item) for item in feedback_rounds}
    evaluate_at = {int(item) for item in eval_rounds}

    def checkpoint(round_index: int) -> None:
        cloned = clone_state(state, work_dir / "ckpt" / str(round_index))
        checkpoints.append(
            Checkpoint(
                policy=policy.name,
                sequence_seed=sequence_seed,
                round_index=round_index,
                state=cloned,
                interaction_cost=ledger.interaction_total(),
                episode_cost=episode_cost,
                token_cost=token_cost,
                probe_calls=len(world.future) + len(world.shadow_eval) + len(world.retention),
            )
        )

    if 0 in evaluate_at:
        checkpoint(0)
    for round_index in range(1, int(n_rounds) + 1):
        features = {
            family: pre_acquisition_features(
                state,
                family=family,
                env=world.env,
                families=families,
                public={"public_difficulty": 0.5, "public_estimated_tools": 1.0},
                history=policy.history,
                declared_envs=declared_envs,
            )
            for family in families
        }
        view = AcquisitionView(
            env=world.env,
            round_index=round_index,
            state_hash=state.content_hash(),
            families=tuple(families),
            remaining={family: len(local_pool[family]) for family in families},
            features_by_family=features,
            pool_hash=pool_hash,
        )
        family = policy.select(view)
        tasks = _draw(local_pool, family, tasks_per_round)
        state_hash = state.content_hash()
        episodes: list[dict[str, Any]] = []
        actor_tokens = 0
        for task in tasks:
            cached = cache.get(state_hash, task.task_id)
            if cached is None:
                episode = world.rollout(state, task)
                cache.put(state_hash, task.task_id, episode)
            else:
                episode = cached
            episodes.append(episode)
            actor_tokens += int(episode.get("actor_tokens") or 0)
            ledger.add(
                account=ACCOUNT_INTERACTION,
                kind="acquisition_episode",
                interaction_units=1.0,
                actor_tokens=int(episode.get("actor_tokens") or 0),
                teacher_tokens=int(episode.get("teacher_tokens") or 0),
                tool_calls=int(episode.get("tool_calls") or 0),
                detail=task.task_id,
            )
        episode_cost += float(len(tasks))
        token_cost += float(actor_tokens)
        scheduled = round_index in feedback
        before_shadow = world.score(state, world.shadow_eval) if scheduled else None
        before_retention = world.score(state, world.retention) if scheduled else None
        if scheduled:
            ledger.add(
                account=ACCOUNT_INTERACTION,
                kind="shadow_feedback",
                interaction_units=float(len(world.shadow_eval) + len(world.retention)),
                validation_calls=len(world.shadow_eval) + len(world.retention),
                detail=f"before:{round_index}",
            )
        backup = clone_state(state, work_dir / f"backup-{round_index}")
        result = assimilator.apply(state, episodes)
        assert_same_algorithm(expected, result)
        accepted = bool(result["accepted"])
        if not accepted:
            state = clone_state(backup, work_dir / "state")
            ledger.add(account=ACCOUNT_COMPUTE, kind="rollback", detail=f"round:{round_index}")
        successes = [1.0 if item.get("success") else 0.0 for item in episodes]
        policy.observe_outcome(family, sum(successes) / len(successes))
        gain_value = None
        if scheduled and accepted and before_shadow is not None and before_retention is not None:
            after_shadow = world.score(state, world.shadow_eval)
            after_retention = world.score(state, world.retention)
            ledger.add(
                account=ACCOUNT_INTERACTION,
                kind="shadow_feedback",
                interaction_units=float(len(world.shadow_eval) + len(world.retention)),
                validation_calls=len(world.shadow_eval) + len(world.retention),
                detail=f"after:{round_index}",
            )
            if after_shadow is not None and after_retention is not None:
                gain_value = acquisition_gain(
                    shadow_before=float(before_shadow),
                    shadow_after=float(after_shadow),
                    retention_before=float(before_retention),
                    retention_after=float(after_retention),
                    retention_weight=retention_weight,
                )["g"]
                policy.observe_gain(
                    {
                        "family": family,
                        "env": world.env,
                        "state_id": state_hash,
                        "sequence_id": str(sequence_seed),
                        "source_template": tasks[0].source_template,
                        "features": features[family],
                        "gain": gain_value,
                        "split": "train_shadow",
                    }
                )
                ledger.add(account=ACCOUNT_COMPUTE, kind="gain_model_update", detail=f"round:{round_index}")
        logs.append(
            empty_log(
                run_id=run_id,
                policy=policy.name,
                sequence_seed=sequence_seed,
                round=round_index,
                state_hash=state_hash,
                candidate_pool_hash=pool_hash,
                observable_features=features[family],
                selected_family=family,
                selected_task_ids=[task.task_id for task in tasks],
                acquisition_episodes=len(tasks),
                assimilation_accepted=accepted,
                shadow_feedback_available=scheduled and gain_value is not None,
                gain_target=gain_value,
                actor_tokens=actor_tokens,
                teacher_tokens=int(result.get("teacher_tokens") or 0),
                validation_calls=len(world.shadow_eval) + len(world.retention) if scheduled else 0,
                cumulative_cost=ledger.interaction_total(),
                future_score_offline=None,
                infra_status=STATUS_OK,
            )
        )
        if round_index in evaluate_at:
            checkpoint(round_index)
    return {"logs": logs, "checkpoints": checkpoints, "policy": policy.name, "sequence_seed": sequence_seed}
