"""Fixed acquisition order, independent states. Test scores are recorded and not fed back."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from expolicy.paper_a.constants import A0_DISCARD
from expolicy.paper_a.features import LeakageError, predecision_features
from expolicy.paper_a.router import _conservative_choice
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState


@dataclass(frozen=True)
class DecisionView:
    """What a selector is allowed to see. Probe and future scores are not fields."""

    env: str
    round_index: int
    features: Mapping[str, Any]
    state_hash: str
    n_experiences: int

    def __getattr__(self, name: str) -> Any:
        if "future" in name or name in {"utility", "shadow_after", "retention_after", "y_after"}:
            raise LeakageError(f"selector requested {name}")
        raise AttributeError(name)


class FixedSelector:
    def __init__(self, action: str):
        self.action = action

    def select(self, view: DecisionView) -> str:
        return self.action

    def consume_cost(self) -> dict[str, Any]:
        return {"kind": "router_infer", "tokens": 0, "tool_calls": 0, "elapsed_s": 0.0}


class EnvOnlySelector:
    def __init__(self, action_by_env: Mapping[str, str]):
        self.action_by_env = dict(action_by_env)

    def select(self, view: DecisionView) -> str:
        return self.action_by_env.get(view.env, A0_DISCARD)

    def consume_cost(self) -> dict[str, Any]:
        return {"kind": "router_infer", "tokens": 0, "tool_calls": 0, "elapsed_s": 0.0}


class FrozenRidgeSelector:
    """In-memory copy of a fold that was already frozen. It does not refit on test scores."""

    def __init__(
        self,
        models: Mapping[str, Any],
        boot: Sequence[Mapping[str, Any]],
        fallback: Mapping[str, str],
        threshold: float,
        *,
        actions: Sequence[str],
        quantile: float,
        force_argmax: bool = False,
    ):
        self.models = models
        self.boot = list(boot)
        self.fallback = dict(fallback)
        self.threshold = float(threshold)
        self.actions = tuple(actions)
        self.quantile = float(quantile)
        self.force_argmax = force_argmax
        self.last_label = ""

    def select(self, view: DecisionView) -> str:
        action, label = _conservative_choice(
            view.features,
            self.models,
            self.boot,
            self.fallback.get(view.env, A0_DISCARD),
            actions=self.actions,
            threshold=self.threshold,
            q=self.quantile,
            force_argmax=self.force_argmax,
        )
        self.last_label = label
        return action

    def consume_cost(self) -> dict[str, Any]:
        return {"kind": "router_infer", "tokens": 0, "tool_calls": 0, "elapsed_s": 0.0}


def selector_from_router(router_result: Mapping[str, Any], *, fold: int = 0, variant: str = "M") -> FrozenRidgeSelector:
    payload = router_result["folds"][fold]["variants"][variant]
    return FrozenRidgeSelector(
        payload["_models"],
        payload["_boot"],
        payload["fallback"],
        float(payload["threshold"]),
        actions=tuple(payload["_models"].keys()),
        quantile=0.10,
        force_argmax=variant == "A4",
    )


def run_fixed_acquisition_sequence(
    snapshot: PersistentState | Path | str,
    experience_rounds: Sequence[Sequence[Mapping[str, Any]]],
    selectors: Mapping[str, Any],
    apply_fn: Callable[[PersistentState, str, Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    evaluate_fn: Callable[[PersistentState, int], Mapping[str, Any]],
    *,
    work_dir: Path,
    env: str,
    eval_at: Sequence[int] = (0, 2, 4),
    capability: Mapping[str, Any] | None = None,
    trained_on_states: Sequence[str] = ("S0",),
) -> dict[str, Any]:
    """Run every selector on the same experience order, each from its own clone of S0.

    `evaluate_fn` may return a future-test score. That score is stored on the result
    and is not copied into the next DecisionView. Teacher-forced matrix gains are not
    written into this summary.
    """
    if isinstance(snapshot, PersistentState):
        snapshot.save()
        source: PersistentState | Path = snapshot
    else:
        source = Path(snapshot)
    work_dir.mkdir(parents=True, exist_ok=True)
    states = {name: clone_state(source, work_dir / name / "state") for name in selectors}
    eval_points = set(int(point) for point in eval_at)
    n_rounds = len(experience_rounds)
    histories: dict[str, list[dict[str, Any]]] = {name: [] for name in selectors}
    costs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for step in range(n_rounds + 1):
        if step in eval_points:
            for name, state in states.items():
                metrics = dict(evaluate_fn(state, step))
                future = metrics.pop("future", None)
                histories[name].append(
                    {
                        "round": step,
                        "state_hash": state.content_hash(),
                        "shadow": metrics.get("shadow"),
                        "retention": metrics.get("retention"),
                        "future": future,
                        "tokens": metrics.get("tokens"),
                        "tool_calls": metrics.get("tool_calls"),
                        "elapsed_s": metrics.get("elapsed_s"),
                        "infra_status": metrics.get("infra_status") or "ok",
                    }
                )
                costs.append(
                    {
                        "kind": "deploy_eval",
                        "policy": name,
                        "round": step,
                        "tokens": metrics.get("tokens"),
                        "tool_calls": metrics.get("tool_calls"),
                        "elapsed_s": metrics.get("elapsed_s"),
                    }
                )
        if step == n_rounds:
            break
        experiences = list(experience_rounds[step])
        for name, state in states.items():
            features = predecision_features(state, experiences, env=env, capability=capability)
            view = DecisionView(
                env=env,
                round_index=step,
                features=features,
                state_hash=state.content_hash(),
                n_experiences=len(experiences),
            )
            action = selectors[name].select(view)
            cost = selectors[name].consume_cost() if hasattr(selectors[name], "consume_cost") else {}
            costs.append({"kind": "router_infer", "policy": name, "round": step, "action": action, **cost})
            try:
                applied = dict(apply_fn(state, action, experiences))
            except Exception as exc:  # noqa: BLE001
                failures.append({"policy": name, "round": step, "action": action, "error": f"{type(exc).__name__}: {exc}"})
                applied = {"changed": None, "accepted": None, "infra_status": "missing"}
            histories[name].append(
                {
                    "round": step,
                    "event": "update",
                    "action": action,
                    "changed": applied.get("changed"),
                    "accepted": applied.get("accepted"),
                    "state_hash_after": state.content_hash(),
                    "infra_status": applied.get("infra_status") or "ok",
                }
            )
    ood = list(trained_on_states) == ["S0"] and any(point > 0 for point in eval_points)
    summary = {
        "policies": list(selectors),
        "n_rounds": n_rounds,
        "eval_at": sorted(eval_points),
        "shared_acquisition": True,
        "future_fed_back": False,
        "trained_on_states": list(trained_on_states),
        "ood_state": ood,
        "claim_sequential": False if ood else True,
        "teacher_forced_gain_copied": False,
        "histories": histories,
        "comparisons": _comparisons(histories),
        "acceptance_rate": _acceptance(histories),
        "negative_transfer_rate": _negative_transfer(histories),
        "costs": costs,
        "failures": failures,
        "negative_transfer_definition": "fraction of evaluated intervals whose shadow success fell",
    }
    return summary


def _comparisons(histories: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    if "M" not in histories or "env-only" not in histories:
        return {}
    paired = []
    for left, right in zip(histories["M"], histories["env-only"], strict=False):
        if left.get("event") == "update" or right.get("event") == "update":
            continue
        if left.get("round") != right.get("round"):
            continue
        if left.get("future") is None or right.get("future") is None:
            continue
        paired.append(
            {
                "round": left["round"],
                "future_m": left["future"],
                "future_env_only": right["future"],
                "paired_diff": float(left["future"]) - float(right["future"]),
                "retention_m": left.get("retention"),
                "retention_env_only": right.get("retention"),
            }
        )
    return {"m_vs_env_only": paired}


def _acceptance(histories: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float | None]:
    rates = {}
    for name, events in histories.items():
        updates = [event for event in events if event.get("event") == "update"]
        if not updates:
            rates[name] = None
            continue
        accepted = sum(1 for event in updates if event.get("accepted") or event.get("changed"))
        rates[name] = accepted / len(updates)
    return rates


def _negative_transfer(histories: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float | None]:
    rates = {}
    for name, events in histories.items():
        points = [event for event in events if event.get("event") != "update" and event.get("shadow") is not None]
        if len(points) < 2:
            rates[name] = None
            continue
        drops = 0
        for prev, nxt in zip(points, points[1:], strict=False):
            if float(nxt["shadow"]) < float(prev["shadow"]):
                drops += 1
        rates[name] = drops / (len(points) - 1)
    return rates
