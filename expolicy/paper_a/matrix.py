"""Utility matrix. The cache key is the full config hash. Failures stay missing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from expolicy.paper_a.constants import ACTIONS, PROTOCOL_ID, RETENTION, SHADOW, STATUS_MISSING, STATUS_OK
from expolicy.paper_a.features import template_key
from expolicy.paper_a.schema import empty_row
from expolicy.r9_lab.evaluation.utility import compute_utility
from expolicy.r9_lab.io_util import read_json, sha256_json, write_json


CellEvaluator = Callable[[Mapping[str, Any], str, Sequence[Mapping[str, Any]]], Mapping[str, Any]]


def to_router_records(
    rows: Sequence[Mapping[str, Any]],
    features_by_batch: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One record per batch. A missing action utility stays None and is not replaced with 0."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("legacy"):
            continue
        batch_id = str(row.get("batch_id"))
        slot = grouped.setdefault(
            batch_id,
            {
                "batch_id": batch_id,
                "env": row.get("env"),
                "source_group": row.get("source_group"),
                "U": {},
                "delta_shadow": {},
                "delta_retention": {},
            },
        )
        action = str(row.get("action"))
        slot["U"][action] = row.get("utility")
        slot["delta_shadow"][action] = row.get("delta_shadow")
        slot["delta_retention"][action] = row.get("delta_retention")
    records = []
    for batch_id, slot in grouped.items():
        features = features_by_batch.get(batch_id)
        if not features:
            continue
        slot["features"] = dict(features)
        records.append(slot)
    return records


def source_group_for_batch(batch: Mapping[str, Any]) -> str:
    if batch.get("source_group"):
        return str(batch["source_group"])
    items = list(batch.get("items") or [])
    if not items:
        items = [{"task_id": tid, "env": batch.get("env"), "instruction": str(tid)} for tid in batch.get("task_ids") or []]
    keys = sorted({template_key(item) for item in items})
    return sha256_json(keys)[:16]


def _cell_path(cache_dir: Path, config_hash: str, batch_id: str, action: str) -> Path:
    return cache_dir / config_hash / batch_id / action / "cell.json"


def _load_cached(path: Path, config_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    if payload.get("config_hash") != config_hash or payload.get("complete") is not True:
        return None
    return payload


def _legacy_hits(cache_dir: Path, config_hash: str, batch_id: str, action: str) -> list[str]:
    hits = []
    if not cache_dir.exists():
        return hits
    for child in cache_dir.iterdir():
        if not child.is_dir() or child.name == config_hash:
            continue
        candidate = child / batch_id / action / "cell.json"
        if candidate.exists():
            hits.append(str(candidate))
    return hits


def summarize_cell(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cell-level U from paired items. Any missing probe nulls U instead of imputing zero."""
    if not items:
        return {"infra_status": STATUS_MISSING, "delta_shadow": None, "delta_retention": None, "utility": None, "retention_penalty": None}
    if any(str(item.get("infra_status") or STATUS_OK) != STATUS_OK or item.get("y_before") is None or item.get("y_after") is None for item in items):
        return {
            "infra_status": STATUS_MISSING,
            "delta_shadow": None,
            "delta_retention": None,
            "utility": None,
            "retention_penalty": None,
            "reason": "incomplete_probe",
        }

    def _mean(split: str, key: str) -> float | None:
        vals = [float(item[key]) for item in items if item.get("eval_split") == split]
        if not vals:
            return None
        return sum(vals) / len(vals)

    shadow_before = _mean(SHADOW, "y_before")
    shadow_after = _mean(SHADOW, "y_after")
    retention_before = _mean(RETENTION, "y_before")
    retention_after = _mean(RETENTION, "y_after")
    if None in (shadow_before, shadow_after, retention_before, retention_after):
        return {
            "infra_status": STATUS_MISSING,
            "delta_shadow": None,
            "delta_retention": None,
            "utility": None,
            "retention_penalty": None,
            "reason": "missing_split",
        }
    util = compute_utility(
        shadow_before=float(shadow_before),
        shadow_after=float(shadow_after),
        retention_before=float(retention_before),
        retention_after=float(retention_after),
    )
    return {
        "infra_status": STATUS_OK,
        "delta_shadow": util["d_shadow"],
        "delta_retention": util["d_retention"],
        "utility": util["U"],
        "retention_penalty": util["retention_penalty"],
        "reason": "",
    }


def _rows_for_cell(
    *,
    batch: Mapping[str, Any],
    action: str,
    payload: Mapping[str, Any],
    model_hash: str,
    config_hash: str,
    protocol_id: str,
) -> list[dict[str, Any]]:
    summary = summarize_cell(list(payload.get("items") or []))
    group = str(payload.get("source_group") or source_group_for_batch(batch))
    rows = []
    items = list(payload.get("items") or [])
    if not items:
        rows.append(
            empty_row(
                protocol_id=protocol_id,
                model_hash=model_hash,
                env=batch.get("env"),
                source_group=group,
                batch_id=batch.get("batch_id"),
                action=action,
                infra_status=STATUS_MISSING,
                config_hash=config_hash,
                cache_key=config_hash,
            )
        )
        return rows
    for item in items:
        status = summary["infra_status"] if summary["infra_status"] != STATUS_OK else str(item.get("infra_status") or STATUS_OK)
        if summary["infra_status"] != STATUS_OK and str(item.get("infra_status") or STATUS_OK) == STATUS_OK:
            status = STATUS_MISSING
        rows.append(
            empty_row(
                protocol_id=protocol_id,
                model_hash=model_hash,
                env=batch.get("env"),
                source_group=group,
                batch_id=batch.get("batch_id"),
                state_hash_before=payload.get("state_hash_before"),
                action=action,
                state_hash_after=payload.get("state_hash_after"),
                changed=payload.get("changed"),
                accepted=payload.get("accepted"),
                probe_task_id=item.get("probe_task_id"),
                eval_split=item.get("eval_split"),
                y_before=item.get("y_before"),
                y_after=item.get("y_after"),
                delta_shadow=summary["delta_shadow"],
                delta_retention=summary["delta_retention"],
                utility=summary["utility"],
                tokens=item.get("tokens"),
                tool_calls=item.get("tool_calls"),
                elapsed_s=item.get("elapsed_s"),
                infra_status=status,
                retention_penalty=summary["retention_penalty"],
                prompt_hash=item.get("prompt_hash"),
                cache_key=config_hash,
                config_hash=config_hash,
                legacy=False,
            )
        )
    return rows


def build_utility_matrix(
    batches: Sequence[Mapping[str, Any]],
    actions: Sequence[str],
    probes_by_env: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluate_cell: CellEvaluator,
    *,
    cache_dir: Path,
    config_hash: str,
    model_hash: str,
    protocol_id: str = PROTOCOL_ID,
) -> dict[str, Any]:
    """Evaluate each batch/action pair. A cache hit requires the same config hash.

    Cells written under any other hash are listed as legacy and are not copied in.
    An exception, or a probe without a score, becomes infra_status=missing and a null utility.
    """
    unknown = [action for action in actions if action not in ACTIONS]
    if unknown:
        raise ValueError(f"actions outside A-v1 are refused: {unknown}")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    legacy: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    features: dict[str, Any] = {}
    for batch in batches:
        env = str(batch["env"])
        probes = list(probes_by_env.get(env) or [])
        for action in actions:
            path = _cell_path(cache_dir, config_hash, str(batch["batch_id"]), action)
            for hit in _legacy_hits(cache_dir, config_hash, str(batch["batch_id"]), action):
                legacy.append({"batch_id": batch["batch_id"], "action": action, "path": hit, "spliced": False})
            cached = _load_cached(path, config_hash)
            if cached is not None:
                payload = cached
            else:
                try:
                    produced = dict(evaluate_cell(batch, action, probes))
                    produced["config_hash"] = config_hash
                    produced["complete"] = True
                    produced.setdefault("source_group", source_group_for_batch(batch))
                    write_json(path, produced)
                    payload = produced
                except Exception as exc:  # noqa: BLE001
                    payload = {
                        "config_hash": config_hash,
                        "complete": False,
                        "source_group": source_group_for_batch(batch),
                        "state_hash_before": None,
                        "state_hash_after": None,
                        "changed": None,
                        "accepted": None,
                        "items": [
                            {
                                "probe_task_id": probe.get("probe_task_id"),
                                "eval_split": probe.get("eval_split"),
                                "y_before": None,
                                "y_after": None,
                                "tokens": None,
                                "tool_calls": None,
                                "elapsed_s": None,
                                "infra_status": STATUS_MISSING,
                                "prompt_hash": None,
                            }
                            for probe in probes
                        ],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    write_json(path, payload)
                    failures.append(
                        {
                            "batch_id": batch["batch_id"],
                            "action": action,
                            "infra_status": STATUS_MISSING,
                            "error": payload["error"],
                        }
                    )
            if payload.get("features"):
                features[str(batch["batch_id"])] = payload["features"]
            cell_rows = _rows_for_cell(
                batch=batch,
                action=action,
                payload=payload,
                model_hash=model_hash,
                config_hash=config_hash,
                protocol_id=protocol_id,
            )
            rows.extend(cell_rows)
            if any(row["infra_status"] != STATUS_OK for row in cell_rows):
                failures.append(
                    {
                        "batch_id": batch["batch_id"],
                        "action": action,
                        "infra_status": STATUS_MISSING,
                        "error": payload.get("error") or "incomplete_probe",
                    }
                )
            costs.append(
                {
                    "kind": "cell",
                    "batch_id": batch["batch_id"],
                    "action": action,
                    "tokens": payload.get("operator_tokens"),
                    "elapsed_s": payload.get("operator_elapsed_s"),
                    "config_hash": config_hash,
                }
            )
    return {
        "rows": rows,
        "failures": failures,
        "legacy": legacy,
        "costs": costs,
        "features": features,
        "config_hash": config_hash,
    }
