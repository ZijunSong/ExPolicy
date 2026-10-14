"""Harness run. It writes the D products and does not claim a real-environment result."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from expolicy.paper_d.artifacts import headroom_row, write_products
from expolicy.paper_d.audit import audit_measurement, valid_noop_bundle
from expolicy.paper_d.config import load_config
from expolicy.paper_d.constants import CONTROL_EXACT_TIE, CONTROL_OBSERVABLE, ESTIMATE_POLICY
from expolicy.paper_d.controls import build_control, evaluate_control, run_known_structure_controls
from expolicy.paper_d.decision import build_decision
from expolicy.paper_d.inventory import collect_artifact_inventory
from expolicy.paper_d.prospective import validate_prospectively


def run_self_check(output: Path | None = None) -> dict[str, Any]:
    config = load_config()
    controls = run_known_structure_controls(prefix="diag")
    inventory = collect_artifact_inventory(config.inventory_roots())
    audit = audit_measurement(valid_noop_bundle())
    prospective = _prospective(controls["reports"])
    decision = build_decision(
        {
            "controls_passed": controls["all_passed"],
            "n_real_configurations": 0,
            "inventory_verified": inventory["n_verified"],
            "audit": audit,
            "prospective": prospective,
            "prospective_passed": prospective["n_false_go"] == 0 and prospective["n_false_nogo"] == 0,
            "reasons": [] if controls["all_passed"] else ["known-structure control failed"],
        }
    )
    paths = {}
    if output is not None:
        paths = write_products(
            output,
            audit=audit,
            inventory=inventory,
            decision=decision,
            control_reports=controls["reports"],
            task_rows=[row for report in controls["reports"] for row in report["rows"]],
            headroom_rows=_headroom_rows(controls["reports"]),
            cards=[report["card"] for report in controls["reports"]],
            prospective_rows=_prospective_rows(prospective),
        )
    return {
        "decision": decision,
        "controls": controls,
        "audit": audit,
        "inventory": inventory,
        "prospective": prospective,
        "paths": paths,
    }


def _prospective(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cards = []
    outcomes = []
    for name in (CONTROL_OBSERVABLE, CONTROL_EXACT_TIE):
        diagnostic = next(report for report in reports if report["name"] == name)
        held = evaluate_control(build_control(name, prefix="hold"))
        cards.append(diagnostic["card"])
        outcomes.append(
            {
                "configuration_id": diagnostic["configuration_id"],
                "batch_ids": sorted({str(row["batch_id"]) for row in held["rows"]}),
                "task_ids": sorted({str(row["task_id"]) for row in held["rows"]}),
                "actual_gain": held["policy"]["feature_gain"],
                "diagnostic_cost": float(sum(float(row["cost"]) for row in diagnostic["rows"])),
                "training_cost": float(sum(float(row["cost"]) for row in held["rows"])),
                "training_avoided": 0.0,
            }
        )
    return validate_prospectively(cards, outcomes)


def _headroom_rows(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        for key in ("naive", "crossfit", "bound"):
            rows.append(headroom_row(str(report["configuration_id"]), report[key]))
        policy = dict(report["policy"])
        policy["estimate_name"] = ESTIMATE_POLICY
        policy["feature_gain"] = policy.get("feature_gain")
        rows.append(headroom_row(str(report["configuration_id"]), policy))
    return rows


def _prospective_rows(prospective: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in prospective["rows"]:
        copied = dict(row)
        copied["correlation_reported"] = prospective["correlation_reported"]
        copied["significance_claimed"] = prospective["significance_claimed"]
        copied["net_savings"] = prospective["net_savings"]
        rows.append(copied)
    return rows
