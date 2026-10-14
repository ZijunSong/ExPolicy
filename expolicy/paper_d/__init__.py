"""D-v1: diagnose whether an experience policy has realizable selection headroom."""

from expolicy.paper_d.audit import audit_measurement
from expolicy.paper_d.cards import freeze_diagnostic_card
from expolicy.paper_d.controls import run_known_structure_controls
from expolicy.paper_d.headroom import (
    bound_panel_headroom,
    crossfit_selection_gain,
    estimate_naive_gap,
)
from expolicy.paper_d.inventory import collect_artifact_inventory
from expolicy.paper_d.matrix import build_task_level_matrix
from expolicy.paper_d.policy import fit_simple_observable_policy
from expolicy.paper_d.prospective import validate_prospectively

__all__ = [
    "audit_measurement",
    "bound_panel_headroom",
    "build_task_level_matrix",
    "collect_artifact_inventory",
    "crossfit_selection_gain",
    "estimate_naive_gap",
    "fit_simple_observable_policy",
    "freeze_diagnostic_card",
    "run_known_structure_controls",
    "validate_prospectively",
]
