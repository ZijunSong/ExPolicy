"""A-v1: predict which persistent assimilation to apply before writing experience."""

from expolicy.paper_a.audit import audit_noop
from expolicy.paper_a.crossfit import crossfit_action_selection
from expolicy.paper_a.groups import split_by_source_group
from expolicy.paper_a.matrix import build_utility_matrix
from expolicy.paper_a.router import fit_ridge_router
from expolicy.paper_a.sequence import run_fixed_acquisition_sequence

__all__ = [
    "audit_noop",
    "build_utility_matrix",
    "crossfit_action_selection",
    "fit_ridge_router",
    "run_fixed_acquisition_sequence",
    "split_by_source_group",
]
