"""C-v1: predict post-assimilation gain before spending the acquisition budget."""

from expolicy.paper_c.audit import audit_assimilator, build_gain_audit
from expolicy.paper_c.curves import evaluate_frozen_curves
from expolicy.paper_c.features import pre_acquisition_features
from expolicy.paper_c.pilot import run_self_check
from expolicy.paper_c.predictor import fit_gain_predictor
from expolicy.paper_c.sequence import run_learning_sequence

__all__ = [
    "audit_assimilator",
    "build_gain_audit",
    "evaluate_frozen_curves",
    "fit_gain_predictor",
    "pre_acquisition_features",
    "run_learning_sequence",
    "run_self_check",
]
