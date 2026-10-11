"""Acquisition method must not change the assimilator. Shared SkillOpt instance protocol."""

from __future__ import annotations

from expolicy.acquisition.base import make_acquisition
from expolicy.r7.skillopt import OPERATORS, SkillOptAssimilator, COMPILER_SKILLOPT
from expolicy.r7.constants import EDIT_BUDGET


def test_shared_assimilator_constants() -> None:
    a = SkillOptAssimilator()
    b = SkillOptAssimilator()
    assert a.edit_budget == b.edit_budget == EDIT_BUDGET
    assert set(OPERATORS) == {"add_step", "delete_step", "replace_query_suffix", "replace_recovery", "adjust_max_opens"}
    assert COMPILER_SKILLOPT.startswith("r7.skillopt")


def test_acquisition_factory_does_not_bind_assimilator() -> None:
    src = __import__("inspect").getsource(make_acquisition)
    assert "SkillOpt" not in src
    assert "assimilator" not in src.lower()
    for name in ("passive", "frontier", "coverage_balanced", "ace_skill"):
        pol = make_acquisition(name, seed=0)
        assert not hasattr(pol, "assimilator") or pol.assimilator is None
