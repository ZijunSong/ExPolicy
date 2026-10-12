"""Frozen runtime registry: tools, features, templates, budget, and hash."""

from __future__ import annotations

from dataclasses import dataclass

from expolicy.paper_b.constants import (
    ESI_OPERATORS,
    FALLBACK_BASE,
    FAMILY_COMPARE,
    FAMILY_RECOVERY,
    LEARNED_REWRITE,
    OBSERVABLE_FEATURES,
    REGISTERED_TEMPLATES,
    REWRITE_TEMPLATES,
    TEMPLATE_REWRITE,
    TEMPLATE_SEARCH,
    TRIGGER_FEATURES,
    TRIGGER_OPS,
)
from expolicy.paper_b.models import Budget, Rule
from expolicy.r9_lab.io_util import sha256_json


def _rule_id(prefix: str, template: str, trigger: dict) -> str:
    digest = sha256_json({"template": template, "trigger": trigger})[:10]
    return f"{prefix}_{template}_{digest}"


def generic_rules() -> list[Rule]:
    """B1 / teacher prior. Fixed before any experience is read."""
    trigger = {
        "all": [
            {"feature": "has_comparison_cue", "op": "eq", "value": 1},
            {"feature": "n_entities", "op": "gte", "value": 2},
            {"feature": "evidence_filled", "op": "lte", "value": 1},
        ]
    }
    return [
        Rule(
            rule_id=_rule_id("b1", TEMPLATE_SEARCH, trigger),
            family=FAMILY_COMPARE,
            trigger=trigger,
            action_template=TEMPLATE_SEARCH,
            parameters={},
            fallback=FALLBACK_BASE,
            validation_record="frozen_generic",
        )
    ]


@dataclass(frozen=True)
class RuntimeRegistry:
    features: tuple[str, ...]
    trigger_features: tuple[str, ...]
    ops: tuple[str, ...]
    templates: tuple[str, ...]
    rewrite_templates: tuple[str, ...]
    budget: Budget
    max_atoms: int
    max_rules: int
    max_rules_per_family: int
    max_candidates_per_batch: int
    max_revisions: int
    harness: str

    def catalog(self) -> dict:
        return {
            "features": list(self.trigger_features),
            "ops": list(self.ops),
            "templates": [
                {
                    "name": name,
                    "esi_operators": list(ESI_OPERATORS.get(name, ())),
                    "parameters": ["rewrite_template"] if name == TEMPLATE_REWRITE else [],
                }
                for name in self.templates
            ],
            "rewrite_templates": list(self.rewrite_templates),
            "learned_rewrite_in_prior": LEARNED_REWRITE in self.prior_rewrite_templates(),
        }

    def prior_rewrite_templates(self) -> tuple[str, ...]:
        """The empty-experience prior does not contain the experience-specific rewrite."""
        return tuple(item for item in self.rewrite_templates if item != LEARNED_REWRITE)

    def template_family(self, template: str) -> str:
        if template == TEMPLATE_REWRITE:
            return FAMILY_RECOVERY
        return FAMILY_COMPARE

    def to_dict(self) -> dict:
        return {
            "harness": self.harness,
            "features": list(self.features),
            "trigger_features": list(self.trigger_features),
            "ops": list(self.ops),
            "templates": list(self.templates),
            "esi_operators": {name: list(ESI_OPERATORS.get(name, ())) for name in self.templates},
            "rewrite_templates": list(self.rewrite_templates),
            "budget": self.budget.to_dict(),
            "max_atoms": self.max_atoms,
            "max_rules": self.max_rules,
            "max_rules_per_family": self.max_rules_per_family,
            "max_candidates_per_batch": self.max_candidates_per_batch,
            "max_revisions": self.max_revisions,
            "permissions": {
                "new_python": False,
                "new_tools": False,
                "memorized_answers": False,
                "tools": ["search", "store_evidence", "compare", "rewrite_query", "legal_action_check"],
            },
        }

    def hash(self) -> str:
        return sha256_json(self.to_dict())


def build_runtime_registry(
    *,
    budget: Budget | None = None,
    max_atoms: int = 3,
    max_rules: int = 8,
    max_rules_per_family: int = 2,
    max_candidates_per_batch: int = 4,
    max_revisions: int = 1,
    harness: str = "local_search",
) -> RuntimeRegistry:
    return RuntimeRegistry(
        features=OBSERVABLE_FEATURES,
        trigger_features=TRIGGER_FEATURES,
        ops=TRIGGER_OPS,
        templates=REGISTERED_TEMPLATES,
        rewrite_templates=REWRITE_TEMPLATES,
        budget=budget or Budget(),
        max_atoms=max_atoms,
        max_rules=max_rules,
        max_rules_per_family=max_rules_per_family,
        max_candidates_per_batch=max_candidates_per_batch,
        max_revisions=max_revisions,
        harness=harness,
    )
