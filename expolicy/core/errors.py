"""Structured errors. Fail loudly; never silently degrade into fake results."""

from __future__ import annotations


class ExPolicyError(Exception):
    """Base error for recoverable-but-explicit failures."""


class BackendUnavailableError(ExPolicyError):
    """Model, embedding, or tool backend is missing or failed.

    Recovery: set ``model.path`` / ``EXPOLICY_MODEL`` to a local instruct
    checkpoint, or start a vLLM OpenAI-compatible server and set
    ``model.backend: openai_compat``.
    """


class InvalidExperienceError(ExPolicyError):
    """Experience failed verification and must not enter weight update."""


class CheckpointError(ExPolicyError):
    """Checkpoint is missing, corrupt, or incompatible."""


class LeakageError(ExPolicyError):
    """Held-out answers or future-split labels leaked into a policy input."""


class ConfigError(ExPolicyError):
    """Configuration failed validation."""


class EnvironmentError(ExPolicyError):
    """Search environment or corpus is unusable."""


class TrainingDivergenceError(ExPolicyError):
    """Optimizer produced NaN/Inf or loss exploded."""


class HarnessPatchRejected(ExPolicyError):
    """Harness patch failed static validation, sandbox, or regression probes."""
