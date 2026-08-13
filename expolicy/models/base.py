"""Model backend protocol. Core experiments must run on open models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from expolicy.core.errors import BackendUnavailableError
from expolicy.core.experience import ModelCallRecord
from expolicy.core.types import JSONDict


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    logprobs: list[float] = field(default_factory=list)
    stop_reason: str = "stop"
    raw: JSONDict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def entropy(self) -> float | None:
        if not self.logprobs:
            return None
        import math

        # token NLL mean as an uncertainty proxy from real model logprobs
        nll = [-lp for lp in self.logprobs]
        return float(sum(nll) / len(nll))


class ModelBackend(ABC):
    name: str
    backend_id: str

    @abstractmethod
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        n: int = 1,
    ) -> GenerationResult: ...

    def generate_batch(
        self,
        batch_messages: Sequence[Sequence[dict[str, str]]],
        **kwargs: Any,
    ) -> list[GenerationResult]:
        return [self.generate(messages, **kwargs) for messages in batch_messages]

    def score(self, text: str, continuation: str) -> float:
        raise BackendUnavailableError(f"{self.backend_id} does not implement score()")

    def logprobs(self, messages: Sequence[dict[str, str]]) -> list[float]:
        result = self.generate(messages, max_new_tokens=1, temperature=0.0)
        return result.logprobs

    def save(self, path: str) -> None:
        raise BackendUnavailableError(f"{self.backend_id} does not implement save()")

    def load(self, path: str) -> None:
        raise BackendUnavailableError(f"{self.backend_id} does not implement load()")

    def to_record(
        self,
        messages: Sequence[dict[str, str]],
        result: GenerationResult,
    ) -> ModelCallRecord:
        return ModelCallRecord(
            backend=self.backend_id,
            model_name=self.name,
            messages=list(messages),
            output_text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency=result.latency,
            logprobs_summary={
                "n": len(result.logprobs),
                "mean": (sum(result.logprobs) / len(result.logprobs)) if result.logprobs else None,
            },
            stop_reason=result.stop_reason,
        )
