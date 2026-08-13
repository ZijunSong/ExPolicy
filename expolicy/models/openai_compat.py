"""OpenAI-compatible HTTP backend (vLLM / local server)."""

from __future__ import annotations

import time
from typing import Sequence

import httpx

from expolicy.core.errors import BackendUnavailableError
from expolicy.models.base import GenerationResult, ModelBackend


class OpenAICompatBackend(ModelBackend):
    backend_id = "openai_compat"

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ) -> None:
        self.name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        n: int = 1,
    ) -> GenerationResult:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.name,
            "messages": list(messages),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "n": n,
            "logprobs": True,
        }
        started = time.perf_counter()
        try:
            response = httpx.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            raise BackendUnavailableError(
                f"OpenAI-compatible backend failed at {url}: {exc}. "
                "Start vLLM (`vllm serve <model>`) or fix model.base_url."
            ) from exc
        latency = time.perf_counter() - started
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise BackendUnavailableError(f"empty choices from {url}: {data}")
        text = (choices[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        logprobs: list[float] = []
        lp = choices[0].get("logprobs") or {}
        content = lp.get("content") or []
        for item in content:
            if item.get("logprob") is not None:
                logprobs.append(float(item["logprob"]))
        return GenerationResult(
            text=text.strip(),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency=latency,
            logprobs=logprobs,
            stop_reason=str(choices[0].get("finish_reason") or "stop"),
            raw={"id": data.get("id")},
        )
