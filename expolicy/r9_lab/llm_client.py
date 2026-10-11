"""OpenAI-compatible vLLM chat.completions client for R9 frozen actors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin

import httpx

from expolicy.r9_lab.constants import (
    TEMPERATURE,
    TOP_P,
    VLLM_PORT,
    ModelSpec,
    model_by_id,
)


@dataclass
class ChatResult:
    text: str
    content: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


def extra_body_for(spec: ModelSpec) -> dict[str, Any] | None:
    """Q0: no thinking flag. Q1: do not disable reasoning. Q2/Q3: enable_thinking=False."""
    if spec.extra_thinking_kwarg:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


class VLLMClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        port: int = VLLM_PORT,
        model: str,
        spec: ModelSpec | None = None,
        timeout: float = 180.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or f"http://127.0.0.1:{port}/v1").rstrip("/")
        self.model = model
        self.spec = spec
        self.timeout = timeout
        self._http = http or httpx.Client(timeout=timeout)

    @classmethod
    def from_model_id(cls, model_id: str, **kwargs: Any) -> "VLLMClient":
        spec = model_by_id(model_id)
        return cls(model=spec.served_name, spec=spec, **kwargs)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        max_tokens: int = 1024,
        extra_body: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = int(seed)
        computed = extra_body if extra_body is not None else (extra_body_for(self.spec) if self.spec else None)
        if computed:
            body["extra_body"] = computed
            # vLLM OpenAI server also accepts chat_template_kwargs at top level via extra_body.
            body.update(computed)
        url = self.base_url + "/chat/completions"
        try:
            resp = self._http.post(url, json=body)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"vLLM chat.completions failed: {exc}") from exc
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
        usage = payload.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(
            completion_details.get("reasoning_tokens")
            or usage.get("reasoning_tokens")
            or 0
        )
        return ChatResult(
            text=content or reasoning,
            content=content,
            reasoning=reasoning,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=reasoning_tokens,
            total_tokens=int(usage.get("total_tokens") or 0),
            raw=payload,
        )


def probe_vllm(port: int, *, timeout: float = 5.0) -> None:
    url = f"http://127.0.0.1:{int(port)}/v1/models"
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"vLLM not reachable at {url}: {exc}") from exc


class FakeLLMClient:
    """Deterministic client for tests. Echoes a valid env-specific finish action."""

    def __init__(self, *, text: str | None = None, env: str = "db", success_token: str = "OK"):
        self.text = text
        self.env = env
        self.success_token = success_token
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatResult:
        self.calls.append([dict(m) for m in messages])
        if self.text is not None:
            body = self.text
        elif self.env == "os":
            body = f"Act: finish [{self.success_token}]"
        elif self.env == "kg":
            body = f"Final Answer: #0 {self.success_token}"
        else:
            body = f"Action: Answer\nFinal Answer: {self.success_token}"
        return ChatResult(text=body, content=body, completion_tokens=8, total_tokens=16)
