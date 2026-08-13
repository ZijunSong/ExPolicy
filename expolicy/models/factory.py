"""Construct a ModelBackend from config. Never silently fall back to a fake model."""

from __future__ import annotations

import os
from typing import Any

from expolicy.core.errors import BackendUnavailableError, ConfigError
from expolicy.models.base import ModelBackend


def build_backend(cfg: dict[str, Any]) -> ModelBackend:
    backend = str(cfg.get("backend") or "transformers").lower()
    model_path = str(cfg.get("path") or os.environ.get("EXPOLICY_MODEL") or "")
    if backend == "transformers":
        from expolicy.models.transformers_backend import TransformersBackend

        if not model_path:
            raise ConfigError("model.path is required for transformers backend")
        return TransformersBackend(
            model_path,
            device=cfg.get("device"),
            dtype=str(cfg.get("dtype") or "bfloat16"),
        )
    if backend == "vllm":
        from expolicy.models.vllm_backend import VLLMBackend

        if not model_path:
            raise ConfigError("model.path is required for vllm backend")
        return VLLMBackend(
            model_path,
            tensor_parallel_size=int(cfg.get("tensor_parallel_size") or 1),
            max_model_len=int(cfg.get("max_model_len") or 4096),
        )
    if backend in {"openai_compat", "openai", "vllm_http"}:
        from expolicy.models.openai_compat import OpenAICompatBackend

        base_url = str(cfg.get("base_url") or os.environ.get("EXPOLICY_API_BASE") or "")
        name = str(cfg.get("name") or model_path or os.environ.get("EXPOLICY_MODEL") or "")
        if not base_url or not name:
            raise ConfigError("openai_compat backend requires model.base_url and model.name")
        return OpenAICompatBackend(
            name,
            base_url=base_url,
            api_key=str(cfg.get("api_key") or os.environ.get("EXPOLICY_API_KEY") or "EMPTY"),
        )
    raise BackendUnavailableError(
        f"unknown model backend {backend!r}. Use transformers | vllm | openai_compat."
    )
