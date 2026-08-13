"""Optional vLLM in-process backend."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from expolicy.core.errors import BackendUnavailableError
from expolicy.models.base import GenerationResult, ModelBackend


class VLLMBackend(ModelBackend):
    backend_id = "vllm"

    def __init__(self, model_path: str, *, tensor_parallel_size: int = 1, max_model_len: int = 4096) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise BackendUnavailableError(
                "vLLM is not installed. pip install -e '.[serve]' or use backend=transformers."
            ) from exc
        if not Path(model_path).exists():
            raise BackendUnavailableError(f"vLLM model path missing: {model_path}")
        self.name = model_path
        self._LLM = LLM
        self._SamplingParams = SamplingParams
        self.engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        n: int = 1,
    ) -> GenerationResult:
        tokenizer = self.engine.get_tokenizer()
        prompt = tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
        params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            n=n,
            logprobs=1,
        )
        started = time.perf_counter()
        outputs = self.engine.generate([prompt], params)
        latency = time.perf_counter() - started
        if not outputs or not outputs[0].outputs:
            raise BackendUnavailableError("vLLM returned no outputs")
        out = outputs[0].outputs[0]
        logprobs: list[float] = []
        if out.logprobs:
            for step in out.logprobs:
                if not step:
                    continue
                first = next(iter(step.values()))
                logprobs.append(float(getattr(first, "logprob", 0.0)))
        return GenerationResult(
            text=(out.text or "").strip(),
            prompt_tokens=len(outputs[0].prompt_token_ids or []),
            completion_tokens=len(out.token_ids or []),
            latency=latency,
            logprobs=logprobs,
        )
