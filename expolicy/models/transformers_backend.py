"""HuggingFace Transformers causal-LM backend."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from expolicy.core.errors import BackendUnavailableError
from expolicy.models.base import GenerationResult, ModelBackend


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise BackendUnavailableError(
            "Transformers backend requires torch and transformers. "
            "Install extras: pip install -e '.[train]' or use an env that already has them."
        ) from exc
    return torch, (AutoModelForCausalLM, AutoTokenizer)


class TransformersBackend(ModelBackend):
    backend_id = "transformers"

    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_input_tokens: int = 4096,
    ) -> None:
        torch, (AutoModelForCausalLM, AutoTokenizer) = _require_torch()
        path = Path(model_path)
        if not path.exists():
            raise BackendUnavailableError(
                f"model path does not exist: {model_path}. "
                "Set configs/model/*.yaml path or EXPOLICY_MODEL to a local checkpoint."
            )
        self.name = str(path)
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)
        if device == "cpu":
            torch_dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path),
            dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self._adapter_path: str | None = None

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        n: int = 1,
    ) -> GenerationResult:
        if n != 1:
            # real multi-sample: run n times and concatenate for self-consistency callers
            texts = []
            total = None
            for _ in range(n):
                total = self.generate(messages, max_new_tokens=max_new_tokens, temperature=max(temperature, 0.3), n=1)
                texts.append(total.text)
            assert total is not None
            total.raw["samples"] = texts
            total.text = texts[0]
            return total
        torch = self.torch
        prompt = self._render(list(messages))
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        started = time.perf_counter()
        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - started
        sequences = output.sequences
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = sequences[0, in_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        logprobs: list[float] = []
        if getattr(output, "scores", None):
            for step_scores, token_id in zip(output.scores, gen_ids.tolist(), strict=False):
                log_sm = torch.log_softmax(step_scores[0].float(), dim=-1)
                logprobs.append(float(log_sm[token_id].item()))
        prompt_tokens = int(in_len)
        completion_tokens = int(gen_ids.shape[0])
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=latency,
            logprobs=logprobs,
            stop_reason="eos",
        )

    def load(self, path: str) -> None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise BackendUnavailableError("peft is required to load a LoRA adapter") from exc
        adapter = Path(path)
        if not adapter.exists():
            raise BackendUnavailableError(f"adapter path missing: {path}")
        self.model = PeftModel.from_pretrained(self.model, str(adapter))
        self.model.to(self.device)
        self.model.eval()
        self._adapter_path = str(adapter)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _render(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        parts = []
        for msg in messages:
            parts.append(f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)
