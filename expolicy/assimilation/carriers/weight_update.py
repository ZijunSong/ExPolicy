"""Real LoRA / QLoRA internalization from verified experiences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from expolicy.core.errors import BackendUnavailableError, TrainingDivergenceError
from expolicy.core.experience import Experience


def build_sft_rows(experiences: Sequence[Experience]) -> list[dict[str, str]]:
    rows = []
    for exp in experiences:
        if not exp.success:
            continue
        transcript = []
        for action, obs in zip(exp.actions, exp.observations, strict=False):
            transcript.append(f"Action: {action.get('tool')} {action.get('arguments')}")
            transcript.append(f"Observation: {str(obs)[:800]}")
        prompt = (
            f"Question: {exp.question}\n"
            f"Use search/open/find then answer.\n" + "\n".join(transcript)
        )
        completion = f"<tool>answer</tool>\n<arg>{exp.predicted_answer}</arg>"
        rows.append({"prompt": prompt, "completion": completion, "experience_id": exp.experience_id})
    return rows


class WeightUpdater:
    def __init__(self, base_model: str, output_dir: Path) -> None:
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        experiences: Sequence[Experience],
        *,
        steps: int = 30,
        lr: float = 1e-4,
        lora_rank: int = 8,
        max_seq_len: int = 1024,
    ) -> dict[str, Any]:
        rows = build_sft_rows(experiences)
        if not rows:
            raise TrainingDivergenceError("no verified successful experiences for LoRA SFT")
        try:
            import torch
            from peft import LoraConfig, TaskType, get_peft_model
            from torch.optim import AdamW
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendUnavailableError(
                "Weight internalization requires torch, transformers, and peft."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=_guess_targets(model),
        )
        model = get_peft_model(model, lora)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.train()
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        losses: list[float] = []
        for step in range(steps):
            row = rows[step % len(rows)]
            text = row["prompt"] + "\n" + row["completion"]
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            encoded["labels"] = encoded["input_ids"].clone()
            out = model(**encoded)
            loss = out.loss
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                raise TrainingDivergenceError(f"NaN/Inf loss at step {step}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        adapter_dir = self.output_dir / "lora"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        summary = {
            "n_rows": len(rows),
            "steps": steps,
            "mean_loss": sum(losses) / len(losses),
            "final_loss": losses[-1],
            "adapter_dir": str(adapter_dir),
        }
        (self.output_dir / "train_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def _guess_targets(model: Any) -> list[str]:
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    preferred = [x for x in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if x in names]
    if preferred:
        return preferred
    return ["q_proj", "v_proj"]
