"""Real LoRA / QLoRA internalization from verified successful trajectories."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from expolicy.core.errors import BackendUnavailableError, TrainingDivergenceError
from expolicy.core.experience import Experience

_DOC_ID = re.compile(r"\bdoc_\d+\b", re.I)


def _sanitize(text: str) -> str:
    return _DOC_ID.sub("[DOC]", text or "")


def _reasoning_state(exp: Experience) -> str:
    queries = []
    for call in exp.tool_calls:
        if call.get("tool") == "search":
            q = (call.get("arguments") or {}).get("query")
            if q:
                queries.append(_sanitize(str(q)))
    opened_titles = []
    for ev in exp.evidence:
        title = ev.get("title") or ev.get("source_type") or "opened_doc"
        opened_titles.append(str(title))
    tools = [str(c.get("tool")) for c in exp.tool_calls]
    return (
        f"capability={exp.capability}; tools={' -> '.join(tools)}\n"
        f"search_queries={queries}\n"
        f"opened_sources={opened_titles[:4]}"
    )


def build_sft_rows(experiences: Sequence[Experience]) -> list[dict[str, str]]:
    """Supervised rows from verified success only. No document-id shortcuts."""
    rows = []
    for exp in experiences:
        if not exp.success:
            continue
        evidence_snips = []
        for obs in exp.observations:
            if not isinstance(obs, dict):
                continue
            payload = obs.get("payload") if obs.get("tool") == "open" else None
            if isinstance(payload, dict) and payload.get("text"):
                src = payload.get("source_type") or "doc"
                evidence_snips.append(f"[{src}] {_sanitize(str(payload['text'])[:500])}")
        prompt = (
            "You are a search agent. Decompose, search, open evidence, then answer.\n"
            f"Task: {_sanitize(exp.question)}\n"
            f"State: {_reasoning_state(exp)}\n"
            "Verified evidence:\n"
            + ("\n".join(evidence_snips[:4]) or "(opened supporting documents)")
            + "\nEmit the next tool call, or the final short answer."
        )
        completion = f"<tool>answer</tool>\n<arg>{exp.predicted_answer}</arg>"
        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "experience_id": exp.experience_id,
                "task_id": exp.task_id,
            }
        )
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
        steps: int | None = None,
        epochs: int = 2,
        lr: float = 5e-5,
        lora_rank: int = 8,
        max_seq_len: int = 1024,
        batch_size: int = 1,
        early_stop_patience: int = 8,
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
        n_rows = len(rows)
        total_steps = steps if steps is not None else max(1, epochs * n_rows)
        losses: list[float] = []
        best = float("inf")
        stale = 0
        for step in range(total_steps):
            batch_rows = [rows[(step * batch_size + j) % n_rows] for j in range(batch_size)]
            texts = [row["prompt"] + "\n" + row["completion"] for row in batch_rows]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                padding=True,
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
            val = float(loss.detach().cpu())
            losses.append(val)
            if val + 1e-4 < best:
                best = val
                stale = 0
            else:
                stale += 1
            if stale >= early_stop_patience and step >= n_rows:
                break
        adapter_dir = self.output_dir / "lora"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        summary = {
            "n_rows": len(rows),
            "steps": len(losses),
            "epochs": epochs,
            "lr": lr,
            "lora_rank": lora_rank,
            "mean_loss": sum(losses) / len(losses),
            "final_loss": losses[-1],
            "best_loss": best,
            "adapter_dir": str(adapter_dir),
            "training_data_ids": [r["experience_id"] for r in rows],
        }
        (self.output_dir / "train_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def _guess_targets(model: Any) -> list[str]:
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    preferred = [x for x in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if x in names]
    if preferred:
        return preferred
    return ["q_proj", "v_proj"]
