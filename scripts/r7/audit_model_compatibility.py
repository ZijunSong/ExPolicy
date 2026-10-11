#!/usr/bin/env python3
"""Real-generation compatibility probe for local causal LMs. No silent quantization."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from expolicy.experiments.common import ROOT, write_json
from expolicy.r7.constants import ALL_PROBE_MODELS, EMBEDDING_ONLY, MODEL_ROOT, R7_ROOT, RETRIEVER_URL


def inspect_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    tok_path = model_dir / "tokenizer_config.json"
    gen_path = model_dir / "generation_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    tok = json.loads(tok_path.read_text()) if tok_path.exists() else {}
    gen = json.loads(gen_path.read_text()) if gen_path.exists() else {}
    arch = cfg.get("architectures") or []
    model_type = cfg.get("model_type")
    is_causal = any("CausalLM" in str(a) or "ForCausalLM" in str(a) for a in arch)
    # Gemma4ForConditionalGeneration is a causal decoder in practice
    if any("ForConditionalGeneration" in str(a) for a in arch):
        is_causal = True
    thinking = bool(
        "thinking" in model_dir.name.lower()
        or "reasoning" in model_dir.name.lower()
        or gen.get("thinking")
        or cfg.get("thinking_mode")
    )
    return {
        "path": str(model_dir),
        "exists": model_dir.exists(),
        "architectures": arch,
        "model_type": model_type,
        "is_causal_lm": is_causal,
        "torch_dtype": cfg.get("torch_dtype"),
        "num_parameters_config": cfg.get("num_parameters"),
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "vocab_size": cfg.get("vocab_size"),
        "max_position_embeddings": cfg.get("max_position_embeddings") or cfg.get("max_sequence_length"),
        "tokenizer_class": tok.get("tokenizer_class"),
        "chat_template_present": bool(tok.get("chat_template")) or (model_dir / "chat_template.jinja").exists() or (model_dir / "tokenizer_config.json").exists(),
        "generation_config": {k: gen.get(k) for k in ("temperature", "top_p", "do_sample", "max_new_tokens") if k in gen},
        "thinking_mode": thinking,
        "quantization_config": cfg.get("quantization_config"),
        "moe": "moe" in str(model_type).lower() or "Moe" in "".join(str(a) for a in arch),
    }


SMOKE_QUESTIONS = [
    "What is the capital of France?",
    "Who wrote Pride and Prejudice?",
    "Where is the Eiffel Tower?",
    "Who was the first president of the United States?",
    "What year did World War II end?",
    "Which planet is known as the Red Planet?",
    "Who painted the Mona Lisa?",
    "What is the largest ocean on Earth?",
]


def smoke_one(model_path: str, gpu: str, retriever_url: str, n: int, out: Path) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    rec: dict = {
        "model_path": model_path,
        "gpu": gpu,
        "ok": False,
        "error": None,
        "peak_mem_gb": None,
        "need_tp": False,
        "vllm_loadable_guess": True,
        "smoke": [],
        "tool_call_compatible": None,
        "dtype": "bfloat16",
        "quantized": False,
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token_id is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        rec["tokenizer"] = type(tok).__name__
        rec["chat_template"] = bool(getattr(tok, "chat_template", None))
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="cuda:0",
        )
        model.eval()
        rec["peak_mem_gb_after_load"] = float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
        from expolicy.environments.public_search.retriever import RetrieverClient
        from expolicy.models.transformers_backend import TransformersBackend

        # wrap already-loaded? TransformersBackend reloads. Use generate via backend for protocol fidelity.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        backend = TransformersBackend(model_path, dtype="bfloat16")
        client = RetrieverClient(retriever_url)
        n_closed = 0
        for q in SMOKE_QUESTIONS[:n]:
            item = {"question": q, "ok": False}
            try:
                hits = client.search(q, topk=3)
                item["n_hits"] = len(hits)
                messages = [
                    {
                        "role": "user",
                        "content": (
                            "You are a Wikipedia search agent. Output exactly one XML tool call.\n"
                            "<tool>search</tool>\n<arg>query</arg>\n"
                            "or <tool>answer</tool>\n<arg>short span</arg>\n"
                            f"Question: {q}"
                        ),
                    }
                ]
                gen = backend.generate(messages, max_new_tokens=64, temperature=0.0)
                item["text"] = (gen.text or "")[:400]
                item["completion_tokens"] = gen.completion_tokens
                text = (gen.text or "").lower()
                item["tool_xml"] = "<tool>" in text
                if hits:
                    ev = (hits[0].get("title") or "") + "\n" + (hits[0].get("text") or "")[:400]
                    syn = backend.generate(
                        [{"role": "user", "content": f"Answer briefly using evidence.\nQuestion: {q}\nEvidence:\n{ev}"}],
                        max_new_tokens=32,
                        temperature=0.0,
                    )
                    item["answer"] = (syn.text or "")[:200]
                    item["answer_tokens"] = syn.completion_tokens
                item["ok"] = bool(item.get("n_hits")) and int(gen.completion_tokens or 0) > 0
                if item["ok"]:
                    n_closed += 1
            except Exception as exc:
                item["error"] = repr(exc)
            rec["smoke"].append(item)
            write_json(out / "STATUS_LIVE.json", {"updated_at": time.time(), "pid": os.getpid(), "last": q})
        rec["n_closed_loop"] = n_closed
        rec["tool_call_compatible"] = sum(1 for s in rec["smoke"] if s.get("tool_xml")) >= max(1, n // 4)
        rec["peak_mem_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else rec.get("peak_mem_gb_after_load")
        rec["ok"] = n_closed >= max(1, n // 2)
        rec["need_tp"] = bool(rec["peak_mem_gb"] and rec["peak_mem_gb"] > 120)
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[name-defined]
        rec["error"] = f"OOM:{exc}"
        rec["need_tp"] = True
        rec["ok"] = False
    except Exception as exc:
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()[-2000:]
        rec["ok"] = False
    rec["wall_seconds"] = time.time() - started
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", default=str(MODEL_ROOT))
    p.add_argument("--models", nargs="+", default=ALL_PROBE_MODELS)
    p.add_argument("--real-generation-smoke", type=int, default=8)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "4")
    p.add_argument("--model", default=None, help="run a single model (worker mode)")
    p.add_argument("--retriever-url", default=os.environ.get("EXPOLICY_RETRIEVER_URL", RETRIEVER_URL))
    p.add_argument("--out", default=str(R7_ROOT / "audit" / "model_compatibility"))
    p.add_argument("--offline-only", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.model_root)

    static = []
    for name in args.models:
        if name in EMBEDDING_ONLY:
            continue
        info = inspect_config(root / name)
        info["name"] = name
        info["enter_policy_table"] = bool(info.get("is_causal_lm"))
        if name == "harness-1" and not info.get("is_causal_lm"):
            info["enter_policy_table"] = False
            info["note"] = "harness-1 is not a loadable causal LM; excluded from primary table"
        static.append(info)
        write_json(out / f"{name}.config.json", info)

    write_json(out / "STATIC.json", static)
    md = ["# MODEL_COMPATIBILITY\n", "Embedding-only checkpoints are excluded from the policy backbone table.\n\n"]
    md.append("| model | arch | causal | thinking | enter_table |\n|---|---|---|---|---|\n")
    for info in static:
        md.append(
            f"| {info['name']} | {info.get('architectures')} | {info.get('is_causal_lm')} | {info.get('thinking_mode')} | {info.get('enter_policy_table')} |\n"
        )

    if args.offline_only and not args.model:
        (out / "MODEL_COMPATIBILITY.md").write_text("".join(md) + "\n(offline config inspection only)\n", encoding="utf-8")
        print("offline static inspection done", flush=True)
        return

    targets = [args.model] if args.model else [s["name"] for s in static if s.get("enter_policy_table")]
    for name in targets:
        done = out / f"{name}.smoke.json"
        if done.exists() and json.loads(done.read_text()).get("ok") is not None:
            print(f"skip {name} (exists)", flush=True)
            rec = json.loads(done.read_text())
        else:
            print(f"smoke {name} gpu={args.gpu}", flush=True)
            rec = smoke_one(str(root / name), str(args.gpu).split(",")[0], args.retriever_url, args.real_generation_smoke, out)
            write_json(done, rec)
        md.append(f"\n## {name}\n\n")
        md.append(f"- ok: {rec.get('ok')}\n- peak_mem_gb: {rec.get('peak_mem_gb')}\n- need_tp: {rec.get('need_tp')}\n- error: {rec.get('error')}\n- n_closed_loop: {rec.get('n_closed_loop')}\n- tool_call_compatible: {rec.get('tool_call_compatible')}\n- quantized: false (quantization is forbidden in the primary table)\n")
    (out / "MODEL_COMPATIBILITY.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out / 'MODEL_COMPATIBILITY.md'}", flush=True)


if __name__ == "__main__":
    main()
