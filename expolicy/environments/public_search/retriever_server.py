#!/usr/bin/env python3
"""Search-R1 compatible local E5 retriever over Wikipedia 2018.

POST /retrieve  {"queries": [...], "topk": 5, "return_scores": true}
POST /open      {"document_id": "123"}
GET  /health
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

# Running this file as a script prepends its directory to sys.path, which would
# shadow HuggingFace `datasets` with local datasets.py.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _HERE]

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise SystemExit("faiss-cpu is required for the public retriever") from exc

try:
    import datasets
except ImportError as exc:  # pragma: no cover
    raise SystemExit("datasets is required for the public retriever") from exc


def pooling(last_hidden_state, attention_mask):
    hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class Encoder:
    def __init__(self, model_path: str, device: str, use_fp16: bool = True) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else torch.float32
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(device=device, dtype=dtype)
        self.model.eval()

    @torch.no_grad()
    def encode(self, queries: list[str]) -> np.ndarray:
        texts = [f"query: {q}" for q in queries]
        encoded = self.tokenizer(
            texts, max_length=256, padding=True, truncation=True, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        output = self.model(**encoded, return_dict=True)
        emb = pooling(output.last_hidden_state, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.detach().cpu().numpy().astype(np.float32, order="C")


def ensure_jsonl_corpus(path: str) -> str:
    """PeterJinGo wiki-18 dump is often a tar wrapping wiki_dump.jsonl."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(path)
    with src.open("rb") as handle:
        head = handle.read(512)
    if head.lstrip().startswith(b"{") or head.lstrip().startswith(b"["):
        return str(src)
    if b"ustar" in head:
        unpacked = src.with_name("wiki-18.unpacked.jsonl")
        if unpacked.exists() and unpacked.stat().st_size > 1_000_000:
            print(f"[retriever] using unpacked corpus {unpacked}", flush=True)
            return str(unpacked)
        print(f"[retriever] extracting tar corpus {src} -> {unpacked}", flush=True)
        with tarfile.open(src, "r") as tf:
            member = next((m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jsonl")), None)
            if member is None:
                member = next((m for m in tf.getmembers() if m.isfile()), None)
            if member is None:
                raise SystemExit(f"no jsonl member in tar {src}")
            src_file = tf.extractfile(member)
            if src_file is None:
                raise SystemExit(f"cannot extract {member.name}")
            with unpacked.open("wb") as out:
                while True:
                    chunk = src_file.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        print(f"[retriever] unpacked size {unpacked.stat().st_size}", flush=True)
        return str(unpacked)
    return str(src)


def load_corpus(path: str):
    jsonl = ensure_jsonl_corpus(path)
    print(f"[retriever] loading huggingface datasets from {jsonl}", flush=True)
    return datasets.load_dataset("json", data_files=jsonl, split="train", num_proc=4)


def serialize_doc(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    contents = str(raw.get("contents") or "")
    title = str(raw.get("title") or "")
    text = str(raw.get("text") or "")
    if contents and not title:
        lines = contents.split("\n", 1)
        title = lines[0].strip().strip('"')
        text = lines[1] if len(lines) > 1 else ""
    doc_id = str(raw.get("id") if raw.get("id") is not None else idx)
    return {
        "id": doc_id,
        "document_id": doc_id,
        "title": title,
        "text": text,
        "contents": contents or f"{title}\n{text}".strip(),
        "corpus_index": idx,
    }


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = True


class OpenRequest(BaseModel):
    document_id: str


def build_app(index_path: str, corpus_path: str, model_path: str, *, topk: int, device: str, faiss_gpu: bool) -> FastAPI:
    print(f"[retriever] loading corpus {corpus_path}", flush=True)
    corpus = load_corpus(corpus_path)
    print(f"[retriever] corpus n={len(corpus)}", flush=True)
    print(f"[retriever] loading faiss {index_path}", flush=True)
    index = faiss.read_index(index_path)
    if faiss_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    print(f"[retriever] index ntotal={index.ntotal}", flush=True)
    encoder = Encoder(model_path, device=device)
    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "corpus": len(corpus),
            "index": int(index.ntotal),
            "device": device,
            "topk": topk,
        }

    @app.post("/retrieve")
    def retrieve(req: QueryRequest):
        k = int(req.topk or topk)
        queries = [str(q) for q in req.queries]
        if not queries:
            return {"result": []}
        emb = encoder.encode(queries)
        scores, idxs = index.search(emb, k)
        result = []
        for qi in range(len(queries)):
            group = []
            for rank, (sc, ix) in enumerate(zip(scores[qi].tolist(), idxs[qi].tolist())):
                if int(ix) < 0:
                    continue
                doc = serialize_doc(corpus[int(ix)], int(ix))
                item = {"document": doc, "score": float(sc), "rank": rank}
                group.append(item if req.return_scores else doc)
            result.append(group)
        return {"result": result}

    @app.post("/open")
    def open_doc(req: OpenRequest):
        key = str(req.document_id)
        ix = None
        try:
            cand = int(key)
            if 0 <= cand < len(corpus):
                ix = cand
        except ValueError:
            ix = None
        if ix is None:
            raise HTTPException(status_code=404, detail=f"unknown document_id {key}")
        return {"document": serialize_doc(corpus[ix], ix)}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--retriever_model", default="/data/ppnm/models/e5-base-v2")
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--faiss_gpu", action="store_true")
    args = parser.parse_args()
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    app = build_app(
        args.index_path,
        args.corpus_path,
        args.retriever_model,
        topk=args.topk,
        device=device,
        faiss_gpu=args.faiss_gpu,
    )
    print(f"[retriever] serving on {args.host}:{args.port} device={device}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
