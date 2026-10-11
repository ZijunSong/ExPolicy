#!/usr/bin/env python3
"""Local OpenAI-compatible embedding server.

Serves MiniLM-L6-v2 and zero-pads to 3072 so unmodified MemRL Qdrant
(vector_dimension=3072) can call it as if it were text-embedding-3-large.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

TARGET_DIM = 3072
MODEL_NAME = "all-MiniLM-L6-v2"


class Handler(BaseHTTPRequestHandler):
    embedder = None

    def log_message(self, fmt: str, *args) -> None:
        print("[%s] " % self.log_date_time_string() + fmt % args, flush=True)

    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path in ("/health", "/v1/health"):
            self._json(200, {"ok": True, "model": MODEL_NAME, "dim": TARGET_DIM})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.rstrip("/") not in ("/v1/embeddings", "/embeddings"):
            self._json(404, {"error": "not found"})
            return
        inp = body.get("input")
        if isinstance(inp, str):
            texts = [inp]
        elif isinstance(inp, list):
            texts = [str(x) for x in inp]
        else:
            self._json(400, {"error": "input must be string or list"})
            return
        vecs = Handler.embedder.encode(texts, convert_to_numpy=True)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        padded = []
        for row in vecs:
            v = np.asarray(row, dtype=np.float32)
            if v.shape[0] < TARGET_DIM:
                out = np.zeros((TARGET_DIM,), dtype=np.float32)
                out[: v.shape[0]] = v
            else:
                out = v[:TARGET_DIM]
            padded.append(out.tolist())
        data = [
            {"object": "embedding", "index": i, "embedding": emb}
            for i, emb in enumerate(padded)
        ]
        self._json(
            200,
            {
                "object": "list",
                "data": data,
                "model": body.get("model") or "text-embedding-3-large",
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18091)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    from sentence_transformers import SentenceTransformer

    Handler.embedder = SentenceTransformer(MODEL_NAME, device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"embedder listening on http://{args.host}:{args.port} dim={TARGET_DIM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
