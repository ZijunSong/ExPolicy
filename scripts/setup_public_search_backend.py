#!/usr/bin/env python3
"""Download Wikipedia-2018 corpus + E5 index and launch the local retriever.

Follows Search-R1 / FlashRAG public search protocol:
  Wikipedia 2018 100-word chunks + e5-base-v2 dense retrieval.
Never uses mock passages or gold supporting facts as search results.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "wiki18"
DEFAULT_MODEL = Path("/data/ppnm/models/e5-base-v2")
DEFAULT_PORT = 8000

CORPUS_REPO = "PeterJinGo/wiki-18-corpus"
HNSW_REPO = "PeterJinGo/wiki-18-e5-index-HNSW64"
FLAT_REPO = "PeterJinGo/wiki-18-e5-index"
E5_REPO = "intfloat/e5-base-v2"
FLASHRAG_REPO = "RUC-NLPIR/FlashRAG_datasets"


def _log(msg: str) -> None:
    print(f"[setup-public-search] {msg}", flush=True)


def _ensure_hf_endpoint() -> None:
    if os.environ.get("HF_ENDPOINT"):
        return
    # huggingface.co is often unreachable from this cluster; hf-mirror is the fallback.
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def snapshot(repo: str, dest: Path, *, repo_type: str = "dataset", allow_patterns=None) -> Path:
    from huggingface_hub import snapshot_download

    _ensure_hf_endpoint()
    dest.mkdir(parents=True, exist_ok=True)
    _log(f"download {repo} -> {dest}")
    path = snapshot_download(
        repo_id=repo,
        repo_type=repo_type,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    return Path(path)


def concat_parts(directory: Path, dest: Path, prefix: str = "part_") -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        _log(f"index already present: {dest} ({dest.stat().st_size} bytes)")
        return dest
    parts = sorted(p for p in directory.iterdir() if p.name.startswith(prefix) and p.is_file())
    if not parts:
        # some dumps use part-00000 style or already concatenated
        for name in ("e5_HNSW64.index", "e5_Flat.index", "index"):
            cand = directory / name
            if cand.exists() and cand.stat().st_size > 1_000_000:
                if cand.resolve() != dest.resolve():
                    shutil.copy2(cand, dest)
                return dest
        raise FileNotFoundError(f"no {prefix}* parts under {directory}")
    _log(f"concat {len(parts)} parts -> {dest}")
    with dest.open("wb") as out:
        for part in parts:
            _log(f"  {part.name} ({part.stat().st_size} bytes)")
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out)
    return dest


def decompress_corpus(directory: Path) -> Path:
    jsonl = directory / "wiki-18.jsonl"
    gz = directory / "wiki-18.jsonl.gz"
    if jsonl.exists() and jsonl.stat().st_size > 1_000_000:
        with jsonl.open("rb") as handle:
            head = handle.read(512)
        if head.lstrip().startswith(b"{") or head.lstrip().startswith(b"["):
            _log(f"corpus already present: {jsonl}")
            return jsonl
        if b"ustar" in head:
            unpacked = directory / "wiki-18.unpacked.jsonl"
            if unpacked.exists() and unpacked.stat().st_size > 1_000_000:
                _log(f"unpacked corpus already present: {unpacked}")
                return unpacked
            _log(f"extract tar corpus {jsonl} -> {unpacked}")
            import tarfile
            with tarfile.open(jsonl, "r") as tf:
                member = next((m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jsonl")), None)
                if member is None:
                    raise FileNotFoundError(f"no jsonl member in tar {jsonl}")
                src_file = tf.extractfile(member)
                if src_file is None:
                    raise FileNotFoundError(f"cannot extract {member.name}")
                with unpacked.open("wb") as out:
                    while True:
                        chunk = src_file.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            return unpacked
        _log(f"corpus already present: {jsonl}")
        return jsonl
    # HF dump may nest files
    if not gz.exists():
        found = list(directory.rglob("wiki-18.jsonl.gz")) + list(directory.rglob("wiki-18.jsonl"))
        if found:
            cand = found[0]
            if cand.suffix == ".gz":
                gz = cand
            else:
                return cand
    if gz.exists():
        _log(f"decompress {gz}")
        subprocess.check_call(["gzip", "-dkf", str(gz)])
        out = gz.with_suffix("")  # wiki-18.jsonl
        if out.exists():
            if out.resolve() != jsonl.resolve():
                shutil.move(str(out), str(jsonl))
            return jsonl
    raise FileNotFoundError(f"wiki-18 corpus not found under {directory}")


def prepare(data_dir: Path, model_dir: Path, *, use_flat: bool) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = data_dir / "corpus"
    index_dir = data_dir / ("e5_flat" if use_flat else "e5_hnsw")
    datasets_dir = data_dir.parent / "public_search" / "flashrag"
    snapshot(CORPUS_REPO, corpus_dir)
    snapshot(HNSW_REPO if not use_flat else FLAT_REPO, index_dir)
    snapshot(E5_REPO, model_dir, repo_type="model")
    try:
        snapshot(
            FLASHRAG_REPO,
            datasets_dir,
            allow_patterns=[
                "hotpotqa/*",
                "2wikimultihopqa/*",
                "musique/*",
                "bamboogle/*",
                "nq/*",
            ],
        )
    except Exception as exc:
        _log(f"FlashRAG download warning (will fall back to native loaders): {exc}")
    corpus = decompress_corpus(corpus_dir)
    index = concat_parts(index_dir, data_dir / ("e5_Flat.index" if use_flat else "e5_HNSW64.index"))
    meta = {
        "corpus": str(corpus),
        "index": str(index),
        "e5_model": str(model_dir),
        "datasets_dir": str(datasets_dir),
        "use_flat": use_flat,
    }
    (data_dir / "BACKEND_PATHS.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def launch_server(meta: dict, *, port: int, gpu: str | None, faiss_gpu: bool) -> None:
    env = os.environ.copy()
    if gpu is not None and gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable,
        str(ROOT / "expolicy" / "environments" / "public_search" / "retriever_server.py"),
        "--index_path",
        meta["index"],
        "--corpus_path",
        meta["corpus"],
        "--retriever_model",
        meta["e5_model"],
        "--retriever_name",
        "e5",
        "--topk",
        "5",
        "--port",
        str(port),
    ]
    if faiss_gpu:
        cmd.append("--faiss_gpu")
    _log("launch " + " ".join(cmd))
    os.execvpe(cmd[0], cmd, env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for E5 encoder (optional)")
    p.add_argument("--flat", action="store_true", help="use exact Flat index instead of HNSW64")
    p.add_argument("--faiss-gpu", dest="faiss_gpu", action="store_true")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--serve-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    if args.serve_only:
        import json

        meta = json.loads((data_dir / "BACKEND_PATHS.json").read_text(encoding="utf-8"))
        launch_server(meta, port=args.port, gpu=args.gpu, faiss_gpu=args.faiss_gpu)
        return
    meta = prepare(data_dir, model_dir, use_flat=args.flat)
    _log(f"ready: {meta}")
    if args.download_only:
        return
    launch_server(meta, port=args.port, gpu=args.gpu, faiss_gpu=args.faiss_gpu)


if __name__ == "__main__":
    main()
