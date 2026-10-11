"""Materialize official LAB entry_dict.json from HuggingFace. Never invent gold."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from expolicy.r9_lab.constants import (
    ENVS,
    HF_ENDPOINTS,
    HF_LAB_DATASET,
    HF_LAB_SUBSET,
    LAB_CHAT_HISTORY_REL,
    LAB_DATA_REL,
    LAB_ROOT,
)
from expolicy.r9_lab.io_util import sha256_json

HF_TREE_API = "/api/datasets/{repo}/tree/main/{subset}"

DB_REQUIRED = ("instruction", "skill_list", "answer_info", "table_info")
DB_ANSWER_KEYS = ("md5", "direct", "sql")
DB_TABLE_KEYS = ("name", "row_list", "column_info_list")
OS_REQUIRED = (
    "instruction",
    "initialization_command_item",
    "evaluation_info",
    "skill_list",
)
KG_REQUIRED = ("question", "entity_dict", "answer_list")


class LabDataError(RuntimeError):
    pass


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # numpy / pandas scalars
    if hasattr(obj, "item") and callable(obj.item) and not isinstance(obj, (bytes, bytearray)):
        try:
            inner = obj.item()
            if not isinstance(inner, (list, dict, tuple)):
                return _jsonable(inner)
        except Exception:
            pass
    if hasattr(obj, "tolist") and not isinstance(obj, (list, dict)):
        try:
            return _jsonable(obj.tolist())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        import pandas as pd

        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def _maybe_json(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                import ast

                try:
                    return ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return value
    return value


def nest_dotted(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in row.items():
        parts = str(key).split(".")
        cursor: dict[str, Any] = out
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = _maybe_json(_jsonable(val))
    return out


def validate_entry(env: str, entry: Mapping[str, Any]) -> None:
    if env == "db":
        missing = [k for k in DB_REQUIRED if k not in entry]
        if missing:
            raise LabDataError(f"db entry missing {missing}; refusing to invent gold")
        ans = entry.get("answer_info")
        if not isinstance(ans, Mapping) or not any(k in ans for k in DB_ANSWER_KEYS):
            raise LabDataError("db answer_info must contain md5|direct|sql; refusing to invent gold")
        table = entry.get("table_info")
        if not isinstance(table, Mapping) or any(k not in table for k in DB_TABLE_KEYS):
            raise LabDataError(f"db table_info must contain {DB_TABLE_KEYS}")
        if not isinstance(entry.get("skill_list"), list):
            raise LabDataError("db skill_list must be a list")
        if not str(entry.get("instruction") or "").strip():
            raise LabDataError("db instruction empty")
        return
    if env == "os":
        missing = [k for k in OS_REQUIRED if k not in entry]
        if missing:
            raise LabDataError(f"os entry missing {missing}; refusing to invent gold")
        return
    if env == "kg":
        missing = [k for k in KG_REQUIRED if k not in entry]
        if missing:
            raise LabDataError(f"kg entry missing {missing}; refusing to invent gold")
        return
    raise LabDataError(f"unknown env {env}")


def _row_to_entry(env: str, row: Mapping[str, Any], index: int) -> tuple[str, dict[str, Any]]:
    nested = nest_dotted(row)
    key = nested.pop("sample_index", None) or nested.pop("sample_id", None) or nested.pop("index", None)
    if key is None:
        key = str(index)
    key = str(key)
    if isinstance(nested.get("entry"), dict):
        entry = dict(nested["entry"])
    elif isinstance(nested.get("entry_dict"), dict):
        entry = dict(nested["entry_dict"])
    else:
        entry = dict(nested)
    for drop in ("__index_level_0__",):
        entry.pop(drop, None)
    for field in (
        "answer_info",
        "table_info",
        "evaluation_info",
        "entity_dict",
        "skill_list",
        "answer_list",
        "initialization_command_item",
    ):
        if field in entry:
            entry[field] = _maybe_json(entry[field])
    if env == "db":
        ans = entry.get("answer_info")
        if isinstance(ans, Mapping):
            entry["answer_info"] = {str(k): _maybe_json(v) for k, v in ans.items()}
        table = entry.get("table_info")
        if isinstance(table, Mapping):
            entry["table_info"] = {str(k): _maybe_json(v) for k, v in table.items()}
        skills = entry.get("skill_list")
        if isinstance(skills, str):
            entry["skill_list"] = _maybe_json(skills)
    if env == "os" and "raw_entry_hash" not in entry:
        entry["raw_entry_hash"] = sha256_json(
            {k: v for k, v in entry.items() if k != "raw_entry_hash"}
        )
    validate_entry(env, entry)
    return key, entry


def parquet_to_entry_dict(path: Path, env: str) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_parquet(path)
    out: dict[str, Any] = {}
    for i, row in enumerate(df.to_dict(orient="records")):
        key, entry = _row_to_entry(env, row, i)
        if key in out:
            raise LabDataError(f"duplicate sample_index {key} in {path}")
        out[key] = entry
    if not out:
        raise LabDataError(f"parquet produced empty entry_dict: {path}")
    return out


def json_file_to_entry_dict(path: Path, env: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw and all(isinstance(v, dict) for v in raw.values()):
        out = {}
        for i, (k, entry) in enumerate(raw.items()):
            _, normalized = _row_to_entry(env, entry, i)
            out[str(k)] = normalized
        return out
    if isinstance(raw, list):
        out = {}
        for i, row in enumerate(raw):
            if not isinstance(row, dict):
                raise LabDataError(f"list item {i} is not an object in {path}")
            key, entry = _row_to_entry(env, row, i)
            out[key] = entry
        return out
    raise LabDataError(f"unrecognized JSON layout in {path}")


def write_entry_dict(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def list_hf_subset_files(subset: str, *, endpoints: Sequence[str] | None = None, timeout: float = 30.0) -> list[dict[str, Any]]:
    import socket

    endpoints = list(endpoints or HF_ENDPOINTS)
    env_ep = os.environ.get("HF_ENDPOINT")
    if env_ep:
        endpoints = [env_ep.rstrip("/"), *[e for e in endpoints if e.rstrip("/") != env_ep.rstrip("/")]]
    errors: list[str] = []
    prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        for base in endpoints:
            url = base.rstrip("/") + HF_TREE_API.format(repo=HF_LAB_DATASET, subset=subset)
            try:
                req = Request(url, headers={"User-Agent": "expolicy-r9"})
                with urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                if isinstance(payload, list):
                    return payload
                errors.append(f"{url}: not a list")
            except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                errors.append(f"{url}: {exc}")
    finally:
        socket.setdefaulttimeout(prev_timeout)
    raise LabDataError("HF tree listing failed: " + " | ".join(errors))


def _download_url(url: str, dest: Path, *, timeout: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "expolicy-r9"})
    with urlopen(req, timeout=timeout) as resp, tmp.open("wb") as handle:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp, dest)


def snapshot_subset(subset: str, dest: Path, *, timeout: float = 120.0) -> Path:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    from huggingface_hub import hf_hub_download, snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    subset_dir = dest / subset
    endpoints = list(HF_ENDPOINTS)
    env_ep = os.environ.get("HF_ENDPOINT")
    if env_ep:
        endpoints = [env_ep.rstrip("/"), *[e for e in endpoints if e.rstrip("/") != env_ep.rstrip("/")]]
    last_err: Exception | None = None
    files: list[dict[str, Any]] = []
    try:
        files = list_hf_subset_files(subset, endpoints=endpoints, timeout=min(timeout, 30.0))
    except Exception as exc:  # noqa: BLE001
        last_err = exc
    names = [str(item.get("path") or "").split("/")[-1] for item in files if item.get("type") == "file"]
    for base in endpoints:
        for name in names:
            url = f"{base.rstrip('/')}/datasets/{HF_LAB_DATASET}/resolve/main/{subset}/{name}"
            target = subset_dir / name
            try:
                _download_url(url, target, timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if subset_dir.exists() and any(subset_dir.iterdir()):
            return subset_dir
    for base in endpoints:
        os.environ["HF_ENDPOINT"] = base
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(int(timeout))
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

        def _snap() -> None:
            snapshot_download(
                repo_id=HF_LAB_DATASET,
                repo_type="dataset",
                allow_patterns=[f"{subset}/*", f"{subset}/**"],
                local_dir=str(dest),
                max_workers=2,
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(_snap).result(timeout=timeout)
            if subset_dir.exists() and any(subset_dir.iterdir()):
                return subset_dir
        except (FuturesTimeout, Exception) as exc:  # noqa: BLE001
            last_err = exc
            for name in names:
                try:
                    hf_hub_download(
                        repo_id=HF_LAB_DATASET,
                        repo_type="dataset",
                        filename=f"{subset}/{name}",
                        local_dir=str(dest),
                    )
                except Exception as inner:  # noqa: BLE001
                    last_err = inner
            if subset_dir.exists() and any(subset_dir.iterdir()):
                return subset_dir
    raise LabDataError(f"snapshot_download failed for {subset}: {last_err}") from last_err


def convert_source_dir(env: str, source: Path, dest: Path) -> dict[str, Any]:
    if not source.exists():
        raise LabDataError(f"source dir missing: {source}")
    json_hits = sorted(source.rglob("entry_dict.json"))
    if json_hits:
        payload = json_file_to_entry_dict(json_hits[0], env)
        write_entry_dict(dest, payload)
        return {"env": env, "source": str(json_hits[0]), "n": len(payload), "kind": "entry_dict.json"}
    other_json = [p for p in sorted(source.rglob("*.json")) if p.name != "dataset_info.json"]
    parquet_hits = sorted(source.rglob("*.parquet"))
    if parquet_hits:
        payload = parquet_to_entry_dict(parquet_hits[0], env)
        write_entry_dict(dest, payload)
        return {"env": env, "source": str(parquet_hits[0]), "n": len(payload), "kind": "parquet"}
    if other_json:
        payload = json_file_to_entry_dict(other_json[0], env)
        write_entry_dict(dest, payload)
        return {"env": env, "source": str(other_json[0]), "n": len(payload), "kind": "json"}
    raise LabDataError(f"no entry_dict.json or parquet under {source}")


def official_dest(env: str) -> Path:
    return LAB_ROOT / LAB_DATA_REL[env]


def materialize_env(env: str, *, cache_dir: Path, skip_download: bool = False, timeout: float = 45.0) -> dict[str, Any]:
    dest = official_dest(env)
    subset = HF_LAB_SUBSET[env]
    source = cache_dir / subset
    if skip_download and not source.exists():
        raise LabDataError(f"skip_download but cache missing: {source}")
    if not skip_download:
        snapshot_subset(subset, cache_dir, timeout=timeout)
    info = convert_source_dir(env, source, dest)
    info["dest"] = str(dest)
    return info


def materialize_all(*, cache_dir: Path | None = None, skip_download: bool = False, timeout: float = 45.0) -> dict[str, Any]:
    cache = cache_dir or (LAB_ROOT / ".hf_cache" / "LifelongAgentBench")
    reports = {}
    errors = {}
    for env in ENVS:
        try:
            reports[env] = materialize_env(env, cache_dir=cache, skip_download=skip_download, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            errors[env] = f"{type(exc).__name__}: {exc}"
    return {"ok": reports, "errors": errors, "cache": str(cache)}


def chat_history_ready() -> bool:
    return all((LAB_ROOT / LAB_CHAT_HISTORY_REL[env]).is_file() for env in ENVS)


def generate_chat_history_items(*, python_bin: str | None = None) -> dict[str, Any]:
    candidates = []
    if python_bin:
        candidates.append(python_bin)
    candidates.append(sys.executable)
    # Official construct.py imports colorama via src.utils; try a conda env if present.
    for extra in (
        "/data/ppnm/miniconda3/bin/python",
        "/data/ppnm/expolicy/.venv/bin/python",
    ):
        if extra not in candidates:
            candidates.append(extra)
    last_err = ""
    cmd = ["python", "src/factories/chat_history_item/offline/construct.py"]
    for py in candidates:
        cmd = [py, "src/factories/chat_history_item/offline/construct.py"]
        proc = subprocess.run(
            cmd,
            cwd=str(LAB_ROOT),
            env={**os.environ, "PYTHONPATH": str(LAB_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            missing = [env for env in ENVS if not (LAB_ROOT / LAB_CHAT_HISTORY_REL[env]).is_file()]
            if missing:
                raise LabDataError(f"construct.py ran but chat_history_items missing for {missing}")
            return {"ok": True, "cmd": cmd, "cwd": str(LAB_ROOT), "python": py}
        last_err = f"rc={proc.returncode} py={py}\nstdout={proc.stdout[-1500:]}\nstderr={proc.stderr[-1500:]}"
        if "colorama" in (proc.stderr or "") or "ModuleNotFoundError" in (proc.stderr or ""):
            continue
    raise LabDataError(f"official construct.py failed. {last_err}")


def ensure_lab_assets(*, cache_dir: Path | None = None, skip_download: bool = False, timeout: float = 45.0) -> dict[str, Any]:
    data = materialize_all(cache_dir=cache_dir, skip_download=skip_download, timeout=timeout)
    chat = None
    chat_err = None
    try:
        if not chat_history_ready():
            chat = generate_chat_history_items()
        else:
            chat = {"ok": True, "skipped": "already_present"}
    except Exception as exc:  # noqa: BLE001
        chat_err = f"{type(exc).__name__}: {exc}"
    return {"data": data, "chat_history": chat, "chat_history_error": chat_err}
