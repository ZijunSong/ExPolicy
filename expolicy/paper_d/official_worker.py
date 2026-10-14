"""One serial official LAB episode. Outcomes come only from OfficialLabEnv."""

from __future__ import annotations

import json
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from expolicy.paper_d.constants import ProtocolError
from expolicy.paper_d.persistent_ledger import PersistentLedger
from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv
from expolicy.r9_lab.io_util import sha256_json
from expolicy.r9_lab.llm_client import VLLMClient
from expolicy.r9_lab.state.persistent_state import PersistentState

EXECUTION_REVISION = "d-v1.1-worker-1"
EXECUTION_SCHEMA = "d-real-attempt-1"


class FrozenTraceClient:
    """Send the locked decode config and keep the actual request body."""

    def __init__(self, inner: VLLMClient, *, temperature: float, top_p: float, max_tokens: int, seed: int) -> None:
        self.inner = inner
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.calls: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.calls = []

    def chat(self, messages: list[dict[str, Any]]) -> Any:
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
        body = {
            "model": self.inner.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "extra_body": extra,
            "top_k_policy": "omit_field_means_no_truncation",
            "messages": list(messages),
        }
        result = self.inner.chat(
            messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            extra_body=extra,
        )
        self.calls.append(
            {
                "request": body,
                "request_hash": sha256_json(body),
                "content": result.content,
                "reasoning": result.reasoning,
                "finish_reason": (result.raw.get("choices") or [{}])[0].get("finish_reason"),
                "usage": result.raw.get("usage") or {},
            }
        )
        return result


def execute_official_job(job: Mapping[str, Any], *, ledger: PersistentLedger, run_root: Path) -> dict[str, Any]:
    """Run one official episode. A missing artifact is not filled with a success value."""
    root = Path(run_root).resolve()
    _require_inside(root, Path(str(job["state_path"])))
    result_path = _require_inside(root, Path(str(job["result_path"])))
    if job.get("env") not in {"db", "os", "kg"}:
        raise ProtocolError(f"refusing unknown env {job.get('env')}")
    if not str(job["task_id"]).startswith(f"{job['env']}:"):
        raise ProtocolError("task env does not match the official sample")
    prior = ledger.completed_hash(str(job["logical_execution_id"]))
    if prior and result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("result_hash") == prior and cached.get("execution_complete") is True:
            cached["cache_hit"] = True
            return cached
    reserved = ledger.reserve_attempt(
        str(job["attempt_id"]),
        str(job["logical_execution_id"]),
        job_kind=str(job["job_kind"]),
    )
    if not reserved:
        raise ProtocolError("episode cap would be exceeded")
    ledger.mark_started(str(job["attempt_id"]))
    state = PersistentState.load(job["state_path"])
    if state.content_hash() != job["state_content_hash"]:
        ledger.fail_attempt(str(job["attempt_id"]))
        raise ProtocolError("state content hash does not match the frozen job")
    before_hash = state.content_hash()
    lab = None
    try:
        lab = OfficialLabEnv(str(job["env"]))
        client = FrozenTraceClient(
            VLLMClient(port=int(job["port"]), model=str(job["served_name"])),
            temperature=float(job["temperature"]),
            top_p=float(job["top_p"]),
            max_tokens=int(job["max_tokens"]),
            seed=int(job["seed"]),
        )
        result = lab.run_episode(
            job["sample_index"],
            client,
            state=state,
            task_id=str(job["task_id"]),
            instruction_hint=str(job["instruction_hint"]),
        )
        after = PersistentState.load(job["state_path"])
        artifact = {
            "execution_complete": True,
            "cache_hit": False,
            "execution_revision": EXECUTION_REVISION,
            "execution_schema_version": EXECUTION_SCHEMA,
            "job": {key: job[key] for key in job if key != "instruction_hint"},
            "instruction_hash": sha256_json(job["instruction_hint"]),
            "success": bool(result.success),
            "infra_failure": bool(result.infra_failure),
            "infra_status": "infra_failure" if result.infra_failure else "ok",
            "outcome": None if result.infra_failure else (1.0 if result.success else 0.0),
            "tokens": int(result.tokens),
            "tool_calls": int(result.steps),
            "wall_seconds": float(result.wall_time),
            "parser_valid": bool(result.parser_valid),
            "state_hash_before": before_hash,
            "state_hash_after": after.content_hash(),
            "state_unchanged": before_hash == after.content_hash(),
            "n_model_calls": len(client.calls),
            "initial_request_hash": None if not client.calls else client.calls[0]["request_hash"],
            "request_trace": client.calls,
            "trajectory": list(result.trajectory),
        }
        artifact["result_hash"] = sha256_json(
            {key: value for key, value in artifact.items() if key != "result_hash"}
        )
        _atomic_write(result_path, artifact)
        _write_traces(result_path.parent, artifact)
        ledger.finish_attempt(
            str(job["attempt_id"]),
            tokens=int(result.tokens),
            tool_calls=int(result.steps),
            result_hash=str(artifact["result_hash"]),
        )
        return artifact
    except Exception as exc:
        _atomic_write(
            result_path.parent / "error.json",
            {
                "execution_complete": False,
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
                "traceback": traceback.format_exc()[-2000:],
                "attempt_id": job["attempt_id"],
            },
        )
        ledger.fail_attempt(str(job["attempt_id"]))
        raise
    finally:
        if lab is not None:
            lab.close()


def _require_inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ProtocolError(f"path escapes the run directory: {path}")
    return resolved


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_traces(directory: Path, artifact: Mapping[str, Any]) -> None:
    requests = directory / "request_trace.jsonl"
    environment = directory / "environment_trace.jsonl"
    requests.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in artifact["request_trace"]),
        encoding="utf-8",
    )
    environment.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in artifact["trajectory"]),
        encoding="utf-8",
    )
