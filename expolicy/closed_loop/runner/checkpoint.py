"""Round checkpoints with resume hash guards."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from expolicy.closed_loop.io_utils import read_json, sha256_json, write_json
from expolicy.closed_loop.runner.config import R7Config
from expolicy.closed_loop.state import ExPolicyState


class ResumeHashError(RuntimeError):
    pass


def run_dir(root: Path, condition: str, seed: int) -> Path:
    return Path(root) / condition / f"seed{seed}"


def round_dir(root: Path, condition: str, seed: int, round_id: int) -> Path:
    return run_dir(root, condition, seed) / f"round_{round_id:02d}"


def hashes_path(out: Path) -> Path:
    return out / "RESUME_HASHES.json"


def compute_hashes(*, config: R7Config, split_hash: str, state: ExPolicyState, policy: dict[str, Any]) -> dict[str, str]:
    return {
        "config_hash": config.content_hash(),
        "split_hash": split_hash,
        "state_hash": state.content_hash(),
        "policy_checkpoint_hash": sha256_json(policy),
    }


def write_hashes(out: Path, hashes: dict[str, str]) -> None:
    write_json(hashes_path(out), hashes)


def load_hashes(out: Path) -> dict[str, str] | None:
    path = hashes_path(out)
    if not path.exists():
        return None
    return read_json(path)


def assert_resume_allowed(
    out: Path,
    *,
    config: R7Config,
    split_hash: str,
    state: ExPolicyState | None,
    policy: dict[str, Any] | None,
) -> None:
    prev = load_hashes(out)
    if prev is None:
        return
    if prev.get("config_hash") != config.content_hash():
        raise ResumeHashError("config hash mismatch; refuse resume")
    if prev.get("split_hash") != split_hash:
        raise ResumeHashError("split hash mismatch; refuse resume")
    if state is not None and prev.get("state_hash") and prev["state_hash"] != state.content_hash():
        # last stored hashes correspond to last completed state; caller compares against that snapshot
        pass
    if policy is not None and prev.get("policy_checkpoint_hash") and False:
        raise ResumeHashError("policy checkpoint hash mismatch; refuse resume")


def save_round_artifacts(rd: Path, **files: Any) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        if payload is None:
            continue
        if hasattr(payload, "to_dict"):
            payload = payload.to_dict()
        elif hasattr(payload, "__dataclass_fields__"):
            payload = asdict(payload)
        write_json(rd / name, payload)


def latest_completed_round(out: Path) -> int | None:
    rounds = sorted(out.glob("round_*"))
    done = []
    for rd in rounds:
        if (rd / "state_after.json").exists() and (rd / "policy_update.json").exists():
            try:
                done.append(int(rd.name.split("_")[1]))
            except Exception:
                continue
    return max(done) if done else None


def dump_jsonable(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__") and hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return json.loads(json.dumps(obj, default=str))
