"""Real LAB runner gate. Dry-run plans; execute refuses to invent outcomes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from expolicy.paper_d.constants import REPO_ROOT, ProtocolError
from expolicy.paper_d.real_config import load_real_config
from expolicy.paper_d.real_manifest import audit_split, build_run_plan
from expolicy.r9_lab.io_util import sha256_file


class RealExecutionRefused(ProtocolError):
    pass


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    model_path = Path(str(model["local_path"]))
    manifest_path = _repo(str(model["manifest_path"]))
    split_path = _repo(str(config["split_path"]))
    reasons: list[str] = []
    if not model_path.is_dir():
        reasons.append(f"model directory is missing: {model_path}")
    manifest = _read_json(manifest_path)
    if manifest is None:
        reasons.append(f"model manifest is missing: {manifest_path}")
    elif manifest.get("local_path") != str(model_path):
        reasons.append("config model path does not match the frozen manifest")
    if not split_path.is_file():
        reasons.append(f"split file is missing: {split_path}")
    plan = build_run_plan(config)
    if not plan["within_cap"]:
        reasons.append("planned episodes exceed the 6000 cap")
    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "model_present": model_path.is_dir(),
        "manifest_hash": None if manifest is None else sha256_file(manifest_path),
        "split_hash": sha256_file(split_path) if split_path.is_file() else None,
        "plan": plan,
        "execute_allowed": False,
        "fake_runner": False,
    }


def execute_smoke(config: Mapping[str, Any], *, execute: bool) -> dict[str, Any]:
    report = preflight(config)
    if not execute:
        return {"status": "DRY_RUN", "episodes": 0, "preflight": report}
    if report["status"] != "PASS":
        raise RealExecutionRefused("smoke execute is blocked by preflight")
    raise RealExecutionRefused(
        "structural preflight passed, but this process has no isolated OfficialLabEnv worker; refusing to synthesize episodes"
    )


def write_real_decision(directory: Path, *, preflight_report: Mapping[str, Any], smoke: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    plan = preflight_report["plan"]
    blocked = preflight_report["status"] != "PASS" or smoke.get("status") != "EXECUTED"
    lines = [
        "# D-v1.1 真实阶段决策",
        "",
        "第一阶段真实 episode 没有被合成结果替代。",
        "",
        f"- 预检：`{preflight_report['status']}`",
        f"- smoke：`{smoke.get('status')}`",
        f"- 计划 episode：`{plan['planned_total']}`，上限 `{plan['cap']}`",
        "- 前瞻验证：`NOT_RUN`",
        "",
        "## 下一步",
        "",
    ]
    if blocked:
        lines.append("停在测量门禁。只有在隔离的官方环境 worker 上执行 smoke，才能进入算子和主面板。")
        for reason in preflight_report.get("reasons") or []:
            lines.append(f"- {reason}")
        if smoke.get("status") == "REFUSED":
            lines.append(f"- {smoke.get('reason')}")
    else:
        lines.append("结构审计通过后，才允许解释 headroom。")
    lines.append("")
    (directory / "D_REAL_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "D_PROTOCOL_LOCK.json").write_text(
        json.dumps({"config_hash": preflight_report.get("manifest_hash"), "plan": plan}, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "D_RUN_PLAN.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (directory / "D_SPLIT_AUDIT.md").write_text(_split_note(preflight_report), encoding="utf-8")


def _split_note(report: Mapping[str, Any]) -> str:
    split = report.get("split_audit") or {"valid": False, "reasons": ["split roles were not loaded in this preflight"]}
    lines = ["# D split 审计", "", f"有效：`{split.get('valid')}`", ""]
    for reason in split.get("reasons") or []:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def load_and_preflight(path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_real_config(path)
    return config, preflight(config)


def _repo(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def split_audit_for(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return audit_split(tasks)
