"""Public entry point. Later stages refuse to start when the previous gate is missing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from expolicy.paper_c.public.c0 import run_c0_screen
from expolicy.paper_c.public.config import load_public_config
from expolicy.paper_c.public.runtime import construct_official_or_stop, run_runtime_audit
from expolicy.paper_c.public.split import assign_c0_screen, render_split_audit
from expolicy.r9_lab.benchmark.official_env import OfficialLabInfraError
from expolicy.r9_lab.constants import model_by_id
from expolicy.r9_lab.llm_client import VLLMClient


def _client(cfg) -> VLLMClient:
    actor = cfg.raw["actor"]
    spec = model_by_id(str(actor["model_id"]))
    return VLLMClient(base_url=str(actor["base_url"]), model=str(actor["served_name"]), spec=spec, timeout=180.0)


def _run_dir(cfg, env: str) -> Path:
    path = cfg.output_dir(env, "c11")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stage_plan(cfg, env: str) -> None:
    if env != cfg.raw["active_environment"]:
        raise SystemExit(f"{env} is not the active pilot environment")
    out = _run_dir(cfg, env)
    assignment = assign_c0_screen(cfg.root / cfg.raw["split"]["path"], env=env, seed=int(cfg.raw["split"]["seed"]))
    _write(out / "C_PUBLIC_PROTOCOL.yaml", yaml.safe_dump(dict(cfg.raw), sort_keys=False))
    _write(out / "C_SPLIT_AUDIT.md", render_split_audit(assignment))
    (out / "C_SPLIT.json").write_text(json.dumps({k: v for k, v in assignment.items() if k != "sequences"}, default=str, indent=2), encoding="utf-8")
    print(f"plan wrote {out}")
    print(f"families={assignment['families']} reduced={assignment['reduced']}")


def stage_runtime(cfg, env: str) -> dict:
    out = _run_dir(cfg, env)
    assignment = assign_c0_screen(cfg.root / cfg.raw["split"]["path"], env=env, seed=int(cfg.raw["split"]["seed"]))
    tasks = []
    for sequence in assignment["sequences"]:
        tasks.extend(sequence)
    tasks = tasks[: int(cfg.raw["stages"]["runtime"]["cap_episodes"])]
    try:
        lab = construct_official_or_stop(env)
    except OfficialLabInfraError as exc:
        decision = {"status": "BLOCKED_INFRA", "reasons": [str(exc)]}
        _write(out / "C_RUNTIME_AUDIT.md", _runtime_md(decision))
        print("BLOCKED_INFRA")
        return decision
    try:
        result = run_runtime_audit(
            lab=lab,
            client=_client(cfg),
            tasks=tasks,
            work_dir=out / "runtime",
            cap=int(cfg.raw["stages"]["runtime"]["cap_episodes"]),
        )
    finally:
        lab.close()
    _write(out / "C_RUNTIME_AUDIT.md", _runtime_md(result))
    (out / "C_RUNTIME.json").write_text(json.dumps({k: v for k, v in result.items() if k != "rows"}, default=str, indent=2), encoding="utf-8")
    print(result["status"])
    return result


def stage_c0(cfg, env: str) -> dict:
    out = _run_dir(cfg, env)
    runtime_path = out / "C_RUNTIME.json"
    if not runtime_path.exists():
        raise SystemExit("c0-screen refused: runtime audit is missing")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("status") != "C_RUNTIME_VALID":
        raise SystemExit(f"c0-screen refused: runtime status is {runtime.get('status')}")
    assignment = assign_c0_screen(cfg.root / cfg.raw["split"]["path"], env=env, seed=int(cfg.raw["split"]["seed"]))
    lab = construct_official_or_stop(env)
    try:
        result = run_c0_screen(
            lab=lab,
            client=_client(cfg),
            assignment=assignment,
            work_dir=out / "c0_screen",
            cap=int(cfg.raw["stages"]["c0_screen"]["cap_episodes"]),
        )
    finally:
        lab.close()
    (out / "C_C0_SCREEN.json").write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
    _write(out / "C_ASSIMILATOR_AUDIT.md", _c0_md(result))
    _write(out / "C_DECISION.md", _decision_md(runtime, result))
    print(result["judgment"]["status"])
    return result


def _require_signal(out: Path, stage: str) -> None:
    path = out / "C_C0_SCREEN.json"
    if not path.exists():
        raise SystemExit(f"{stage} refused: C0 screen is missing")
    status = json.loads(path.read_text(encoding="utf-8"))["judgment"]["status"]
    if status not in {"C0-SCREEN-SIGNAL", "C0-PUBLIC-PASS"}:
        raise SystemExit(f"{stage} refused: C0 status is {status}")


def _runtime_md(result: dict) -> str:
    lines = [
        "# C 运行时审计",
        "",
        f"状态：`{result.get('status')}`",
        "",
        f"成功轨迹：{result.get('n_success')}；失败：{result.get('n_failure')}；基础设施失败：{result.get('infra')}",
        "",
        f"注入：`{result.get('injection')}`",
        "",
        f"no-op 原始差：`{result.get('noop')}`",
        "",
        f"停止器：`{result.get('stopper')}`",
        "",
    ]
    for reason in result.get("reasons") or []:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def _c0_md(result: dict) -> str:
    judgment = result["judgment"]
    lines = [
        "# C0 screen 审计",
        "",
        f"状态：`{judgment.get('status')}`",
        "",
        "这是开发性质的 screen，不是公开 C0 通过。",
        "",
    ]
    for reason in judgment.get("reasons") or []:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def _decision_md(runtime: dict, c0: dict | None) -> str:
    c0_status = None if c0 is None else c0["judgment"]["status"]
    lines = [
        "# C-v1.1 决策",
        "",
        "论文主张：`NOT_RUN`",
        "",
        f"- 公开运行时：`{runtime.get('status')}`",
        f"- C0：`{c0_status or 'NOT_RUN'}`",
        "- C1 / C2 / 学习序列：未启动",
        "- 第二环境 OS：未启动",
        "",
        "合成 C-v1 的 `C-INCONCLUSIVE` 保持不变。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="C-v1.1 public KG pilot")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=["plan", "runtime", "c0-screen", "c0-confirm", "c1", "c2", "pilot"])
    parser.add_argument("--env", required=True)
    args = parser.parse_args()
    cfg = load_public_config(args.config)
    if args.stage == "plan":
        stage_plan(cfg, args.env)
        return
    if args.stage == "runtime":
        stage_runtime(cfg, args.env)
        return
    if args.stage == "c0-screen":
        stage_c0(cfg, args.env)
        return
    out = _run_dir(cfg, args.env)
    if args.stage == "c0-confirm":
        _require_signal(out, args.stage)
        raise SystemExit("c0-confirm is registered but this screen has not been accepted as C0-PUBLIC-PASS")
    _require_signal(out, args.stage)
    raise SystemExit(f"{args.stage} stays closed until C0 content confirmation")


if __name__ == "__main__":
    main()
