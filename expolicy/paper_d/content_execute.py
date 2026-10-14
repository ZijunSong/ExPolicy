"""Freeze the OS content protocol and run only the planned official episodes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from expolicy.paper_d.content_analysis import (
    align_plan,
    batch_equal_task_deltas,
    benefiting_tasks,
    bootstrap_interval,
    decide,
    holm,
    leave_one_task_out,
    role_delta,
    single_task_dependent,
    utility_from_deltas,
)
from expolicy.paper_d.content_controls import (
    clone_state,
    empty_leaks_source,
    injection_for,
    source_markers,
    write_empty,
    write_no_update,
    write_real,
)
from expolicy.paper_d.content_plan import build_block, execution_identity, expected_counts
from expolicy.paper_d.content_protocol import PROTOCOL
from expolicy.paper_d.content_tasks import history_subset, rank_donors, select_main_tasks
from expolicy.paper_d.constants import ProtocolError
from expolicy.paper_d.official_worker import execute_official_job
from expolicy.paper_d.persistent_ledger import PersistentLedger
from expolicy.r9_lab.assimilators.episodic import apply_episodic
from expolicy.r9_lab.io_util import sha256_json, write_json
from expolicy.r9_lab.state.persistent_state import PersistentState


SMOKE_ROOT = Path("outputs/paper_d_real_v11/20261014-smoke-r2")
CALIBRATION_LEDGER = Path("outputs/paper_d_real_v11/20261014-noise-calibration-v2/D_COST_LEDGER.sqlite")
HISTORY_STATE = Path("outputs/paper_d_real_v11/20261014-noise-calibration-v2/states/os/S_history_consumed_v2")
SPLIT_PATH = Path("outputs/r9_lab/split.json")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_os_buckets() -> dict[str, list[dict[str, Any]]]:
    return _load_json(SPLIT_PATH)["envs"]["os"]


def code_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _artifacts(root: Path, kind: str) -> list[dict[str, Any]]:
    found = []
    for path in sorted(root.glob("executions/*/*/result.json")):
        data = _load_json(path)
        job = data.get("job") or {}
        if job.get("env") == "os" and job.get("job_kind") == kind and data.get("infra_status") == "ok":
            data["result_path"] = str(path)
            found.append(data)
    return found


def exposed_ids(smoke_root: Path = SMOKE_ROOT) -> dict[str, list[str]]:
    groups = {"history": set(), "acquisition": set(), "development_probe": set(), "executed": set()}
    for path in Path("outputs").glob("paper_d_real_v11/**/executions/*/*/result.json"):
        data = _load_json(path)
        job = data.get("job") or {}
        if job.get("env") == "os" and job.get("task_id"):
            groups["executed"].add(str(job["task_id"]))
    for item in _artifacts(smoke_root, "history"):
        groups["history"].add(str(item["job"]["task_id"]))
    for item in _artifacts(smoke_root, "acquisition"):
        groups["acquisition"].add(str(item["job"]["task_id"]))
    for item in _artifacts(smoke_root, "smoke"):
        groups["development_probe"].add(str(item["job"]["task_id"]))
    return {name: sorted(values, key=lambda task_id: int(task_id.split(":")[1])) for name, values in groups.items()}


def _experience(item: dict[str, Any], buckets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    task_id = str(item["job"]["task_id"])
    official = _official(buckets, task_id)
    success = bool(item.get("success"))
    return {
        "task_id": task_id,
        "env": "os",
        "instruction": official["instruction"],
        "trajectory": list(item.get("trajectory") or []),
        "success": success,
        "verifier_outcome": "CORRECT" if success else "INCORRECT",
        "infra_status": "ok",
        "bucket": "acquisition_pool",
        "artifact_hash": item.get("result_hash"),
        "skill_list": official.get("skill_list") or [],
    }


def _official(buckets: dict[str, list[dict[str, Any]]], task_id: str) -> dict[str, Any]:
    for rows in buckets.values():
        for record in rows:
            if str(record["task_id"]) == task_id:
                return record
    raise KeyError(task_id)


def source_batches(smoke_root: Path, buckets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    items = _artifacts(smoke_root, "acquisition")
    items.sort(key=lambda item: int(item["job"]["sample_index"]))
    batches = {"os-B1": items[:8], "os-B2": items[8:16]}
    payload = {}
    for name, rows in batches.items():
        experiences = [_experience(item, buckets) for item in rows]
        successes = [item for item in experiences if item["success"]]
        payload[name] = {
            "all_source_ids": [item["task_id"] for item in experiences],
            "success_ids": [item["task_id"] for item in successes],
            "experiences": successes,
            "failures_kept": [item["task_id"] for item in experiences if not item["success"]],
        }
    return payload


def freeze(output: Path, smoke_root: Path = SMOKE_ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    buckets = load_os_buckets()
    exposed = exposed_ids(smoke_root)
    sources = source_batches(smoke_root, buckets)
    forbidden = set(exposed["executed"]) | set(exposed["history"]) | set(exposed["acquisition"]) | set(exposed["development_probe"])
    blocked_text = []
    for rows in buckets.values():
        for record in rows:
            if str(record["task_id"]) in forbidden:
                blocked_text.append(str(record.get("instruction") or ""))
    tasks = select_main_tasks(
        buckets["future_test"],
        buckets["retention_probe"],
        excluded_ids=forbidden,
        blocked_text=blocked_text,
    )
    cases = []
    for task_id in ("os:9", "os:6"):
        record = _official(buckets, task_id)
        cases.append(
            {
                "task_id": task_id,
                "sample_index": str(record["sample_index"]),
                "instruction": record["instruction"],
                "raw_task_hash": sha256_json({"instruction": record["instruction"].lower()}),
                "probe_role": "retention" if task_id == "os:9" else "future",
                "task_family": "development_case",
                "development_or_new": "development_case",
                "skill_list": record.get("skill_list") or [],
            }
        )
    donors = {}
    used_for_donors = set(forbidden) | {task["task_id"] for task in tasks}
    for batch_id, bundle in sources.items():
        skills = {skill for item in bundle["experiences"] for skill in item["skill_list"]}
        donors[batch_id] = rank_donors(
            buckets["acquisition_pool"],
            excluded_ids=used_for_donors,
            source_skills=skills,
            limit=32,
        )
        used_for_donors.update(item["task_id"] for item in donors[batch_id])
    manifest = {
        "main": tasks,
        "history_subset": history_subset(tasks),
        "cases": cases,
        "prior_exposure_unknown": True,
        "heldout_claim": "task-heldout relative to known D executions; whole-project exposure was not fully traceable",
    }
    write_json(output / "D_CONTENT_TASK_MANIFEST.json", manifest)
    write_json(output / "D_CONTENT_SOURCE_BUNDLES.json", {"sources": sources, "donor_order": donors})
    counts = expected_counts()
    plan = []
    plan.extend(build_block(tasks, block_id="main", state_id="S0"))
    plan.extend(build_block(manifest["history_subset"], block_id="history", state_id="S_history_consumed_v2"))
    plan.extend(build_block(cases, block_id="case", state_id="S0", development_or_new="development_case"))
    (output / "D_CONTENT_EXECUTION_PLAN.jsonl").write_text(
        "".join(json.dumps({key: value for key, value in cell.items() if key != "instruction"}, ensure_ascii=False) + "\n" for cell in plan),
        encoding="utf-8",
    )
    lock = {
        **PROTOCOL,
        "code_sha": code_sha(),
        "split_sha": _load_json(SPLIT_PATH).get("sha256"),
        "planned_episodes": counts,
        "plan_rows": len(plan),
        "donor_rule": "rank unused acquisition_pool tasks by ascending skill Jaccard to the source bundle, then sample_index; take successes in that order",
        "analysis_rule": "future contrasts are primary; positive retention is reported and not added to utility; missing retention makes utility not estimable",
    }
    write_json(output / "D_CONTENT_PROTOCOL_LOCK.json", lock)
    write_json(
        output / "D_BUDGET_SCOPE.json",
        {
            "budget_scope_id": "D-v1.1-pilot-6000",
            "prior_occupied_episodes": PROTOCOL["prior_occupied_episodes"],
            "phase_episode_cap": PROTOCOL["phase_episode_cap"],
            "cumulative_cap": PROTOCOL["cumulative_cap"],
            "phase_token_cap": PROTOCOL["phase_token_cap"],
            "prior_ledger": str(CALIBRATION_LEDGER),
        },
    )
    (output / "D_TASK_EXPOSURE_AUDIT.md").write_text(_exposure_md(exposed, tasks, donors), encoding="utf-8")
    return {"tasks": len(tasks), "plan": len(plan), "counts": counts}


def _exposure_md(exposed: dict[str, list[str]], tasks: list[dict[str, Any]], donors: dict[str, list[dict[str, Any]]]) -> str:
    families = sorted({task["task_family"] for task in tasks})
    lines = [
        "# 任务暴露审计",
        "",
        "主任务来自 future_test 与 retention_probe，排除了已知 D 执行过的 OS 任务、history 来源和 B1/B2 来源。",
        "全项目更早的 R9/A 暴露没有完整轨迹可追踪，因此记 `prior_exposure_unknown`。本轮只声称相对已知 D 开发集的 task-heldout。",
        "",
        f"- 已执行 OS 任务：{len(exposed['executed'])}",
        f"- history 来源：{len(exposed['history'])}",
        f"- acquisition 来源：{len(exposed['acquisition'])}",
        f"- 开发 probe：{len(exposed['development_probe'])}",
        f"- 新 future：{sum(1 for task in tasks if task['probe_role']=='future')}",
        f"- 新 retention：{sum(1 for task in tasks if task['probe_role']=='retention')}",
        f"- skill family 数：{len(families)}。同一 skill 组合视为一个族；只有一个任务的族不能单独支撑跨族结论。",
        f"- donor 预定顺序：{ {name: len(rows) for name, rows in donors.items()} }",
        "",
    ]
    return "\n".join(lines)


def _import_ledger(ledger: PersistentLedger, path: Path) -> None:
    import sqlite3

    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT attempt_id, logical_execution_id, job_kind, planned_episodes, COALESCE(tokens, 0), state FROM attempts"
        ).fetchall()
    finally:
        db.close()
    ledger.import_historical([(str(a), str(b), str(c), int(d), int(e), str(state)) for a, b, c, d, e, state in rows])


def _user_text(artifact: dict[str, Any]) -> str:
    trace = artifact.get("request_trace") or []
    if not trace:
        return ""
    body = trace[0]["request_wire"]
    if isinstance(body, str):
        body = json.loads(body)
    users = [str(item.get("content") or "") for item in body.get("messages") or [] if item.get("role") == "user"]
    return "\n".join(users)


def _run_one(cell: dict[str, Any], *, output: Path, ledger: PersistentLedger, state_path: Path, state_hash: str, control_hash: str, attempt_index: int) -> dict[str, Any]:
    identity = execution_identity(cell, state_hash=state_hash, control_hash=control_hash)
    attempt = f"{identity}-{output.name}-a{attempt_index}"
    job = {
        **cell,
        "env": "os",
        "job_kind": "content_eval" if cell["block_id"] != "donor" else "content_donor",
        "state_path": str(state_path),
        "state_content_hash": state_hash,
        "logical_execution_id": identity,
        "attempt_id": attempt,
        "result_path": str(output / "executions" / identity / f"{identity}-a{attempt_index}" / "result.json"),
        "port": PROTOCOL["port"],
        "served_name": PROTOCOL["served_name"],
        "temperature": PROTOCOL["temperature"],
        "top_p": PROTOCOL["top_p"],
        "max_tokens": PROTOCOL["max_tokens"],
        "seed": PROTOCOL["inference_seed"],
        "execution_revision": PROTOCOL["stage_id"],
        "instruction_hint": cell["instruction"],
    }
    artifact = execute_official_job(job, ledger=ledger, run_root=output)
    view = injection_for(PersistentState.load(state_path), cell["instruction"])
    return {
        **{key: cell[key] for key in cell if key != "instruction"},
        "logical_execution_id": identity,
        "attempt_id": attempt,
        "cache_hit": bool(artifact.get("cache_hit")),
        "outcome": artifact.get("outcome"),
        "infra_status": artifact.get("infra_status"),
        "structural_status": artifact.get("structural_status"),
        "tokens": artifact.get("tokens"),
        "tool_calls": artifact.get("tool_calls"),
        "wall_seconds": artifact.get("wall_seconds"),
        "request_wire_hash": artifact.get("initial_request_hash"),
        "user_contains_official_instruction": cell["instruction"][:80] in _user_text(artifact),
        "retrieved_ids": view["retrieved_ids"],
        "injection_hash": view["injection_hash"],
        "injection_tokens": view["injection_tokens_estimate"],
        "state_hash": state_hash,
        "control_transform_hash": control_hash,
    }


def acquire_donors(output: Path, ledger: PersistentLedger, baseline_spent: int) -> dict[str, Any]:
    bundles = _load_json(output / "D_CONTENT_SOURCE_BUNDLES.json")
    buckets = load_os_buckets()
    s0_src = PersistentState.load(SMOKE_ROOT / "states" / "os" / "S0")
    s0 = clone_into(s0_src, output / "states" / "os" / "S0_donor")
    collected: dict[str, list[dict[str, Any]]] = {}
    attempts = 0
    for batch_id, order in bundles["donor_order"].items():
        needed = 4 if batch_id == "os-B1" else 2
        got: list[dict[str, Any]] = []
        for candidate in order:
            if len(got) >= needed or attempts >= int(PROTOCOL["donor_episode_cap"]):
                break
            if ledger.spent() - baseline_spent >= int(PROTOCOL["phase_episode_cap"]):
                break
            cell = {
                "block_id": "donor",
                "state_id": "S0",
                "batch_id": batch_id,
                "condition": "DONOR_ACQUISITION",
                "task_id": candidate["task_id"],
                "sample_index": candidate["sample_index"],
                "instruction": candidate["instruction"],
                "raw_task_hash": sha256_json({"instruction": candidate["instruction"]}),
                "task_family": candidate["task_family"],
                "probe_role": "donor",
                "development_or_new": "donor",
                "repeat_id": 0,
                "randomized_position": attempts,
            }
            row = _run_one(
                cell, output=output, ledger=ledger, state_path=s0.root_path(),
                state_hash=s0.content_hash(), control_hash="donor-acquisition", attempt_index=0,
            )
            attempts += 1
            print(f"donor {attempts} {batch_id} {candidate['task_id']} outcome={row['outcome']}", flush=True)
            if row["infra_status"] == "ok" and row["outcome"] == 1.0:
                artifact = _load_json(Path(output / "executions" / row["logical_execution_id"] / f"{row['logical_execution_id']}-a0" / "result.json"))
                got.append(_experience(artifact | {"job": {"task_id": candidate["task_id"], "sample_index": candidate["sample_index"]}}, buckets))
        collected[batch_id] = got
    bundles["donors"] = collected
    write_json(output / "D_CONTENT_SOURCE_BUNDLES.json", bundles)
    return collected


def clone_into(state: PersistentState, destination: Path) -> PersistentState:
    copied = clone_state(state, str(destination))
    copied.save()
    return copied


def build_condition_states(output: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    bundles = _load_json(output / "D_CONTENT_SOURCE_BUNDLES.json")
    s0 = clone_into(PersistentState.load(SMOKE_ROOT / "states" / "os" / "S0"), output / "states" / "bases" / "S0")
    history = clone_into(PersistentState.load(HISTORY_STATE), output / "states" / "bases" / "S_history_consumed_v2")
    bases = {"S0": s0, "S_history_consumed_v2": history}
    prepared: dict[tuple[str, str, str], dict[str, Any]] = {}
    audits = []
    for state_id, base in bases.items():
        for batch_id, bundle in bundles["sources"].items():
            experiences = bundle["experiences"]
            donors = bundles.get("donors", {}).get(batch_id) or []
            real_path = output / "states" / state_id / batch_id / "REAL"
            real = write_real(base, experiences, str(real_path))
            no_update = write_no_update(base, str(output / "states" / state_id / batch_id / "NO_UPDATE"))
            empty = write_empty(
                PersistentState.load(real_path),
                real["written_ids"],
                str(output / "states" / state_id / batch_id / "EMPTY"),
            )
            wrong_path = output / "states" / state_id / batch_id / "MATCHED_WRONG"
            wrong_state = clone_into(base, wrong_path)
            wrong_report = apply_episodic(wrong_state, donors) if donors else {"task_ids": [], "n": 0}
            specs = {
                "NO_UPDATE": no_update,
                "REAL": real,
                "EMPTY": empty,
                "MATCHED_WRONG": {
                    "condition": "MATCHED_WRONG",
                    "constructed_control": False,
                    "written_ids": list(wrong_report.get("task_ids") or []),
                    "n_written": int(wrong_report.get("n") or 0),
                    "state_hash": wrong_state.content_hash(),
                    "path": str(wrong_path),
                },
            }
            for condition, spec in specs.items():
                spec["control_hash"] = sha256_json(
                    {"condition": condition, "state_hash": spec["state_hash"], "batch": batch_id, "state_id": state_id}
                )
                prepared[(state_id, batch_id, condition)] = spec
            audits.append(_audit_one(base, experiences, donors, specs, state_id, batch_id))
    write_json(output / "D_CONTENT_STATE_MANIFEST.json", [
        {"state_id": key[0], "batch_id": key[1], "condition": key[2], **value}
        for key, value in prepared.items()
    ])
    (output / "D_CONTENT_CONTROL_AUDIT.md").write_text(_audit_md(audits), encoding="utf-8")
    write_json(output / "D_CONTENT_CONTROL_AUDIT.json", audits)
    return prepared


def _audit_one(base: PersistentState, experiences: list[dict[str, Any]], donors: list[dict[str, Any]], specs: dict[str, dict[str, Any]], state_id: str, batch_id: str) -> dict[str, Any]:
    probe = experiences[0]["instruction"] if experiences else "Create a directory"
    views = {
        condition: injection_for(PersistentState.load(spec["path"]), probe)
        for condition, spec in specs.items()
    }
    leaks = empty_leaks_source(views["EMPTY"]["injection"], experiences)
    real_tokens = views["REAL"]["injection_tokens_estimate"]
    wrong_tokens = views["MATCHED_WRONG"]["injection_tokens_estimate"]
    tolerance = float(PROTOCOL["length_tolerance"])
    length_ok = real_tokens > 0 and abs(wrong_tokens - real_tokens) / real_tokens <= tolerance
    expected = len(experiences)
    return {
        "state_id": state_id,
        "batch_id": batch_id,
        "real_written": specs["REAL"]["n_written"],
        "wrong_written": specs["MATCHED_WRONG"]["n_written"],
        "expected": expected,
        "empty_leaks": leaks,
        "real_ids": views["REAL"]["retrieved_ids"],
        "empty_ids": views["EMPTY"]["retrieved_ids"],
        "wrong_ids": views["MATCHED_WRONG"]["retrieved_ids"],
        "no_update_ids": views["NO_UPDATE"]["retrieved_ids"],
        "real_tokens": real_tokens,
        "empty_tokens": views["EMPTY"]["injection_tokens_estimate"],
        "wrong_tokens": wrong_tokens,
        "length_match": "within_20_percent" if length_ok else "insufficient",
        "source_markers_present_in_real": all(marker in views["REAL"]["injection"] for marker in source_markers(experiences)),
        "donor_ids": [item["task_id"] for item in donors],
    }


def _audit_md(audits: list[dict[str, Any]]) -> str:
    lines = ["# 内容对照审计", "", "审计发生在新任务评分使用之前，依据是条目、注入文本和来源，不是评测分数。", ""]
    for item in audits:
        lines.append(
            f"- {item['state_id']} {item['batch_id']}: REAL {item['real_written']} 条, "
            f"WRONG {item['wrong_written']} 条, EMPTY 泄漏 {item['empty_leaks'] or '无'}, "
            f"长度匹配 {item['length_match']}, token REAL/EMPTY/WRONG="
            f"{item['real_tokens']}/{item['empty_tokens']}/{item['wrong_tokens']}"
        )
    lines.append("")
    return "\n".join(lines)


def run_cells(
    cells: list[dict[str, Any]],
    *,
    output: Path,
    ledger: PersistentLedger,
    prepared: dict[tuple[str, str, str], dict[str, Any]],
    baseline_spent: int,
    tokens_used: list[int],
    retries: list[int],
) -> tuple[list[dict[str, Any]], str | None]:
    results = []
    for index, cell in enumerate(cells, start=1):
        if ledger.spent() - baseline_spent >= int(PROTOCOL["phase_episode_cap"]):
            return results, "phase episode cap"
        if tokens_used[0] >= int(PROTOCOL["phase_token_cap"]):
            return results, "phase token cap"
        spec = prepared[(cell["state_id"], cell["batch_id"], cell["condition"])]
        row = None
        for attempt in range(3):
            if attempt and retries[0] >= int(PROTOCOL["retry_pool"]):
                break
            if ledger.spent() - baseline_spent >= int(PROTOCOL["phase_episode_cap"]):
                return results, "phase episode cap"
            try:
                row = _run_one(
                    cell, output=output, ledger=ledger, state_path=Path(spec["path"]),
                    state_hash=spec["state_hash"], control_hash=spec["control_hash"], attempt_index=attempt,
                )
            except ProtocolError as exc:
                if "cannot be restarted" not in str(exc):
                    raise
                continue
            if attempt and not row.get("cache_hit"):
                retries[0] += 1
            tokens_used[0] += int(row.get("tokens") or 0)
            if row.get("infra_status") == "ok":
                break
        if row is not None:
            results.append(row)
            print(
                f"content {cell['block_id']} {index}/{len(cells)} {cell['task_id']} {cell['condition']} "
                f"outcome={row.get('outcome')} tokens={tokens_used[0]}",
                flush=True,
            )
    return results, None


def structure_problem(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return "结构检查没有结果"
    bad = [row for row in rows if row.get("infra_status") != "ok" or row.get("structural_status") not in {None, "ok"}]
    if bad:
        return f"{len(bad)} 条结构检查 episode 基础设施或状态无效"
    mismatched = [row for row in rows if not row.get("user_contains_official_instruction")]
    if mismatched:
        return "官方任务文本没有出现在首条用户消息"
    return None


def controls_invalid(audits: list[dict[str, Any]]) -> str | None:
    for item in audits:
        if item["state_id"] != "S0":
            continue
        if item["real_written"] != item["expected"] or item["wrong_written"] != item["expected"]:
            return "donor 或 REAL 条目数与合格源轨迹数不一致"
        if item["empty_leaks"]:
            return "EMPTY 注入仍含源任务文本"
        if set(item["donor_ids"]) & set(item["real_ids"]):
            return "WRONG 与 REAL 使用了同一来源任务"
    return None
