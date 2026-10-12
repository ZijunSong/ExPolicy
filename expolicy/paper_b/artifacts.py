"""Write the B-v1 artifact set. Evaluation reads rules; it does not rewrite them."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from expolicy.paper_b.constants import ROW_FIELDS
from expolicy.paper_b.models import Rule
from expolicy.r9_lab.io_util import sha256_json, sha256_text, write_json


def write_protocol(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_rules(path: Path, payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body["rule_set_sha256"] = sha256_json({key: value for key, value in body.items() if key != "rule_set_sha256"})
    write_json(path, body)
    return sha256_text(path.read_text(encoding="utf-8"))


def assert_rules_unchanged(path: Path, digest: str) -> None:
    current = sha256_text(path.read_text(encoding="utf-8"))
    if current != digest:
        raise RuntimeError("rule file changed during evaluation")


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in ROW_FIELDS})


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def split_audit_md(audit: Mapping[str, Any]) -> str:
    counts = ", ".join(f"{key}={value}" for key, value in sorted((audit.get("counts") or {}).items()))
    errors = audit.get("errors") or []
    error_text = "无" if not errors else "；".join(errors)
    warning = "有" if audit.get("precision_warning") else "无"
    return "\n".join(
        [
            "# B 划分审计",
            "",
            f"通过：`{audit.get('ok')}`",
            "",
            f"各 split 任务数：{counts}",
            "",
            f"错误：{error_text}",
            "",
            f"规模低于预注册 pilot：{warning}",
            "",
            f"划分哈希：`{audit.get('assignment_sha256')}`",
            "",
            "划分键是来源模板、实体组合和数据库对象。正确答案没有参与划分。",
            "",
        ]
    )


def control_audit_md(audit: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# B 经验对照审计",
            "",
            f"真实轨迹数：{audit.get('n_real_traces')}",
            f"真实对齐片段数：{audit.get('n_real_pairs')}",
            f"空经验条数：{audit.get('n_empty')}；空经验里是否还留着动作：`{audit.get('empty_has_actions')}`",
            f"错配片段数：{audit.get('n_mismatched')}；关联已被打断：`{audit.get('association_broken')}`",
            f"无关真实经验条数：{audit.get('n_unrelated')}；与训练任务不交：`{audit.get('unrelated_disjoint')}`",
            f"B1 与 B3 的可执行规则哈希相同：`{audit.get('b1_b3_same_executable')}`",
            "",
            "只把经验重新排序不算对照。错配必须换成另一条片段上的修复动作，并保持 schema 有效。",
            "",
        ]
    )


def provenance_rows(condition: str, rules: Sequence[Rule], rejected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rule in rules:
        rows.append(
            {
                "condition": condition,
                "accepted": True,
                "rule_id": rule.rule_id,
                "action_template": rule.action_template,
                "support_trace_ids": list(rule.support_trace_ids),
                "counterexample_trace_ids": list(rule.counterexample_trace_ids),
                "revision_count": rule.revision_count,
                "validation_record": rule.validation_record,
                "executable": rule.executable(),
            }
        )
    for item in rejected:
        rows.append({"condition": condition, "accepted": False, **dict(item)})
    return rows
