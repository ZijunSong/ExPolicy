"""Run one B-v1 profile: collect, induce, validate, freeze, score, decide."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from expolicy.paper_b.artifacts import (
    control_audit_md,
    provenance_rows,
    split_audit_md,
    write_jsonl,
    write_protocol,
    write_rows,
    write_text,
)
from expolicy.paper_b.config import PaperBConfig
from expolicy.paper_b.constants import (
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    CONDITIONS,
    FAMILY_RETENTION,
    HARNESS_LOCAL,
    SPLIT_RETENTION,
    SPLIT_TRAIN,
    SPLIT_UNRELATED,
    SPLIT_VAL,
    TARGET_FAMILIES,
    TARGET_SPLITS,
    TEMPLATE_REWRITE,
    M,
)
from expolicy.paper_b.controls import make_experience_controls
from expolicy.paper_b.decision import judge, practical_floor, render_decision_md
from expolicy.paper_b.evaluate import freeze_and_evaluate
from expolicy.paper_b.execute import base_policy, execute_rules, exploration_policy
from expolicy.paper_b.induce import executable_hash, induce_rules
from expolicy.paper_b.invariants import (
    answer_leakage,
    check_budget_headroom,
    check_confirmation_sealed,
    check_rule_visibility,
    check_unmatched_falls_back,
)
from expolicy.paper_b.models import Budget, Rule, Task, Trace
from expolicy.paper_b.registry import RuntimeRegistry, generic_rules
from expolicy.paper_b.split import audit_splits
from expolicy.paper_b.stats import family_diffs, paired_comparison, retention_delta
from expolicy.paper_b.traces import build_trace_pairs
from expolicy.paper_b.validate import validate_rules
from expolicy.paper_b.world import build_world
from expolicy.r9_lab.io_util import sha256_json, write_json

GenerateFn = Callable[[str, float], str]


def run_protocol(
    config: PaperBConfig,
    out_dir: Path,
    *,
    profile: str,
    generate_fn: GenerateFn | None = None,
    tasks: list[Task] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = config.budget()
    registry = config.registry()
    world = tasks if tasks is not None else build_world(config.counts(profile), seed=config.split_seed)
    full_counts = {split: sum(families.values()) for split, families in config.counts("pilot").items()}
    audit = audit_splits(world, full_counts=full_counts)
    audit["split_seed"] = config.split_seed
    if not audit["ok"]:
        raise RuntimeError(f"split audit failed: {audit['errors']}")
    headroom = check_budget_headroom(budget)
    if headroom:
        raise RuntimeError(f"budget invariant failed: {headroom}")

    grouped = _by_split(world)
    train = grouped[SPLIT_TRAIN]
    val_target = [task for task in grouped[SPLIT_VAL] if task.family in TARGET_FAMILIES]
    val_retention = [task for task in grouped[SPLIT_VAL] if task.family == FAMILY_RETENTION]
    confirmation = [task for task in world if task.split in (*TARGET_SPLITS, SPLIT_RETENTION)]
    sealed = check_confirmation_sealed(train, confirmation)
    sealed.extend(check_unmatched_falls_back(train[0], registry, budget))
    if sealed:
        raise RuntimeError(f"protocol invariant failed: {sealed}")

    train_traces, collect_tokens, collect_tools = _collect(train, registry, budget, ("base", "exploration"))
    unrelated_traces, _, _ = _collect(grouped[SPLIT_UNRELATED], registry, budget, ("base",))
    pairs = build_trace_pairs(train_traces)
    controls = make_experience_controls(pairs, train_traces, unrelated_traces)
    prior = generic_rules()
    train_answers = [task.gold_answer for task in train]

    induced = {
        B3: induce_rules(
            mode="empty",
            condition=B3,
            registry=registry,
            empty_count=len(train_traces),
            generate_fn=generate_fn,
            temperature=config.inducer_temperature,
        ),
        B4: induce_rules(
            mode="mismatched",
            condition=B4,
            registry=registry,
            pairs=controls["mismatched_pairs"],
            generate_fn=None,
        ),
        B5: induce_rules(
            mode="direct",
            condition=B5,
            registry=registry,
            traces=train_traces,
            generate_fn=None,
        ),
        M: induce_rules(
            mode="aligned",
            condition=M,
            registry=registry,
            pairs=pairs,
            generate_fn=generate_fn,
            temperature=config.inducer_temperature,
        ),
    }
    validated: dict[str, dict[str, Any]] = {}
    for condition, proposal in induced.items():
        validated[condition] = validate_rules(
            proposal["rules"],
            condition=condition,
            train_tasks=train,
            val_tasks=val_target,
            retention_tasks=val_retention,
            registry=registry,
            budget=budget,
            generic_rules=prior,
            traces=train_traces,
            counterexamples=condition != B5,
            blocklist=train_answers,
            retention_tolerance=config.retention_tolerance,
        )
    conditions: dict[str, list[Rule]] = {
        B0: [],
        B1: prior,
        B2: [],
        B3: validated[B3]["rules"],
        B4: validated[B4]["rules"],
        B5: validated[B5]["rules"],
        M: validated[M]["rules"],
    }
    visibility = check_rule_visibility(
        [rule for rules in conditions.values() for rule in rules],
        registry,
        train_answers,
    )
    leaked = answer_leakage(
        [rule for rules in conditions.values() for rule in rules],
        [task.gold_answer for task in world],
    )
    if visibility or leaked:
        raise RuntimeError(f"rule visibility failed: {visibility or leaked}")

    probe = _replacement_probe(val_target, conditions[M], registry, budget)
    text = _text_experience(train_traces)
    pool_hash = _pool_hashes(train_traces, controls)
    inducer = {name: "none" for name in CONDITIONS}
    for name, proposal in induced.items():
        inducer[name] = proposal["inducer_hash"]
    generation_cost = {name: 0 for name in CONDITIONS}
    validation_cost = {name: 0 for name in CONDITIONS}
    for name, proposal in induced.items():
        generation_cost[name] = len(proposal["prompt"]) // 4
        validation_cost[name] = int(validated[name]["validation_tokens"])
    validation_cost[M] += int(probe["tokens"])

    evaluation = freeze_and_evaluate(
        tasks=confirmation,
        conditions=conditions,
        text_context={B2: text},
        registry=registry,
        budget=budget,
        rules_path=out_dir / "B_RULES.json",
        runtime_hash=registry.hash(),
        pool_hash=pool_hash,
        inducer_hash=inducer,
        generation_cost=generation_cost,
        validation_cost=validation_cost,
    )
    rows = evaluation["rows"]
    floor = practical_floor(config.min_gain, _n_target(rows), config.resolution_tasks)
    alpha = config.alpha
    comparisons = {
        "M_vs_B1": paired_comparison(
            rows, M, B1, set(TARGET_SPLITS), n_boot=config.bootstrap_replicates, seed=config.bootstrap_seed, alpha=alpha / 2
        ),
        "M_vs_B3": paired_comparison(
            rows, M, B3, set(TARGET_SPLITS), n_boot=config.bootstrap_replicates, seed=config.bootstrap_seed + 1, alpha=alpha / 2
        ),
        "M_vs_B5": paired_comparison(
            rows, M, B5, set(TARGET_SPLITS), n_boot=config.bootstrap_replicates, seed=config.bootstrap_seed + 2, alpha=alpha
        ),
    }
    families = family_diffs(rows, M, B1, set(TARGET_SPLITS), TARGET_FAMILIES)
    retention = retention_delta(rows, M, B0, SPLIT_RETENTION)
    trigger_rate = _trigger_rate(rows, M)
    proposed = validated[M]["proposed"]
    illegal_rate = (validated[M]["static_rejected"] / proposed) if proposed else 0.0
    judged = judge(
        comparisons,
        floor=floor,
        retention_delta=retention,
        retention_tolerance=config.retention_tolerance,
        family_diffs=families,
        replacement_changed=bool(probe["changed"]),
        trigger_rate=trigger_rate,
        illegal_rate=illegal_rate,
        min_trigger=config.min_trigger,
        max_illegal=config.max_illegal,
    )
    paper_claim_allowed = judged["status"] == "B-GO" and config.harness != HARNESS_LOCAL
    decision = {
        **judged,
        "protocol_id": config.protocol_id,
        "profile": profile,
        "harness": config.harness,
        "paper_claim_allowed": paper_claim_allowed,
        "floor": floor,
        "retention_delta": retention,
        "family_diffs": families,
        "trigger_rate": trigger_rate,
        "illegal_rate": illegal_rate,
        "replacement": probe,
        "scope_note": _scope_note(config.harness, profile, audit),
        **comparisons,
    }
    controls["audit"]["b1_b3_same_executable"] = executable_hash(conditions[B1]) == executable_hash(conditions[B3])
    _write_outputs(
        out_dir,
        config,
        audit,
        controls,
        train_traces,
        pairs,
        conditions,
        induced,
        validated,
        evaluation,
        decision,
        collect_tokens,
        collect_tools,
    )
    return decision


def _collect(
    tasks: list[Task],
    registry: RuntimeRegistry,
    budget: Budget,
    continuations: tuple[str, ...],
) -> tuple[list[Trace], int, int]:
    traces: list[Trace] = []
    tokens = 0
    tools = 0
    policies = {"base": base_policy, "exploration": exploration_policy}
    for task in tasks:
        for name in continuations:
            episode = execute_rules(task, [], registry, budget=budget, policy=policies[name])
            tokens += episode.tokens
            tools += episode.tool_calls
            traces.append(
                Trace(
                    trace_id=f"{task.task_id}:{name}",
                    task_id=task.task_id,
                    source_group=task.source_group,
                    split=task.split,
                    family=task.family,
                    continuation=name,
                    success=episode.success,
                    steps=list(episode.steps),
                )
            )
    return traces, tokens, tools


def _replacement_probe(
    tasks: list[Task],
    rules: list[Rule],
    registry: RuntimeRegistry,
    budget: Budget,
) -> dict[str, Any]:
    if not rules:
        return {"changed": False, "n_action_changed": 0, "n_success_changed": 0, "tokens": 0, "n_tasks": len(tasks)}
    original = [execute_rules(task, rules, registry, budget=budget) for task in tasks]
    swapped = _swap_rewrite(rules)
    alternate = [execute_rules(task, swapped, registry, budget=budget) for task in tasks]
    n_action = sum(_episode_signature(left) != _episode_signature(right) for left, right in zip(original, alternate, strict=True))
    n_success = sum(left.success != right.success for left, right in zip(original, alternate, strict=True))
    tokens = sum(episode.tokens for episode in original) + sum(episode.tokens for episode in alternate)
    return {
        "changed": n_action > 0 and n_success > 0,
        "n_action_changed": n_action,
        "n_success_changed": n_success,
        "tokens": tokens,
        "n_tasks": len(tasks),
        "used_only_development_tasks": True,
    }


def _episode_signature(episode: Any) -> list[tuple[str, str]]:
    signatures = []
    for template, step in zip(episode.action_templates, episode.steps, strict=False):
        signatures.append((template, sha256_json(step.parameters)))
    return signatures


def _swap_rewrite(rules: list[Rule]) -> list[Rule]:
    swapped = []
    for rule in rules:
        parameters = dict(rule.parameters)
        template = rule.action_template
        if template == TEMPLATE_REWRITE:
            parameters["rewrite_template"] = "{entity}"
        swapped.append(
            Rule(
                rule_id=rule.rule_id + "_swap",
                family=rule.family,
                trigger={"all": [dict(atom) for atom in rule.trigger.get("all") or []]},
                action_template=template,
                parameters=parameters,
                fallback=rule.fallback,
            )
        )
    return swapped


def _text_experience(traces: list[Trace]) -> str:
    lines = []
    for trace in traces:
        if not trace.success:
            continue
        for step in trace.steps:
            if step.action_template not in {TEMPLATE_REWRITE, "search_entity_attribute"}:
                continue
            lines.append(
                "A successful continuation used "
                f"{step.action_template} with parameters {step.parameters}."
            )
    return "\n".join(lines)


def _pool_hashes(traces: list[Trace], controls: dict[str, Any]) -> dict[str, str]:
    real = sha256_json([trace.to_dict() for trace in traces])
    empty = sha256_json(controls["empty_experiences"])
    mismatched = sha256_json([pair.to_dict() for pair in controls["mismatched_pairs"]])
    none = sha256_json({"experience": "none"})
    return {B0: none, B1: none, B2: real, B3: empty, B4: mismatched, B5: real, M: real}


def _trigger_rate(rows: list[dict[str, Any]], condition: str) -> float:
    subset = [row for row in rows if row["condition"] == condition and row["split"] in TARGET_SPLITS]
    if not subset:
        return 0.0
    return sum(1.0 for row in subset if row["triggered_rules"]) / len(subset)


def _n_target(rows: list[dict[str, Any]]) -> int:
    return len({row["task_id"] for row in rows if row["condition"] == M and row["split"] in TARGET_SPLITS})


def _scope_note(harness: str, profile: str, audit: dict[str, Any]) -> str:
    precision = "确认集小于预注册 pilot，分辨率下限会抬高。" if audit.get("precision_warning") else "确认集达到预注册 pilot 规模。"
    if harness == HARNESS_LOCAL:
        return (
            "这次运行的执行器是本地搜索替身，任务是合成的，不是 ESI 或 LifelongAgentBench 的官方测试。"
            f" profile={profile}。{precision} 统计状态只描述这次样本，不能写入论文结果。"
        )
    return f"执行器哈希已冻结。profile={profile}。{precision}"


def _by_split(tasks: list[Task]) -> dict[str, list[Task]]:
    grouped: dict[str, list[Task]] = {name: [] for name in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_UNRELATED, *TARGET_SPLITS, SPLIT_RETENTION)}
    for task in tasks:
        grouped.setdefault(task.split, []).append(task)
    for name in grouped:
        grouped[name].sort(key=lambda task: task.task_id)
    return grouped


def _write_outputs(
    out_dir: Path,
    config: PaperBConfig,
    audit: dict[str, Any],
    controls: dict[str, Any],
    traces: list[Trace],
    pairs: list,
    conditions: dict[str, list[Rule]],
    induced: dict[str, dict[str, Any]],
    validated: dict[str, dict[str, Any]],
    evaluation: dict[str, Any],
    decision: dict[str, Any],
    collect_tokens: int,
    collect_tools: int,
) -> None:
    write_protocol(out_dir / "B_PROTOCOL.yaml", config.resolved(decision["profile"]))
    write_text(out_dir / "B_SPLIT_AUDIT.md", split_audit_md(audit))
    write_text(out_dir / "B_TEXT_EXPERIENCE.txt", _text_experience(traces))
    write_jsonl(out_dir / "B_TRACE_PAIRS.jsonl", [pair.to_dict() for pair in pairs])
    write_jsonl(out_dir / "B_TRACES.jsonl", [trace.to_dict() for trace in traces])
    provenance = []
    for condition, proposal in validated.items():
        provenance.extend(provenance_rows(condition, conditions[condition], proposal["rejected"]))
    write_jsonl(out_dir / "B_RULE_PROVENANCE.jsonl", provenance)
    write_text(out_dir / "B_CONTROL_AUDIT.md", control_audit_md(controls["audit"]))
    write_rows(out_dir / "B_TASK_RESULTS.csv", evaluation["rows"])
    write_jsonl(out_dir / "B_TRIGGER_LOG.jsonl", evaluation["triggers"])
    ledger = [
        {
            "stage": "collect",
            "condition": "shared",
            "read_experience": True,
            "tokens": collect_tokens,
            "tool_calls": collect_tools,
            "note": "空经验条件不能读取这些轨迹",
        }
    ]
    for condition, proposal in induced.items():
        ledger.append(
            {
                "stage": "induce",
                "condition": condition,
                "read_experience": proposal["read_experience"],
                "llm_calls": proposal["llm_calls"],
                "proposed": validated[condition]["proposed"],
                "generation_tokens": len(proposal["prompt"]) // 4,
            }
        )
        ledger.append(
            {
                "stage": "validate",
                "condition": condition,
                "accepted": len(conditions[condition]),
                "validation_tokens": validated[condition]["validation_tokens"],
                "rejected": len(validated[condition]["rejected"]),
            }
        )
    write_jsonl(out_dir / "B_COST_LEDGER.jsonl", ledger)
    write_text(out_dir / "B_DECISION.md", render_decision_md(decision))
    write_json(out_dir / "decision.json", decision)
    write_json(out_dir / "split_audit.json", audit)
    write_json(out_dir / "control_audit.json", controls["audit"])
