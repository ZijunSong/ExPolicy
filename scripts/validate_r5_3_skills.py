#!/usr/bin/env python3
"""R2-style executable skill validation on the frozen public skill_validation split."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from expolicy.environments.public_search.capability_map import PUBLIC_CAPS
from expolicy.environments.public_search.evaluator import score_public
from expolicy.environments.public_search.skills import PRIMITIVES, run_public_skill_on_task
from expolicy.experiments.common import mean, write_json
from expolicy.experiments.r5_3 import (
    R53_ROOT,
    build_public_agent,
    heartbeat,
    load_base_rows,
    load_split,
    split_tasks,
    world_from_tasks,
)
from expolicy.experiments.r_common import apply_gpu
from expolicy.logging.event_logger import append_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique"])
    p.add_argument("--output-dir", default=str(R53_ROOT))
    p.add_argument("--model", default=None)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    p.add_argument("--retriever-url", default=os.environ.get("EXPOLICY_RETRIEVER_URL", "http://127.0.0.1:8000"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    apply_gpu(args.gpu)
    root = Path(args.output_dir)
    out = root / "skill_validation"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists() and args.resume:
        print("[skill-val] already done", flush=True)
        return
    split = load_split(root)
    tasks = [t for t in split_tasks(split, "skill_validation") if (t.metadata or {}).get("dataset") in set(args.datasets)]
    world = world_from_tasks(tasks)
    base_rows = load_base_rows(root)
    model_path = args.model or os.environ.get("EXPOLICY_MODEL") or "/data/ppnm/models/Qwen2.5-3B-Instruct"
    agent, state, backend, env = build_public_agent(
        world, model_path=model_path, output_dir=out / "agent", retriever_url=args.retriever_url
    )
    per_cap = {cap: {"no_skill": [], "skill": []} for cap in PUBLIC_CAPS}
    diag = []
    for i, task in enumerate(tasks):
        cap = task.capability
        if cap not in per_cap:
            continue
        heartbeat(out, phase="skill_validation", i=i, n=len(tasks), task_id=task.task_id, cap=cap)
        base = base_rows.get(task.task_id)
        if base is None:
            exp0 = agent.run_task(task, state=state)
            base = score_public(exp0, task)
            base["dispatch_source"] = "base_llm"
        skill = PRIMITIVES[cap]()
        exp_s = run_public_skill_on_task(skill, env, task, model=backend)
        row_s = score_public(exp_s, task)
        row_s["dispatch_source"] = f"skill:{cap}"
        row_s["procedure_compliance"] = float(bool((exp_s.outcome or {}).get("compliance", {}).get("executed")))
        per_cap[cap]["no_skill"].append(base)
        per_cap[cap]["skill"].append(row_s)
        append_jsonl(out / "rows.jsonl", {"task_id": task.task_id, "capability": cap, "no_skill": base, "skill": row_s})

    accepted = []
    summary = {}
    for cap, blob in per_cap.items():
        ns = blob["no_skill"]
        sk = blob["skill"]
        acc_ns = mean(float(r.get("em") or 0.0) for r in ns)
        acc_sk = mean(float(r.get("em") or 0.0) for r in sk)
        n_gain = sum(1 for a, b in zip(ns, sk) if float(b.get("em") or 0) > float(a.get("em") or 0))
        loop = mean(float(r.get("loop_rate") or 0.0) for r in sk)
        invalid = mean(float(r.get("invalid_tool_rate") or 0.0) for r in sk)
        delta = acc_sk - acc_ns
        ok = delta > 0 and n_gain >= 2 and loop < 0.5 and invalid < 0.5
        summary[cap] = {
            "n": len(sk),
            "no_skill_em": acc_ns,
            "skill_em": acc_sk,
            "delta_em": delta,
            "no_skill_f1": mean(float(r.get("f1") or 0.0) for r in ns),
            "skill_f1": mean(float(r.get("f1") or 0.0) for r in sk),
            "n_improved_examples": n_gain,
            "avg_tool_calls": mean(float(r.get("tool_calls") or 0.0) for r in sk),
            "avg_search": mean(float(r.get("n_search") or 0.0) for r in sk),
            "supporting_recall": mean(float(r.get("supporting_recall") or 0.0) for r in sk if r.get("supporting_recall") == r.get("supporting_recall")),
            "procedure_compliance": mean(float(r.get("procedure_compliance") or 1.0) for r in sk),
            "invalid_tool_rate": invalid,
            "loop_rate": loop,
            "token_cost": int(sum(float(r.get("tokens") or 0.0) for r in sk)),
            "accepted": ok,
        }
        if ok:
            accepted.append(cap)
    decision = "PASS" if len(accepted) >= 2 else "STOP"
    payload = {"decision": decision, "accepted": accepted, "per_capability": summary, "n_accepted": len(accepted)}
    write_json(out / "metrics.json", payload)
    lines = [
        "# SKILL_VALIDATION",
        "",
        f"decision: **{decision}** ({len(accepted)}/3 accepted: {accepted})",
        "",
        "Accept rule: ΔAccuracy > 0 AND ≥2 improved examples AND no catastrophic tool-loop/parse failure.",
        "",
        json.dumps(summary, indent=2),
    ]
    (root / "SKILL_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "SKILL_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if decision == "PASS":
        (out / "DONE").write_text("ok\n", encoding="utf-8")
    else:
        (out / "STOP").write_text("only 0-1 skills accepted\n", encoding="utf-8")
        raise SystemExit("skill validation STOP: fewer than 2/3 capabilities accepted")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
