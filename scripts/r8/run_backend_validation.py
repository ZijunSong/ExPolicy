#!/usr/bin/env python3
"""R8-B backend validation: official MemRL / SkillOpt / no-update on the frozen split.

GPU jobs talk to a shared vLLM OpenAI server (Qwen2.5-3B-Instruct). ALFWorld
itself is CPU. Does not train an ExPolicy controller.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expolicy.r8.config_emit import dump_yaml, memrl_env_yaml, memrl_run_yaml, SKILLOPT_INITIAL
from expolicy.r8.constants import (
    ALFWORLD_DATA,
    EMBED_PORT,
    FAMILIES,
    MODEL_NAME,
    MODEL_PATH,
    ROOT as EXROOT,
    VLLM_PORT,
)
from expolicy.r8.family import family_from_gamefile
from expolicy.r8.memrl_official import evaluate_memrl_checkpoint, run_memrl_official
from expolicy.r8.skillopt_official import find_best_skill, run_skillopt_eval, run_skillopt_train

OUT = EXROOT / "outputs" / "r8_backend_validation"


def _load_split(split_dir: Path) -> dict:
    payload = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
    digest = (split_dir / "split.sha256").read_text(encoding="utf-8").strip()
    payload["_sha256"] = digest
    return payload


def _wait_http(url: str, timeout: int = 600) -> None:
    import urllib.request

    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:
            last = exc
        time.sleep(5)
    raise SystemExit(f"timeout waiting for {url}: {last}")


def _write_skillopt_eval_split(split_dir: Path, bucket: str, dest: Path) -> None:
    payload = _load_split(split_dir)
    items = []
    for i, row in enumerate(payload["buckets"][bucket]):
        items.append(
            {
                "id": f"{bucket}:{i:04d}:{row['trial_id']}",
                "gamefile": row["gamefile"],
                "task_type": row["task_type"],
            }
        )
    for name in ("train", "val", "test"):
        (dest / name).mkdir(parents=True, exist_ok=True)
        (dest / name / "items.json").write_text(
            json.dumps(items if name == "test" else [], indent=2) + "\n",
            encoding="utf-8",
        )


def _lock_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mark(job: Path, name: str, payload: dict | None = None) -> None:
    (job / name).write_text(json.dumps(payload or {}, indent=2), encoding="utf-8")


def _parse_skillopt_scores(out_root: Path) -> dict:
    hits = list(out_root.rglob("*.json"))
    best: dict = {}
    rows: list[dict] = []
    wins = 0
    n_ep = 0
    by_fam: dict[str, list[bool]] = {fam: [] for fam in FAMILIES}
    for path in hits:
        if path.name in {"items.json", "split_manifest.json", "mos_config.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if path.name == "conversation.json" and isinstance(data, list) and data:
            n_ep += 1
            last = data[-1] if isinstance(data[-1], dict) else {}
            success = bool(last.get("reward", 0) >= 1 or last.get("success") or last.get("won"))
            if any(isinstance(step, dict) and float(step.get("reward") or 0) >= 1 for step in data):
                success = True
            wins += int(success)
            fam = family_from_gamefile(path.parent.name)
            if fam:
                by_fam[fam].append(success)
            rows.append({"id": path.parent.name, "success": success, "steps": len(data)})
            continue
        if isinstance(data, dict) and ("success" in data or "score" in data or "test_score" in data):
            best = {
                "path": str(path),
                "success": data.get("success", data.get("score", data.get("test_score"))),
                "raw_keys": sorted(data.keys())[:40],
            }
        if isinstance(data, dict) and isinstance(data.get("per_item"), list):
            rows.extend(data["per_item"])
    rates = {fam: (sum(vals) / len(vals) if vals else None) for fam, vals in by_fam.items()}
    if n_ep:
        best = {"path": "conversation.json", "success": wins / n_ep, "n": n_ep, "wins": wins}
    return {"summary": best, "family_rates": rates, "n_rows": max(n_ep, len(rows))}


def run_foundation(args) -> dict:
    split_dir = Path(args.split)
    dest = OUT / "jobs" / "foundation_no_learning"
    if (dest / "DONE").exists() and not args.force:
        return json.loads((dest / "DONE").read_text())
    _lock_dir(dest)
    eval_split = dest / "calib_split"
    _write_skillopt_eval_split(split_dir, "calibration", eval_split)
    rc = run_skillopt_eval(
        split_dir=eval_split,
        skill_path=Path(SKILLOPT_INITIAL),
        out_root=dest / "eval",
        llm_base=args.llm_base,
        python_bin=args.python,
        workers=args.workers,
        seed=0,
        log_path=dest / "eval.log",
    )
    parsed = _parse_skillopt_scores(dest / "eval")
    result = {
        "condition": "foundation_no_learning",
        "returncode": rc,
        "model": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "parsed": parsed,
    }
    _mark(dest, "DONE" if rc == 0 else "FAIL", result)
    return result


def run_skillopt_job(args, *, seed: int, smoke: bool) -> dict:
    split_dir = Path(args.split)
    tag = f"skillopt_s{seed}" + ("_smoke" if smoke else "")
    dest = OUT / "jobs" / tag
    if (dest / "DONE").exists() and not args.force:
        return json.loads((dest / "DONE").read_text())
    _lock_dir(dest)
    train_split = split_dir / ("skillopt_smoke" if smoke else "skillopt")
    if smoke:
        future_split = split_dir / "skillopt_smoke"
    else:
        future_split = dest / "future_split"
        _write_skillopt_eval_split(split_dir, "future_test", future_split)
    rc = run_skillopt_train(
        split_dir=train_split,
        out_root=dest / "train",
        llm_base=args.llm_base,
        python_bin=args.python,
        num_epochs=1 if smoke else args.skillopt_epochs,
        batch_size=4 if smoke else args.batch_size,
        workers=args.workers,
        seed=seed,
        log_path=dest / "train.log",
    )
    skill = find_best_skill(dest / "train") or Path(SKILLOPT_INITIAL)
    rc2 = run_skillopt_eval(
        split_dir=future_split,
        skill_path=skill,
        out_root=dest / "future_eval",
        llm_base=args.llm_base,
        python_bin=args.python,
        workers=args.workers,
        seed=seed,
        log_path=dest / "future_eval.log",
    )
    result = {
        "condition": "skillopt",
        "seed": seed,
        "smoke": smoke,
        "train_rc": rc,
        "eval_rc": rc2,
        "skill": str(skill),
        "parsed": _parse_skillopt_scores(dest / "future_eval"),
        "split_sha256": _load_split(split_dir)["_sha256"],
    }
    ok = rc == 0 and rc2 == 0
    _mark(dest, "DONE" if ok else "FAIL", result)
    return result


def run_memrl_job(args, *, seed: int, smoke: bool) -> dict:
    split_dir = Path(args.split)
    tag = f"memrl_s{seed}" + ("_smoke" if smoke else "")
    dest = OUT / "jobs" / tag
    if (dest / "DONE").exists() and not args.force:
        return json.loads((dest / "DONE").read_text())
    _lock_dir(dest)
    train_view = split_dir / "views" / ("smoke" if smoke else "acquisition_pool")
    # Eval during train is disabled; dummy eval view = assim_validation, never future_test.
    dummy_eval = split_dir / "views" / "assim_validation"
    future_view = split_dir / "views" / ("shadow_reward_probe" if smoke else "future_test")
    env_train = dest / "env_train.yaml"
    env_future = dest / "env_future.yaml"
    run_yaml = dest / "memrl.yaml"
    dump_yaml(env_train, memrl_env_yaml(train_view=train_view, eval_view=dummy_eval))
    dump_yaml(env_future, memrl_env_yaml(train_view=train_view, eval_view=future_view))
    dump_yaml(
        run_yaml,
        memrl_run_yaml(
            env_yaml=env_train,
            llm_base=args.llm_base,
            embed_base=args.embed_base,
            seed=seed,
            out_dir=dest / "run",
            num_sections=1 if smoke else args.memrl_sections,
            batch_size=2 if smoke else args.batch_size,
            max_steps=args.max_steps,
            valid_interval=0,
            experiment_name=tag,
        ),
    )
    prev = os.getcwd()
    os.chdir(dest)
    try:
        train_summary = run_memrl_official(config_yaml=run_yaml, env_yaml=env_train, work_dir=dest / "train")
    finally:
        os.chdir(prev)
    snapshots = sorted((dest / "train" / "local_cache").glob("snapshot/*"))
    snapshot = snapshots[-1] if snapshots else None
    future = evaluate_memrl_checkpoint(
        config_yaml=run_yaml,
        env_yaml=env_future,
        work_dir=dest / "future_eval",
        snapshot=snapshot,
    )
    result = {
        "condition": "memrl",
        "seed": seed,
        "smoke": smoke,
        "train": train_summary,
        "future": future,
        "snapshot": str(snapshot) if snapshot else None,
        "split_sha256": _load_split(split_dir)["_sha256"],
    }
    _mark(dest, "DONE", result)
    return result


def run_none_job(args, *, seed: int, smoke: bool) -> dict:
    split_dir = Path(args.split)
    tag = f"none_s{seed}" + ("_smoke" if smoke else "")
    dest = OUT / "jobs" / tag
    if (dest / "DONE").exists() and not args.force:
        return json.loads((dest / "DONE").read_text())
    _lock_dir(dest)
    if smoke:
        future_split = split_dir / "skillopt_smoke"
    else:
        future_split = dest / "future_split"
        _write_skillopt_eval_split(split_dir, "future_test", future_split)
    rc = run_skillopt_eval(
        split_dir=future_split,
        skill_path=Path(SKILLOPT_INITIAL),
        out_root=dest / "eval",
        llm_base=args.llm_base,
        python_bin=args.python,
        workers=args.workers,
        seed=seed,
        log_path=dest / "eval.log",
    )
    result = {
        "condition": "none",
        "seed": seed,
        "smoke": smoke,
        "returncode": rc,
        "parsed": _parse_skillopt_scores(dest / "eval"),
        "split_sha256": _load_split(split_dir)["_sha256"],
    }
    _mark(dest, "DONE" if rc == 0 else "FAIL", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, default=OUT / "split")
    p.add_argument("--conditions", default="no_update,memrl,skillopt")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--python", default=str(EXROOT / ".venv" / "bin" / "python"))
    p.add_argument("--llm_base", default=f"http://127.0.0.1:{VLLM_PORT}/v1")
    p.add_argument("--embed_base", default=f"http://127.0.0.1:{EMBED_PORT}/v1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--memrl_sections", type=int, default=2)
    p.add_argument("--skillopt_epochs", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--foundation_only", action="store_true")
    p.add_argument("--job", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip_health", action="store_true")
    args = p.parse_args()

    if not args.skip_health:
        _wait_http(args.llm_base.replace("/v1", "/v1/models"), timeout=900)
        _wait_http(args.embed_base.replace("/v1", "/health"), timeout=120)

    if args.foundation_only or args.job == "foundation":
        print(json.dumps(run_foundation(args), indent=2))
        return
    if args.job.startswith("memrl"):
        m = re.search(r"_s(\d+)(?:_|$)", args.job)
        seed = int(m.group(1)) if m else 0
        print(json.dumps(run_memrl_job(args, seed=seed, smoke=args.smoke), indent=2))
        return
    if args.job.startswith("skillopt"):
        m = re.search(r"_s(\d+)(?:_|$)", args.job)
        seed = int(m.group(1)) if m else 0
        print(json.dumps(run_skillopt_job(args, seed=seed, smoke=args.smoke), indent=2))
        return
    if args.job.startswith("none"):
        m = re.search(r"_s(\d+)(?:_|$)", args.job)
        seed = int(m.group(1)) if m else 0
        print(json.dumps(run_none_job(args, seed=seed, smoke=args.smoke), indent=2))
        return

    # Sequential driver for a single process; supervise script parallelizes --job.
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.smoke:
        seeds = seeds[:1]
    mapping = {
        "no_update": run_none_job,
        "none": run_none_job,
        "memrl": run_memrl_job,
        "skillopt": run_skillopt_job,
    }
    results = []
    if not args.smoke:
        results.append(run_foundation(args))
    for cond in conds:
        fn = mapping[cond]
        for seed in seeds:
            results.append(fn(args, seed=seed, smoke=args.smoke))
    (OUT / "driver_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(results)}, indent=2))


if __name__ == "__main__":
    main()
