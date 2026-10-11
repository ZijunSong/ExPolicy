"""Shared helpers for R5.1 robustness tracks. Reuse R5 loaders / assimilator / eval."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from expolicy.acquisition.base import FrontierAcquisition, make_acquisition
from expolicy.assimilation.carriers.skill_store import (
    Skill,
    SkillStore,
    run_skill_on_task,
)
from expolicy.assimilation.skill_assimilator import (
    COMPILER_VERSION,
    PRIMITIVES,
    SkillAssimilator,
    _stable_id,
    maps_to_skill_capability,
    skill_has_leakage,
)
from expolicy.core.experience import Experience
from expolicy.core.state import empty_agent
from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.experiments.common import ROOT, load_world, mean, score_with_retrieval, write_json
from expolicy.experiments.r_common import load_e0_experiences
from expolicy.logging.event_logger import append_jsonl

R5_ROOT = ROOT / "outputs" / "r5_acquisition_skill"
R51_ROOT = ROOT / "outputs" / "r5_1_robustness"
TARGET_CAPS = {
    "s1_multi_hop": "multi_hop",
    "s2_temporal": "temporal_reasoning",
    "s4_cross_doc": "cross_document_synthesis",
}
TARGET_CAP_LIST = tuple(TARGET_CAPS.values())
REFIT_AT = (8, 16, 24, 32, 40, 48)
EVAL_AT = (0, 16, 32, 48)
ROUTER_LABELS = ("none", "multi_hop", "temporal_reasoning", "cross_document_synthesis")
SKILL_ORDER = ("multi_hop", "temporal_reasoning", "cross_document_synthesis")
TASK_ID_RE = re.compile(r"\bt_[a-z]+_[a-z0-9]+_\d+\b", re.I)
STAGE_RE = re.compile(r"\bstage_[0-4]\b", re.I)
FROZEN_SPLIT_HASH = {
    0: "61753eee02a792b27d21b5d05ced9c68e976e5b3645bc48477dfaa021ab645db",
    1: "710a123cc92fa1f0b7f92a8ed80eebcd2aa6c1e93fe4b0fb1cd70e14f2a10e78",
    2: "94c9e74b5aa26b1cefa136ebe1304ed83e93be96ae78b9b05c64be12c41d2962",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def heartbeat(out: Path, **payload: Any) -> None:
    payload = {"updated_at": time.time(), **payload}
    write_json(out / "STATUS_LIVE.json", payload)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    names = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in names})


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_script(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assert_not_r5_output(path: Path) -> None:
    resolved = path.resolve()
    r5 = R5_ROOT.resolve()
    if resolved == r5 or r5 in resolved.parents or resolved == r5:
        raise RuntimeError(f"refusing to write into original R5 output: {path}")


def verify_r5_frozen(r5_root: Path) -> dict[str, Any]:
    audit = load_json(r5_root / "R5_CAUSAL_AUDIT.json")
    if audit.get("decision") != "PASS":
        raise RuntimeError(f"R5_CAUSAL_AUDIT is not PASS: {audit.get('decision')}")
    split_hashes = {}
    for seed, expected in FROZEN_SPLIT_HASH.items():
        path = r5_root / "splits" / f"seed_{seed}.json"
        got = sha256_file(path)
        if got != expected:
            raise RuntimeError(f"R5 split seed={seed} hash changed: {got} != {expected}")
        split_hashes[seed] = got
        split = load_json(path)
        fut = set(split["future_probe_ids"])
        ret = set(split["retention_probe"])
        acq = set(split["acquisition_pool"])
        if fut & acq or fut & ret or acq & ret:
            raise RuntimeError(f"R5 split seed={seed} has task-id overlap")
    for method in ("frontier", "passive"):
        for seed in (0, 1, 2):
            done = r5_root / method / f"seed_{seed}" / "DONE"
            if not done.exists():
                raise RuntimeError(f"missing original R5 run {done}")
    return {"audit": "PASS", "split_hashes": split_hashes}


def load_split(root: Path, seed: int, *, r5_root: Path = R5_ROOT) -> dict:
    for candidate in (root / "splits" / f"seed_{seed}.json", r5_root / "splits" / f"seed_{seed}.json"):
        if candidate.exists():
            return load_json(candidate)
    raise FileNotFoundError(f"split for seed {seed} not found under {root} or {r5_root}")


def split_overlap_report(split: dict) -> dict[str, int]:
    fut = set(split["future_probe_ids"])
    ret = set(split["retention_probe"])
    acq = set(split["acquisition_pool"])
    return {
        "future_acq": len(fut & acq),
        "future_ret": len(fut & ret),
        "acq_ret": len(acq & ret),
        "n_future": len(fut),
        "n_retention": len(ret),
        "n_acquisition": len(acq),
    }


def tasks_by_ids(world, ids: Sequence[str]):
    return [world.tasks[i] for i in ids if i in world.tasks]


def future_by_cap(world, split: dict) -> dict[str, list]:
    return {cap: tasks_by_ids(world, split["future_probe"][bucket]) for bucket, cap in TARGET_CAPS.items()}


def load_acquired(run_dir: Path) -> list[Experience]:
    traj = run_dir / "trajectories.jsonl"
    rows: list[Experience] = []
    if traj.exists():
        for line in traj.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("run_id") != "acquire":
                continue
            rows.append(Experience.model_validate(payload))
        if rows:
            return rows
    selected = run_dir / "selected_experiences.jsonl"
    if selected.exists():
        for line in selected.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(Experience.model_validate(json.loads(line)))
    return rows


def load_base_metrics(r5_root: Path, seed: int, fallback_root: Path | None = None) -> dict:
    for root in (fallback_root, r5_root):
        if root is None:
            continue
        path = root / "base" / f"seed_{seed}" / "final_metrics.json"
        if path.exists():
            return load_json(path)
    raise FileNotFoundError(f"base metrics missing for seed {seed}")


def load_base_rows(r5_root: Path, seed: int, fallback_root: Path | None = None) -> dict[str, dict]:
    for root in (fallback_root, r5_root):
        if root is None:
            continue
        path = root / "base" / f"seed_{seed}" / "trajectories.jsonl"
        if not path.exists():
            continue
        rows: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            tid = payload.get("task_id")
            if tid:
                rows[str(tid)] = payload
        if rows:
            return rows
    raise FileNotFoundError(f"base trajectories missing for seed {seed}")


def first_activation(run_dir: Path) -> dict[str, int]:
    path = run_dir / "skill_activation.jsonl"
    first: dict[str, int] = {}
    if not path.exists():
        metrics = run_dir / "final_metrics.json"
        if metrics.exists():
            return {k: int(v) for k, v in (load_json(metrics).get("first_skill_activation") or {}).items()}
        return first
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        cap = ev.get("capability")
        budget = int(ev.get("budget") or 0)
        if cap and cap not in first:
            first[str(cap)] = budget
    return first


def prefix_acquired(acquired: Sequence[Experience], n: int) -> list[Experience]:
    return list(acquired[: max(0, n)])


def cap_counts(acquired: Sequence[Experience]) -> dict[str, int]:
    counts: dict[str, int] = Counter()
    for exp in acquired:
        mapped = maps_to_skill_capability(exp.capability) or (exp.capability or "unknown")
        counts[mapped] += 1
        counts[exp.capability or "unknown"] += 1
    return dict(counts)


def matched_count(acquired: Sequence[Experience], cap: str) -> int:
    return sum(1 for e in acquired if maps_to_skill_capability(e.capability) == cap)


def count_only_skill(cap: str, *, n_support: int, min_support: int, version: int, probe_ids, gold_docs) -> Skill:
    skill = PRIMITIVES[cap]()
    skill.status = "active"
    skill.executable = True
    skill.n_support = int(n_support)
    skill.version = int(version)
    skill.compiler_version = COMPILER_VERSION
    skill.runtime_config = {"primitive_library": "r2_frozen_v1", "fit_mode": "count_only", "n_support_used": n_support}
    skill.activation_reason = f"count_only n_support={n_support}>={min_support}; capability={cap}; no content read"
    skill.provenance = [f"compiler:{COMPILER_VERSION}", f"primitive:{skill.name}", "fit_mode:count_only"]
    skill.source_experience_ids = []
    skill.skill_id = _stable_id(
        {"compiler": COMPILER_VERSION, "capability": cap, "version": version, "fit_mode": "count_only", "n_support": n_support}
    )
    leaks = skill_has_leakage(skill, probe_ids=probe_ids, gold_doc_ids=gold_docs)
    if leaks:
        raise RuntimeError(f"count_only skill leaked: {leaks[:8]}")
    return skill


def _pool_experiences(
    *,
    r5_root: Path,
    seed: int,
    blocked: set[str],
    exclude_task_ids: set[str],
) -> list[Experience]:
    rows: list[Experience] = []
    seen: set[str] = set()
    for exp in load_e0_experiences():
        if exp.task_id in blocked or exp.task_id in exclude_task_ids:
            continue
        key = exp.experience_id or f"e0:{exp.task_id}"
        if key in seen:
            continue
        seen.add(key)
        rows.append(exp)
    for method in ("frontier", "passive", "uncertainty", "failure_replay", "reward_seeking"):
        run_dir = r5_root / method / f"seed_{seed}"
        if not (run_dir / "DONE").exists():
            continue
        for exp in load_acquired(run_dir):
            if exp.task_id in blocked or exp.task_id in exclude_task_ids:
                continue
            key = exp.experience_id or f"{method}:{exp.task_id}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(exp)
    return rows


def sample_experiences(
    pool: Sequence[Experience],
    n: int,
    rng: random.Random,
    *,
    cap: str | None = None,
    require_match: bool = True,
) -> list[Experience]:
    if cap is None:
        cand = list(pool)
    elif require_match:
        cand = [e for e in pool if maps_to_skill_capability(e.capability) == cap]
    else:
        cand = [e for e in pool if maps_to_skill_capability(e.capability) != cap]
    if not cand:
        cand = list(pool)
    rng.shuffle(cand)
    if len(cand) >= n:
        return cand[:n]
    if not cand:
        return []
    out = list(cand)
    while len(out) < n:
        out.append(cand[len(out) % len(cand)])
    return out


def fit_store_for_mode(
    acquired: Sequence[Experience],
    *,
    mode: str,
    min_support: int,
    activation_mask: dict[str, int],
    probe_ids: Sequence[str],
    gold_docs: Sequence[str],
    skill_dir: Path,
    seed: int,
    method: str,
    r5_root: Path,
    blocked: set[str],
    budget: int,
) -> tuple[SkillStore, list[dict]]:
    skill_dir.mkdir(parents=True, exist_ok=True)
    store = SkillStore(skill_dir / "store")
    assimilator = SkillAssimilator(min_support=1)  # mask is frozen; do not re-gate on count
    events: list[dict] = []
    acquired_ids = {e.task_id for e in acquired if e.task_id}
    pool = _pool_experiences(r5_root=r5_root, seed=seed, blocked=blocked, exclude_task_ids=acquired_ids)
    rng = random.Random(50_000 + seed * 17 + sum(ord(c) for c in method + mode) % 10_000)
    versions: dict[str, int] = {}
    for cap, first_at in sorted(activation_mask.items()):
        if first_at > budget:
            continue
        n_support = matched_count(acquired, cap)
        versions[cap] = versions.get(cap, 0) + 1
        skill = None
        used_ids: list[str] = []
        if mode == "real":
            skill = SkillAssimilator(min_support=min_support).fit(
                acquired,
                cap,
                version=versions[cap],
                probe_ids=probe_ids,
                gold_doc_ids=gold_docs,
            )
        elif mode == "count_only":
            skill = count_only_skill(
                cap,
                n_support=max(n_support, min_support),
                min_support=min_support,
                version=versions[cap],
                probe_ids=probe_ids,
                gold_docs=gold_docs,
            )
        elif mode == "within_cap_shuffle":
            batch = sample_experiences(pool, max(n_support, min_support), rng, cap=cap, require_match=True)
            if len(batch) < min_support:
                batch = sample_experiences(list(acquired), max(n_support, min_support), rng, cap=cap, require_match=True)
            used_ids = [e.experience_id for e in batch]
            skill = assimilator.fit(
                batch,
                cap,
                version=versions[cap],
                probe_ids=probe_ids,
                gold_doc_ids=gold_docs,
                require_capability_match=True,
            )
        elif mode == "cross_cap_content":
            batch = sample_experiences(pool, max(n_support, min_support), rng, cap=cap, require_match=False)
            used_ids = [e.experience_id for e in batch]
            skill = assimilator.fit(
                batch,
                cap,
                version=versions[cap],
                probe_ids=probe_ids,
                gold_doc_ids=gold_docs,
                require_capability_match=False,
            )
        else:
            raise ValueError(f"unknown fit mode {mode}")
        if skill is None:
            continue
        store.install(skill)
        path = skill_dir / f"{mode}_v{versions[cap]:03d}_{cap}.json"
        write_json(path, skill.model_dump(mode="json"))
        events.append(
            {
                "mode": mode,
                "capability": cap,
                "budget": budget,
                "activation_checkpoint": first_at,
                "skill_id": skill.skill_id,
                "n_support": skill.n_support,
                "runtime_config": skill.runtime_config,
                "source_experience_ids": skill.source_experience_ids or used_ids,
                "path": str(path),
            }
        )
    return store, events


def replay_threshold_store(
    acquired: Sequence[Experience],
    *,
    min_support: int,
    budget: int,
    probe_ids: Sequence[str],
    gold_docs: Sequence[str],
    skill_dir: Path,
) -> tuple[SkillStore, list[dict], dict[str, int]]:
    skill_dir.mkdir(parents=True, exist_ok=True)
    store = SkillStore(skill_dir / "store")
    assimilator = SkillAssimilator(min_support=min_support)
    versions: dict[str, int] = {}
    first: dict[str, int] = {}
    events: list[dict] = []
    subset = prefix_acquired(acquired, budget)
    for n in REFIT_AT:
        if n > budget:
            break
        chunk = prefix_acquired(acquired, n)
        for cap in TARGET_CAP_LIST:
            matched_n = matched_count(chunk, cap)
            if matched_n < min_support:
                continue
            versions[cap] = versions.get(cap, 0) + 1
            skill = assimilator.fit(
                chunk,
                cap,
                version=versions[cap],
                probe_ids=probe_ids,
                gold_doc_ids=gold_docs,
            )
            if skill is None:
                continue
            store.install(skill)
            if cap not in first:
                first[cap] = n
            events.append(
                {
                    "budget": n,
                    "capability": cap,
                    "n_support": skill.n_support,
                    "version": skill.version,
                    "skill_id": skill.skill_id,
                    "min_support": min_support,
                }
            )
    return store, events, first


def choose_skill(
    task,
    store: SkillStore | None,
    *,
    dispatch: str,
    router: "TextRouter | None" = None,
) -> Skill | None:
    if store is None or dispatch in {"no_skill", "disabled"}:
        return None
    if dispatch in {"oracle", "oracle_capability", "oracle_dispatch"}:
        return store.executable_for(task.capability)
    if dispatch in {"text_router", "text_router_dispatch"}:
        if router is None:
            raise RuntimeError("text router dispatch requires a fitted router")
        pred = router.predict_one(task.question)
        if pred == "none":
            return None
        return store.executable_for(pred)
    if dispatch in {"all_skills_fixed_order", "all_skills"}:
        return None  # handled in eval loop
    raise ValueError(f"unknown dispatch {dispatch}")


def eval_with_dispatch(
    world,
    future_map: dict[str, list],
    retention: Sequence,
    *,
    store: SkillStore | None,
    base_rows: dict[str, dict],
    dispatch: str,
    router: "TextRouter | None" = None,
    out: Path | None = None,
) -> dict[str, Any]:
    env = SearchEnvironment(world)
    future_rows: list[dict] = []
    retention_rows: list[dict] = []
    n_skill = 0
    n_base = 0
    s0_fp = 0

    def one(task, split: str) -> dict:
        nonlocal n_skill, n_base, s0_fp
        skill = None
        if dispatch in {"all_skills_fixed_order", "all_skills"} and store is not None:
            for cap in SKILL_ORDER:
                cand = store.executable_for(cap)
                if cand is None:
                    continue
                exp = run_skill_on_task(cand, env, task)
                row = score_with_retrieval(exp, task)
                if str(exp.predicted_answer or "").strip():
                    skill = cand
                    row["dispatch_source"] = f"skill:{cand.capability}"
                    row["predicted"] = exp.predicted_answer
                    row["gold"] = task.gold_answer
                    row["task_id"] = task.task_id
                    row["capability"] = task.capability
                    row["split"] = split
                    n_skill += 1
                    if split == "retention":
                        s0_fp += 1
                    return row
        else:
            skill = choose_skill(task, store, dispatch=dispatch, router=router)
        if skill is not None:
            exp = run_skill_on_task(skill, env, task)
            row = score_with_retrieval(exp, task)
            row["dispatch_source"] = f"skill:{skill.capability}"
            row["predicted"] = exp.predicted_answer
            n_skill += 1
            if split == "retention":
                s0_fp += 1
        else:
            fallback = dict(base_rows.get(task.task_id) or {})
            if not fallback:
                raise RuntimeError(f"missing base row for {task.task_id}")
            row = dict(fallback)
            row["dispatch_source"] = "base_llm"
            n_base += 1
        row["task_id"] = task.task_id
        row["capability"] = task.capability
        row["gold"] = task.gold_answer
        row["split"] = split
        if out is not None:
            heartbeat(out, phase=f"eval_{split}", task_id=task.task_id)
        return row

    per_cap: dict[str, dict] = {}
    for cap, tasks in future_map.items():
        rows = [one(task, "future") for task in tasks]
        future_rows.extend(rows)
        per_cap[cap] = {
            "n": len(rows),
            "em": mean(float(r.get("em_strict") or r.get("em") or 0.0) for r in rows),
            "n_skill": sum(1 for r in rows if str(r.get("dispatch_source") or "").startswith("skill")),
        }
    for task in retention:
        retention_rows.append(one(task, "retention"))
    future_em = mean(float(r.get("em_strict") or r.get("em") or 0.0) for r in future_rows)
    retention_em = mean(float(r.get("em_strict") or r.get("em") or 0.0) for r in retention_rows)
    return {
        "future_em": future_em,
        "retention_em": retention_em,
        "future_per_cap": per_cap,
        "future_rows": future_rows,
        "retention_rows": retention_rows,
        "n_skill_invocations": n_skill,
        "n_base_fallback": n_base,
        "s0_false_positive": s0_fp,
        "s0_fp_rate": (s0_fp / len(retention_rows)) if retention_rows else 0.0,
        "n_active_skills": 0
        if store is None
        else sum(1 for s in store.skills.values() if s.status == "active" and s.executable),
    }


def bootstrap_ci(xs: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    xs = [float(x) for x in xs]
    if not xs:
        return float("nan"), float("nan")
    stats = []
    for _ in range(n):
        sample = [xs[rng.randrange(len(xs))] for _ in xs]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    lo = stats[int(alpha / 2 * n)]
    hi = stats[min(n - 1, int((1 - alpha / 2) * n))]
    return lo, hi


def mean_std(xs: Sequence[float]) -> tuple[float, float]:
    xs = [float(x) for x in xs]
    if not xs:
        return float("nan"), float("nan")
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return mu, math.sqrt(var)


def entropy_from_counts(counts: dict[str, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        ent -= p * math.log(p + 1e-12)
    return ent


def auc_over_budget(points: Sequence[tuple[int, float]]) -> float:
    pts = sorted((int(b), float(y)) for b, y in points)
    if len(pts) < 2:
        return 0.0
    area = 0.0
    for (b0, y0), (b1, y1) in zip(pts, pts[1:]):
        area += (b1 - b0) * (y0 + y1) / 2.0
    width = pts[-1][0] - pts[0][0]
    return area / width if width else 0.0


def question_leakage_issues(text: str) -> list[str]:
    issues = []
    if TASK_ID_RE.search(text or ""):
        issues.append("task_id_in_question")
    if STAGE_RE.search(text or ""):
        issues.append("stage_in_question")
    return issues


class TextRouter:
    """TF-IDF + multinomial logistic regression over question text only."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.vectorizer = None
        self.clf = None
        self.labels = list(ROUTER_LABELS)

    @staticmethod
    def label_of(capability: str | None) -> str:
        mapped = maps_to_skill_capability(capability)
        return mapped if mapped in TARGET_CAP_LIST else "none"

    def fit(self, questions: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split

        leak = []
        for q in questions:
            leak.extend(question_leakage_issues(q))
        x = list(questions)
        y = list(labels)
        counts = Counter(y)
        can_stratify = all(v >= 2 for v in counts.values()) and len(set(y)) > 1
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.2, random_state=self.seed, stratify=y if can_stratify else None
        )
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=8000)
        xt = self.vectorizer.fit_transform(x_train)
        self.clf = LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            random_state=self.seed,
            solver="lbfgs",
        )
        self.clf.fit(xt, y_train)
        pred_val = self.clf.predict(self.vectorizer.transform(x_val))
        # refit on all labeled training-side text for downstream
        x_all = self.vectorizer.fit_transform(x)
        self.clf.fit(x_all, y)
        vocab = list(self.vectorizer.get_feature_names_out())
        vocab_leaks = [t for t in vocab if TASK_ID_RE.search(t) or t.startswith("t_") or "stage_" in t]
        labels_sorted = [str(x) for x in self.clf.classes_]
        cm = confusion_matrix(y_val, pred_val, labels=list(self.clf.classes_))
        per_class_recall = {}
        for i, lab in enumerate(labels_sorted):
            denom = int(cm[i].sum()) if i < cm.shape[0] else 0
            per_class_recall[lab] = float(cm[i, i] / denom) if denom else 0.0
        return {
            "n_train": len(x_train),
            "n_val": len(x_val),
            "accuracy": float(accuracy_score(y_val, pred_val)),
            "macro_f1": float(f1_score(y_val, pred_val, average="macro", zero_division=0)),
            "labels": labels_sorted,
            "confusion": cm.tolist(),
            "per_class_recall": per_class_recall,
            "label_counts": dict(counts),
            "question_leakage": leak,
            "vocab_leakage": vocab_leaks,
            "n_features": len(vocab),
            "leakage_pass": not leak and not vocab_leaks,
        }

    def predict_one(self, question: str) -> str:
        if self.vectorizer is None or self.clf is None:
            raise RuntimeError("router not fitted")
        pred = self.clf.predict(self.vectorizer.transform([question]))[0]
        return str(pred)


def train_router_for_split(world, split: dict, *, seed: int) -> tuple[TextRouter, dict[str, Any]]:
    pool = tasks_by_ids(world, split["acquisition_pool"])
    questions = [t.question for t in pool]
    labels = [TextRouter.label_of(t.capability) for t in pool]
    extra_issues = []
    for t in pool:
        if t.task_id and t.task_id in (t.question or ""):
            extra_issues.append(t.task_id)
    router = TextRouter(seed=seed)
    metrics = router.fit(questions, labels)
    metrics["task_id_in_question_examples"] = extra_issues[:8]
    metrics["leakage_pass"] = bool(metrics.get("leakage_pass")) and not extra_issues
    return router, metrics


def frontier_pool_audit(world, split: dict, *, seed: int) -> dict[str, Any]:
    pool = tasks_by_ids(world, split["acquisition_pool"])
    blocked = set(split["future_probe_ids"]) | set(split["retention_probe"])
    e0 = [e for e in load_e0_experiences() if e.task_id not in blocked]
    diagnoser = CapabilityDiagnoser()
    profile = diagnoser.diagnose(e0)
    dummy = empty_agent(ROOT / "outputs" / "r5_1_robustness" / "_dummy_audit", model_name="audit")
    policy = FrontierAcquisition()
    props = policy.propose(dummy, profile, pool, len(pool), recent=e0)
    scores = props[0].scores or {}
    ranked = sorted(pool, key=lambda t: float(scores.get(t.task_id, 0.0)), reverse=True)
    by_cap: dict[str, list[float]] = defaultdict(list)
    for task in pool:
        by_cap[task.capability or "unknown"].append(float(scores.get(task.task_id, 0.0)))
    temporal = [t for t in pool if t.capability == "temporal_reasoning"]
    temporal_ranks = [i + 1 for i, t in enumerate(ranked) if t.capability == "temporal_reasoning"]
    n = len(ranked)

    def n_temporal_in_top(k: int) -> int:
        return sum(1 for t in ranked[:k] if t.capability == "temporal_reasoning")

    percentiles = []
    for rank in temporal_ranks:
        percentiles.append(1.0 - (rank - 1) / max(n - 1, 1))
    cap_counts_pool = dict(Counter(t.capability or "unknown" for t in pool))
    return {
        "seed": seed,
        "n_pool": n,
        "pool_by_cap": cap_counts_pool,
        "score_stats_by_cap": {
            cap: {
                "n": len(vals),
                "mean": mean(vals),
                "std": mean_std(vals)[1],
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
            }
            for cap, vals in sorted(by_cap.items())
        },
        "n_temporal_in_pool": len(temporal),
        "temporal_best_rank": min(temporal_ranks) if temporal_ranks else None,
        "temporal_worst_rank": max(temporal_ranks) if temporal_ranks else None,
        "temporal_score_percentiles": percentiles,
        "temporal_mean_percentile": mean(percentiles) if percentiles else None,
        "temporal_in_top_48": n_temporal_in_top(48),
        "temporal_in_top_64": n_temporal_in_top(64),
        "temporal_in_top_96": n_temporal_in_top(96),
        "would_reach_8_temporal_at_budget_48": n_temporal_in_top(48) >= 8,
        "would_reach_8_temporal_at_budget_64": n_temporal_in_top(64) >= 8,
        "would_reach_8_temporal_at_budget_96": n_temporal_in_top(96) >= 8,
    }


def ensure_seed_splits(out: Path, seeds: Sequence[int], *, r5_root: Path = R5_ROOT) -> dict[str, Any]:
    """Build splits for new seeds only. Copy frozen R5 0-2 without rewriting them in R5."""
    assert_not_r5_output(out)
    split_dir = out / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    world = load_world()
    builder = load_script("build_r5_splits", ROOT / "scripts" / "build_r5_splits.py")
    report = {"copied_frozen": [], "built": [], "hashes": {}, "overlap": {}}
    for seed in seeds:
        dest = split_dir / f"seed_{seed}.json"
        if seed in FROZEN_SPLIT_HASH:
            src = r5_root / "splits" / f"seed_{seed}.json"
            data = src.read_bytes()
            if sha256_file(src) != FROZEN_SPLIT_HASH[seed]:
                raise RuntimeError(f"refusing to copy mutated R5 split seed={seed}")
            if dest.exists() and sha256_file(dest) != FROZEN_SPLIT_HASH[seed]:
                raise RuntimeError(f"existing r5.1 split seed={seed} does not match frozen R5 hash")
            if not dest.exists():
                dest.write_bytes(data)
            report["copied_frozen"].append(seed)
        elif not dest.exists():
            row = builder.build_seed_split(world, seed)
            write_json(dest, row)
            report["built"].append(seed)
        split = load_json(dest)
        ov = split_overlap_report(split)
        if ov["future_acq"] or ov["future_ret"] or ov["acq_ret"]:
            raise RuntimeError(f"split overlap seed={seed}: {ov}")
        report["overlap"][str(seed)] = ov
        report["hashes"][str(seed)] = sha256_file(dest)
    write_json(split_dir / "manifest.json", {"seeds": list(seeds), **report})
    return report
