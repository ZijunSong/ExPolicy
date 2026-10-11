"""R5.3 external generalization helpers: public search, frozen frontier, CountOnly K=8."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from expolicy.acquisition.ace_skill import AceSkillPrioritizedAcquisition
from expolicy.acquisition.base import make_acquisition
from expolicy.agents.llm_agent import LLMSearchAgent
from expolicy.assimilation.carriers.skill_store import SkillStore
from expolicy.core.experience import Experience
from expolicy.core.state import HarnessConfig, empty_agent
from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
from expolicy.environments.public_search.capability_map import PUBLIC_CAPS
from expolicy.environments.public_search.environment import PUBLIC_SYSTEM_PROMPT, PublicSearchEnvironment, PublicWorld
from expolicy.environments.public_search.evaluator import score_public
from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.environments.public_search.skills import (
    PRIMITIVES,
    count_only_public_skill,
    run_public_skill_on_task,
)
from expolicy.environments.search.world import Task
from expolicy.experiments.common import ROOT, mean, write_json
from expolicy.experiments.r_common import apply_gpu, write_run_header
from expolicy.logging.event_logger import EventLogger, append_jsonl
from expolicy.models.factory import build_backend

R53_ROOT = ROOT / "outputs" / "r5_3_external_generalization"
DEFAULT_MODEL = "/data/ppnm/models/Qwen2.5-3B-Instruct"
DEFAULT_EMBEDDER = "/data/ppnm/models/all-MiniLM-L6-v2"
MIN_SUPPORT = 8
ROUND_SIZE = 8
BUDGET_DEFAULT = 48
REFIT_AT = (8, 16, 24, 32, 40, 48)
FULL_EVAL_AT = (0, 16, 32, 48)
DATASETS = ("hotpotqa", "2wiki", "musique")
PRIMARY_METHODS = ("frontier", "ace_skill_prioritized", "passive")


def heartbeat(out: Path, **payload: Any) -> None:
    payload = {"updated_at": time.time(), "pid": os.getpid(), **payload}
    write_json(out / "STATUS_LIVE.json", payload)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(root: Path) -> dict[str, Any]:
    path = root / "splits" / "split.json"
    if not path.exists():
        raise FileNotFoundError(f"frozen split missing: {path}")
    return load_json(path)


def records_to_tasks(records: Sequence[dict[str, Any]], *, split: str) -> list[Task]:
    from expolicy.environments.public_search.datasets import to_task

    tasks = []
    for rec in records:
        cap = rec.get("capability") or "unmapped"
        if cap == "unmapped":
            continue
        tasks.append(to_task(rec, capability=str(cap), split=split))
    return tasks


def world_from_tasks(tasks: Sequence[Task]) -> PublicWorld:
    world = PublicWorld()
    for task in tasks:
        world.add_task(task)
    return world


def task_index(split: dict[str, Any]) -> dict[str, Task]:
    buckets = []
    for key in ("calibration", "acquisition_pool", "skill_validation", "future_probe", "bamboogle"):
        buckets.extend(split.get(key) or [])
    tasks = records_to_tasks(buckets, split="mixed")
    return {t.task_id: t for t in tasks}


def split_tasks(split: dict[str, Any], key: str) -> list[Task]:
    return records_to_tasks(split.get(key) or [], split=key)


def unique_task_ids(acquired: Sequence[Experience], cap: str) -> set[str]:
    return {str(e.task_id) for e in acquired if e.capability == cap and e.task_id}


def acquisition_distribution(acquired: Sequence[Experience]) -> dict[str, Any]:
    ids = [str(e.task_id) for e in acquired if e.task_id]
    counts = Counter(ids)
    n = len(ids)
    unique = len(counts)
    per_cap_unique: dict[str, set[str]] = {cap: set() for cap in PUBLIC_CAPS}
    per_cap_rollouts: Counter[str] = Counter()
    per_ds: Counter[str] = Counter()
    for exp in acquired:
        cap = exp.capability if exp.capability in per_cap_unique else "other"
        per_cap_rollouts[cap] += 1
        if exp.task_id and cap in per_cap_unique:
            per_cap_unique[cap].add(str(exp.task_id))
        ds = (exp.capability_context or {}).get("dataset") or ""
        if not ds and exp.task_id and ":" in str(exp.task_id):
            ds = str(exp.task_id).split(":", 1)[0]
        if ds:
            per_ds[ds] += 1
    rollouts_cap = {cap: per_cap_rollouts[cap] for cap in PUBLIC_CAPS}
    n_cap = sum(rollouts_cap.values()) or 1
    entropy = 0.0
    for c in rollouts_cap.values():
        if c <= 0:
            continue
        p = c / n_cap
        entropy -= p * __import__("math").log(p + 1e-12)
    return {
        "total_rollouts": n,
        "unique_tasks": unique,
        "duplicate_rollouts": n - unique,
        "unique_tasks_per_capability": {cap: len(per_cap_unique[cap]) for cap in PUBLIC_CAPS},
        "rollouts_per_capability": dict(per_cap_rollouts),
        "rollouts_per_dataset": dict(per_ds),
        "capability_entropy": entropy,
        "n_revisit_tasks": int(sum(1 for c in counts.values() if c > 1)),
        "max_visits": int(max(counts.values()) if counts else 0),
    }


def activation_times(acquired: Sequence[Experience], *, k: int = MIN_SUPPORT) -> dict[str, int]:
    first: dict[str, int] = {}
    seen: dict[str, set[str]] = {cap: set() for cap in PUBLIC_CAPS}
    for i, exp in enumerate(acquired, 1):
        cap = exp.capability if exp.capability in seen else None
        if not cap or not exp.task_id:
            continue
        seen[cap].add(str(exp.task_id))
        if cap not in first and len(seen[cap]) >= k:
            first[cap] = i
    return first


def target_skill_times(first: dict[str, int], *, budget: int = 48) -> dict[str, Any]:
    ordered = sorted(int(v) for v in first.values())

    def fmt(val: int | None):
        return val if val is not None else f">{budget}"

    return {
        "budget_to_first_skill": fmt(ordered[0] if ordered else None),
        "budget_to_second_skill": fmt(ordered[1] if len(ordered) > 1 else None),
        "budget_to_third_skill": fmt(ordered[2] if len(ordered) > 2 else None),
        "n_activated_target_skills": len(ordered),
        "first_by_cap": dict(first),
    }


def count_only_store(acquired: Sequence[Experience], *, min_support: int, skill_dir: Path, budget: int | None = None) -> tuple[SkillStore, dict[str, int]]:
    import shutil

    skill_dir.mkdir(parents=True, exist_ok=True)
    store_path = skill_dir / ("store" if budget is None else f"store_b{budget}")
    if store_path.exists():
        shutil.rmtree(store_path)
    store = SkillStore(store_path)
    subset = list(acquired if budget is None else acquired[:budget])
    first = activation_times(subset, k=min_support)
    for cap in PUBLIC_CAPS:
        n_unique = len(unique_task_ids(subset, cap))
        if n_unique < min_support:
            continue
        skill = count_only_public_skill(cap, n_support=n_unique, min_support=min_support, version=1)
        store.install(skill)
        write_json(skill_dir / f"count_only_{cap}.json", skill.model_dump(mode="json"))
    return store, first


def build_public_agent(
    world: PublicWorld,
    *,
    model_path: str,
    output_dir: Path,
    retriever_url: str,
    top_k: int = 5,
    max_steps: int = 8,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    client = RetrieverClient(retriever_url)
    env = PublicSearchEnvironment(world, client, top_k=top_k)
    backend = build_backend({"backend": "transformers", "path": model_path, "dtype": "bfloat16"})
    harness = HarnessConfig(max_steps=max_steps, search_top_k=top_k, min_evidence_before_answer=1, temperature=0.0)
    logger = EventLogger(output_dir / "events.jsonl")
    state = empty_agent(output_dir / "agent_v000", model_name=model_path)
    state.harness = harness
    state.write_manifest()
    agent = LLMSearchAgent(
        backend,
        env,
        harness=harness,
        logger=logger,
        system_prompt=PUBLIC_SYSTEM_PROMPT,
    )
    return agent, state, backend, env


def macro_from_rows(rows: Sequence[dict[str, Any]], *, key: str = "em") -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    by_cap: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        ds = str(row.get("dataset") or "unknown")
        cap = str(row.get("capability") or "unknown")
        val = float(row.get(key) or 0.0)
        by_ds[ds].append(val)
        by_cap[cap].append(val)
    per_dataset = {ds: mean(vs) for ds, vs in by_ds.items() if ds in DATASETS}
    macro = mean([per_dataset[ds] for ds in DATASETS if ds in per_dataset])
    return {
        "macro": macro,
        "per_dataset": per_dataset,
        "per_capability": {cap: mean(vs) for cap, vs in by_cap.items()},
        "n": len(rows),
    }


def eval_with_skills(
    tasks: Sequence[Task],
    *,
    store: SkillStore | None,
    base_rows: dict[str, dict],
    env: PublicSearchEnvironment,
    model=None,
    dispatch: str = "oracle",
    router=None,
    out: Path | None = None,
) -> dict[str, Any]:
    rows = []
    n_skill = 0
    n_base = 0
    for task in tasks:
        if out is not None:
            heartbeat(out, phase="eval", task_id=task.task_id)
        skill = None
        if store is not None:
            cap = task.capability
            if dispatch == "text_router" and router is not None:
                cap = router.predict_one(task.question)
            if dispatch == "embed_router" and router is not None:
                cap = router.predict_one(task.question)
            if cap in PUBLIC_CAPS:
                skill = store.executable_for(cap)
        if skill is not None:
            exp = run_public_skill_on_task(skill, env, task, model=model)
            row = score_public(exp, task)
            row["dispatch_source"] = f"skill:{skill.capability}"
            n_skill += 1
        else:
            fallback = dict(base_rows.get(task.task_id) or {})
            if not fallback:
                raise RuntimeError(f"missing base row for {task.task_id}")
            row = dict(fallback)
            row["dispatch_source"] = "base_llm"
            n_base += 1
        row["task_id"] = task.task_id
        row["dataset"] = (task.metadata or {}).get("dataset")
        row["capability"] = task.capability
        rows.append(row)
    summary = macro_from_rows(rows)
    summary.update(
        {
            "rows": rows,
            "n_skill_invocations": n_skill,
            "n_base_fallback": n_base,
            "n_active_skills": 0
            if store is None
            else sum(1 for s in store.skills.values() if s.status == "active" and s.executable),
        }
    )
    return summary


class TextRouter:
    def __init__(self, seed: int = 0, labels: Sequence[str] = PUBLIC_CAPS) -> None:
        self.seed = seed
        self.labels = list(labels)
        self.vectorizer = None
        self.clf = None

    def fit(self, questions: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split

        x, y = list(questions), list(labels)
        counts = Counter(y)
        can_stratify = all(v >= 2 for v in counts.values()) and len(set(y)) > 1
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.2, random_state=self.seed, stratify=y if can_stratify else None
        )
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=8000)
        self.clf = LogisticRegression(max_iter=400, class_weight="balanced", random_state=self.seed)
        self.clf.fit(self.vectorizer.fit_transform(x_train), y_train)
        pred_val = self.clf.predict(self.vectorizer.transform(x_val))
        self.clf.fit(self.vectorizer.fit_transform(x), y)
        per_cap = {}
        for cap in self.labels:
            gold = [1 if yy == cap else 0 for yy in y_val]
            pr = [1 if yy == cap else 0 for yy in pred_val]
            tp = sum(g and p for g, p in zip(gold, pr))
            fn = sum(g and not p for g, p in zip(gold, pr))
            per_cap[cap] = tp / (tp + fn) if (tp + fn) else 0.0
        return {
            "accuracy": float(accuracy_score(y_val, pred_val)),
            "macro_f1": float(f1_score(y_val, pred_val, average="macro")),
            "per_cap_recall": per_cap,
            "n_train": len(x),
        }

    def predict_one(self, question: str) -> str:
        if self.vectorizer is None or self.clf is None:
            return "none"
        pred = self.clf.predict(self.vectorizer.transform([question]))[0]
        return str(pred)


class EmbedRouter:
    def __init__(self, seed: int = 0, labels: Sequence[str] = PUBLIC_CAPS, model_path: str = DEFAULT_EMBEDDER) -> None:
        self.seed = seed
        self.labels = list(labels)
        self.model_path = model_path
        self.clf = None
        self.embedder = None

    def _encode(self, texts: Sequence[str]):
        if self.embedder is None:
            from expolicy.models.embedder import Embedder

            self.embedder = Embedder(self.model_path, device="cpu")
        return self.embedder.encode(list(texts))

    def fit(self, questions: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split

        x, y = list(questions), list(labels)
        vecs = self._encode(x)
        counts = Counter(y)
        can_stratify = all(v >= 2 for v in counts.values()) and len(set(y)) > 1
        idx = list(range(len(x)))
        tr, va = train_test_split(idx, test_size=0.2, random_state=self.seed, stratify=y if can_stratify else None)
        self.clf = LogisticRegression(max_iter=400, class_weight="balanced", random_state=self.seed)
        self.clf.fit(vecs[tr], [y[i] for i in tr])
        pred = self.clf.predict(vecs[va])
        y_val = [y[i] for i in va]
        self.clf.fit(vecs, y)
        return {
            "accuracy": float(accuracy_score(y_val, pred)),
            "macro_f1": float(f1_score(y_val, pred, average="macro")),
            "n_train": len(x),
            "backend": self.model_path,
        }

    def predict_one(self, question: str) -> str:
        if self.clf is None:
            return "none"
        pred = self.clf.predict(self._encode([question]))[0]
        return str(pred)


def train_routers(split: dict[str, Any], seed: int = 0) -> tuple[TextRouter, EmbedRouter, dict[str, Any]]:
    train_recs = list(split.get("calibration") or []) + list(split.get("acquisition_pool") or [])
    questions = [r["question"] for r in train_recs if r.get("capability") in PUBLIC_CAPS]
    labels = [r["capability"] for r in train_recs if r.get("capability") in PUBLIC_CAPS]
    text = TextRouter(seed=seed)
    embed = EmbedRouter(seed=seed)
    text_metrics = text.fit(questions, labels)
    embed_metrics = embed.fit(questions, labels)
    future = list(split.get("future_probe") or [])
    y_true = [r["capability"] for r in future]
    y_text = [text.predict_one(r["question"]) for r in future]
    y_emb = [embed.predict_one(r["question"]) for r in future]
    def acc(pred):
        return mean(float(a == b) for a, b in zip(pred, y_true)) if y_true else 0.0
    metrics = {
        "text_router": text_metrics,
        "embed_router": embed_metrics,
        "future_text_accuracy": acc(y_text),
        "future_embed_accuracy": acc(y_emb),
        "n_train": len(questions),
        "n_future": len(future),
    }
    return text, embed, metrics


def write_frontier_port_audit(out: Path) -> None:
    text = """# FRONTIER_PORT_AUDIT

R5.3 ports the frozen R5.2 acquisition / gating contract onto a public Wikipedia
search environment. No frontier coefficient was changed after seeing HotpotQA /
2Wiki / MuSiQue numbers.

## Reused unchanged

| piece | location | frozen value |
|---|---|---|
| frontier score | `expolicy/acquisition/base.py` `FrontierAcquisition.propose` | `(learnability * novelty * transfer * unc) / cost` |
| diagnoser bands / learnability / novelty | `expolicy/diagnosis/capability_profile.py` | `learnability = 4s(1-s)`; `novelty = 1 if n<8 else max(0.2, 8/n)` |
| expected_cost in profile | same | `1.0 + 0.4*(1-score)` |
| fallback cost if cap not in profile | `FrontierAcquisition` | `1.0 + 0.3 * task.hops` |
| Ace-Skill sampler | `expolicy/acquisition/ace_skill.py` | rho=0.95, gamma=0.4, epsilon=0.1, zeros init |
| gating | CountOnly unique-task | K=8 distinct task IDs |
| model | Qwen2.5-3B-Instruct | temperature 0 |
| budget / checkpoints | 48; full eval 0/16/32/48 | |

`transfer_potential` stays the Helion diagnoser rule: 0.8 only for the original
Helion names `{multi_hop, query_decomposition, source_reliability}`, else 0.45.
Public capabilities `bridge_decomposition` / `comparison_alignment` /
`long_chain_search` therefore all receive **0.45**. That is a frozen-port
consequence, not a new public-benchmark bonus.

## Environment adapters only

| replaced | adapter |
|---|---|
| capability IDs | C1/C2/C3 public taxonomy |
| calibration statistics | public calibration rollouts |
| task pool | HotpotQA / 2Wiki / MuSiQue frozen split |
| retrieval backend | Wikipedia 2018 + E5-base-v2 local server |
| executable primitives | public search control flow (no Helion schema) |

## Explicitly not done

- no frontier coefficient retune
- no MuSiQue-only sampling bonus
- no K retune
- no RealFit / CrossCap
- no LoRA/SFT assimilation sweep
- no mock retrieval / gold-doc injection
"""
    (out / "FRONTIER_PORT_AUDIT.md").write_text(text, encoding="utf-8")


def _load_jsonl_exp(path: Path, run_id: str | None = "acquire") -> list[Experience]:
    rows: list[Experience] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if run_id is not None and payload.get("run_id") and payload.get("run_id") != run_id:
            continue
        try:
            rows.append(Experience.model_validate(payload))
        except Exception:
            continue
    return rows


def load_base_rows(root: Path) -> dict[str, dict]:
    path = root / "base_eval" / "rows.jsonl"
    rows: dict[str, dict] = {}
    if not path.exists():
        # shards
        for shard in sorted((root / "base_eval").glob("shard_*.jsonl")):
            path = shard
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    tid = rec.get("task_id")
                    if tid:
                        rows[str(tid)] = rec
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            tid = rec.get("task_id")
            if tid:
                rows[str(tid)] = rec
    return rows


def run_one_seed(
    *,
    method: str,
    seed: int,
    output_dir: Path,
    model: str,
    budget: int,
    gpu: str | None,
    resume: bool,
    retriever_url: str,
    min_support: int = MIN_SUPPORT,
    dispatch: str = "oracle",
    round_size: int = ROUND_SIZE,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    apply_gpu(gpu)
    out = output_dir / method / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists() and resume:
        return {"status": "already_done", "out": str(out), "method": method, "seed": seed}

    split = load_split(output_dir)
    pool = split_tasks(split, "acquisition_pool")
    future = split_tasks(split, "future_probe")
    by_id = {t.task_id: t for t in pool}
    world = world_from_tasks(list(pool) + list(future) + split_tasks(split, "calibration"))
    base_rows = load_base_rows(output_dir)
    if not base_rows:
        raise RuntimeError("base_eval rows missing; run --phase base_eval first")
    calib_path = output_dir / "calibration" / "experiences.jsonl"
    calib_exps = _load_jsonl_exp(calib_path, run_id=None)
    write_run_header(
        out,
        config={
            "script": "run_r5_3_external.py",
            "method": method,
            "seed": seed,
            "model": model,
            "gpu": gpu,
            "budget": budget,
            "dispatch": dispatch,
            "min_support": min_support,
            "fit_mode": "count_only_unique_task",
            "retriever_url": retriever_url,
        },
    )

    acquired = _load_jsonl_exp(out / "trajectories.jsonl") if resume else []
    dummy_state = empty_agent(out / "dummy", model_name=model)
    diagnoser = CapabilityDiagnoser()
    policy = make_acquisition(method, seed=seed)
    pending_round: list[str] = []
    t_round = len(acquired) // round_size
    state_path = out / "sampler_state.json"
    if isinstance(policy, AceSkillPrioritizedAcquisition):
        policy.bind([t.task_id for t in pool])
        if resume and state_path.exists():
            payload = load_json(state_path)
            policy.load_state(payload)
            pending_round = list(payload.get("pending_round") or [])
            t_round = int(payload.get("t", t_round))
        elif acquired:
            recent_replay = list(calib_exps)
            t_round = 0
            profile = diagnoser.diagnose(recent_replay or acquired)
            for ridx in range(0, len(acquired), round_size):
                chunk = acquired[ridx : ridx + round_size]
                if len(chunk) == round_size:
                    policy.propose(dummy_state, profile, pool, len(chunk), recent=recent_replay, t=t_round)
                for exp in chunk:
                    policy.observe(exp.task_id, 1.0 if exp.success else 0.0, t_round)
                    recent_replay.append(exp)
                    profile = diagnoser.diagnose(recent_replay)
                if len(chunk) == round_size:
                    t_round += 1
            pending_round = []

    cache = cache if cache is not None else {}
    checkpoint_path = out / "checkpoint_metrics.jsonl"
    done_evals: set[int] = set()
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_evals.add(int(json.loads(line)["budget"]))
    need_llm = len(acquired) < budget
    need_eval = any(n not in done_evals for n in FULL_EVAL_AT if n <= budget)
    agent = cache.get("agent")
    base_state = cache.get("base_state")
    backend = cache.get("backend")
    env = cache.get("env")
    if (need_llm or need_eval) and agent is None:
        heartbeat(out, phase="load_model", acquired=len(acquired), budget=budget, method=method, seed=seed)
        agent, base_state, backend, env = build_public_agent(
            world, model_path=model, output_dir=out / "base_agent", retriever_url=retriever_url
        )
        cache.update({"agent": agent, "base_state": base_state, "backend": backend, "env": env})
    env = env or cache.get("env") or PublicSearchEnvironment(world, RetrieverClient(retriever_url))
    recent = list(calib_exps) + list(acquired)
    t0 = time.time()

    def save_sampler(pending: list[str]) -> None:
        if isinstance(policy, AceSkillPrioritizedAcquisition):
            payload = policy.state_dict()
            payload["t"] = t_round
            payload["pending_round"] = pending
            write_json(state_path, payload)

    remaining = list(pool)
    if method not in {"ace_skill_prioritized", "ace_skill"}:
        seen = {e.task_id for e in acquired}
        remaining = [t for t in remaining if t.task_id not in seen]

    while len(acquired) < budget:
        n = len(acquired)
        want = min(round_size, budget - n)
        profile = diagnoser.diagnose(recent) if recent else diagnoser.diagnose(calib_exps)
        if pending_round:
            batch_ids = pending_round[:want]
            pending_round = pending_round[want:]
        else:
            if method in {"ace_skill_prioritized", "ace_skill"}:
                props = policy.propose(dummy_state, profile, pool, want, recent=recent, t=t_round)
            else:
                props = policy.propose(dummy_state, profile, remaining, want, recent=recent)
            batch_ids = list(props[0].task_ids)[:want]
        save_sampler(batch_ids)
        for tid in list(batch_ids):
            task = by_id.get(tid) or next((t for t in remaining if t.task_id == tid), None)
            if task is None:
                task = remaining[0] if remaining else pool[n % len(pool)]
            heartbeat(out, phase="acquire", acquired=len(acquired), budget=budget, method=method, seed=seed, task_id=task.task_id)
            if agent is None:
                raise RuntimeError("LLM agent required for acquisition")
            agent.skill_store = None
            agent.use_executable_skill = False
            exp = agent.run_task(task, state=base_state)
            exp.run_id = "acquire"
            exp.capability_context = {**(exp.capability_context or {}), "dataset": (task.metadata or {}).get("dataset")}
            acquired.append(exp)
            recent.append(exp)
            if method not in {"ace_skill_prioritized", "ace_skill"}:
                remaining = [t for t in remaining if t.task_id != task.task_id]
            reward = 1.0 if exp.success else 0.0
            if isinstance(policy, AceSkillPrioritizedAcquisition):
                policy.observe(task.task_id, reward, t_round)
            append_jsonl(out / "trajectories.jsonl", json.loads(exp.model_dump_json()))
            append_jsonl(
                out / "selected_experiences.jsonl",
                {
                    "step": len(acquired),
                    "task_id": task.task_id,
                    "capability": task.capability,
                    "dataset": (task.metadata or {}).get("dataset"),
                    "success": exp.success,
                    "reward": reward,
                    "tokens": exp.cost.tokens,
                    "tool_calls": exp.cost.tool_calls,
                    "round": t_round,
                    "predicted": exp.predicted_answer,
                },
            )
            batch_ids.pop(0)
            save_sampler(batch_ids)
        t_round += 1
        pending_round = []
        save_sampler([])

    first_all: dict[str, int] = {}
    last = None
    base_macro = None
    # budget 0 uses base rows
    for n in (0, *REFIT_AT):
        if n > budget:
            break
        chunk = acquired[:n]
        if n == 0:
            first: dict[str, int] = {}
            store = SkillStore(out / "skill_versions" / "store_empty")
        else:
            store, first = count_only_store(chunk, min_support=min_support, skill_dir=out / "skill_versions", budget=n)
            for cap, b in first.items():
                first_all.setdefault(cap, b)
        if n in FULL_EVAL_AT and n not in done_evals:
            heartbeat(out, phase="eval", acquired=n, budget=budget, method=method, seed=seed)
            if n == 0:
                payload = eval_with_skills(future, store=None, base_rows=base_rows, env=env, model=None, dispatch=dispatch, out=out)
                base_macro = payload["macro"]
            else:
                payload = eval_with_skills(
                    future,
                    store=store,
                    base_rows=base_rows,
                    env=env,
                    model=backend,
                    dispatch=dispatch,
                    out=out,
                )
            dist = acquisition_distribution(chunk)
            row = {
                "budget": n,
                "method": method,
                "seed": seed,
                "macro": payload["macro"],
                "per_dataset": payload["per_dataset"],
                "per_capability": payload["per_capability"],
                "n_active_skills": payload["n_active_skills"],
                "n_skill_invocations": payload["n_skill_invocations"],
                "delta_future": payload["macro"] - (base_macro if base_macro is not None else payload["macro"]),
                **dist,
                "first_skill_activation": dict(first_all),
            }
            append_jsonl(checkpoint_path, row)
            done_evals.add(n)
            last = row
            write_json(out / f"eval_budget_{n}.json", {k: v for k, v in payload.items() if k != "rows"})
        elif n in FULL_EVAL_AT:
            rows = [json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            last = next((r for r in rows if int(r["budget"]) == n), last)
            if n == 0:
                base_macro = float(last["macro"]) if last else 0.0

    if last is None:
        raise RuntimeError(f"no eval rows for {method} seed={seed}")
    # ensure base_macro
    if base_macro is None:
        zero = next(
            (json.loads(l) for l in checkpoint_path.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["budget"] == 0),
            None,
        )
        base_macro = float(zero["macro"]) if zero else 0.0
    dist = acquisition_distribution(acquired)
    times = target_skill_times(first_all, budget=budget)
    summary = {
        "method": method,
        "seed": seed,
        "n_acquired": len(acquired),
        "base_future_macro": base_macro,
        "future_macro": last["macro"],
        "delta_future": last["macro"] - base_macro,
        "per_dataset": last["per_dataset"],
        "per_capability": last["per_capability"],
        "n_active_skills": last["n_active_skills"],
        "tokens": int(sum(e.cost.tokens for e in acquired)),
        "tool_calls": int(sum(e.cost.tool_calls for e in acquired)),
        "dispatch": dispatch,
        "fit_mode": "count_only_unique_task",
        "min_support": min_support,
        "wall_seconds": time.time() - t0,
        **dist,
        **times,
        "protocol": "r5.3 public search; unique-task CountOnly; frozen frontier port",
    }
    write_json(out / "final_metrics.json", summary)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    heartbeat(out, phase="done", acquired=len(acquired), budget=budget, method=method, seed=seed)
    return summary
