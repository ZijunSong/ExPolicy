"""R5.4 RASA helpers: frozen holdout, deployable router, clusters, seed loop."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import numpy as np

from expolicy.acquisition.ace_skill import AceSkillPrioritizedAcquisition
from expolicy.acquisition.base import make_acquisition
from expolicy.acquisition.rasa import RASAAcquisition
from expolicy.assimilation.carriers.skill_store import SkillStore
from expolicy.assimilation.readiness_gate import STATE_ACTIVE, STATE_PARKED, STATE_UNRESOLVED
from expolicy.core.experience import Experience
from expolicy.core.state import empty_agent
from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
from expolicy.environments.public_search.capability_map import PUBLIC_CAPS
from expolicy.environments.public_search.environment import PublicSearchEnvironment
from expolicy.environments.public_search.evaluator import score_public
from expolicy.environments.public_search.retriever import RetrieverClient
from expolicy.environments.public_search.skills import (
    COMPILER_VERSION,
    PRIMITIVES,
    count_only_public_skill,
    run_public_skill_on_task,
)
from expolicy.environments.search.world import Task
from expolicy.experiments.common import ROOT, mean, write_json
from expolicy.experiments.r5_3 import (
    DEFAULT_MODEL,
    TextRouter,
    acquisition_distribution,
    build_public_agent,
    eval_with_skills,
    heartbeat,
    macro_from_rows,
    world_from_tasks,
)
from expolicy.experiments.r_common import apply_gpu, write_run_header
from expolicy.logging.event_logger import append_jsonl

R54_ROOT = ROOT / "outputs" / "r5_4_rasa"
R53_ROOT = ROOT / "outputs" / "r5_3_external_generalization"
E5_MODEL = "/data/ppnm/models/e5-base-v2"
SPLIT_SEED = 20260815
ACQ_PER_DS = 300
FUTURE_PER_DS = 150
BUDGET_DEFAULT = 48
FULL_EVAL_AT = (0, 16, 32, 48)
MIN_SUPPORT = 8
KMEANS_RANDOM_STATE = 13
RETRIEVER_URL = "http://127.0.0.1:18080"
DATASETS = ("hotpotqa", "2wiki", "musique")

CONDITIONS: dict[str, dict[str, Any]] = {
    "passive_k8": {"acquisition": "passive", "activation": "count_only", "qonly": False, "diversity": True},
    "ace_k8": {"acquisition": "ace_skill_prioritized", "activation": "count_only", "qonly": False, "diversity": True},
    "legacy_frontier_k8": {"acquisition": "frontier", "activation": "count_only", "qonly": False, "diversity": True},
    "rasa_k8": {"acquisition": "rasa", "activation": "count_only", "qonly": False, "diversity": True},
    "passive_srt": {"acquisition": "passive", "activation": "srt", "qonly": False, "diversity": True},
    "rasa_full": {"acquisition": "rasa", "activation": "srt", "qonly": False, "diversity": True},
    "rasa_qonly": {"acquisition": "rasa", "activation": "srt", "qonly": True, "diversity": True},
    "rasa_nodiversity": {"acquisition": "rasa", "activation": "srt", "qonly": False, "diversity": False},
}

NEEDS_REPLAY = {"rasa_k8", "passive_srt", "rasa_full", "rasa_qonly", "rasa_nodiversity"}
NEEDS_RASA_STATE = {"rasa_k8", "rasa_full", "rasa_qonly", "rasa_nodiversity", "passive_srt"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def r53_blocked() -> tuple[set[str], set[str]]:
    split = load_json(R53_ROOT / "splits" / "split.json")
    ids: set[str] = set()
    qs: set[str] = set()
    for key in ("calibration", "acquisition_pool", "skill_validation", "future_probe", "bamboogle"):
        for rec in split.get(key) or []:
            ids.add(str(rec["task_id"]))
            qs.add(str(rec.get("question") or "").strip().lower())
    return ids, qs


def skill_byte_hashes() -> dict[str, str]:
    out = {"compiler_version": COMPILER_VERSION}
    for cap, fn in PRIMITIVES.items():
        skill = fn()
        raw = skill.model_dump_json()
        out[cap] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return out


def frozen_frontier_formula() -> str:
    return "(learnability * novelty * transfer * unc) / cost"


def load_split_r54(root: Path | None = None) -> dict[str, Any]:
    root = Path(root or R54_ROOT)
    path = root / "split_r5_4.json"
    if not path.exists():
        raise FileNotFoundError(f"R5.4 split missing: {path}")
    return load_json(path)


def records_to_tasks(records: Sequence[dict[str, Any]], *, split: str, predictions: Mapping[str, str] | None = None) -> list[Task]:
    from expolicy.environments.public_search.datasets import to_task

    predictions = dict(predictions or {})
    tasks = []
    for rec in records:
        true_cap = rec.get("capability") or "unmapped"
        if true_cap == "unmapped":
            continue
        pred = predictions.get(rec["task_id"]) or rec.get("predicted_capability") or true_cap
        task = to_task(rec, capability=str(pred), split=split)
        task.metadata = {
            **(task.metadata or {}),
            "true_capability": true_cap,
            "predicted_capability": str(pred),
        }
        tasks.append(task)
    return tasks


def load_predictions(root: Path) -> dict[str, str]:
    path = root / "predictions.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    return {str(k): str(v) for k, v in (payload.get("predicted") or payload).items()}


class FrozenTextRouter:
    def __init__(self, vectorizer, clf, labels: Sequence[str] = PUBLIC_CAPS) -> None:
        self.vectorizer = vectorizer
        self.clf = clf
        self.labels = list(labels)

    def predict_one(self, question: str) -> str:
        pred = self.clf.predict(self.vectorizer.transform([question]))[0]
        return str(pred)

    def predict_many(self, questions: Sequence[str]) -> list[str]:
        if not questions:
            return []
        return [str(x) for x in self.clf.predict(self.vectorizer.transform(list(questions)))]


def train_r54_router(root: Path) -> FrozenTextRouter:
    """Train only on R5.3 calibration + acquisition question text + capability labels."""
    r53 = load_json(R53_ROOT / "splits" / "split.json")
    train_recs = list(r53.get("calibration") or []) + list(r53.get("acquisition_pool") or [])
    questions = [r["question"] for r in train_recs if r.get("capability") in PUBLIC_CAPS]
    labels = [r["capability"] for r in train_recs if r.get("capability") in PUBLIC_CAPS]
    counts = Counter(labels)
    router = TextRouter(seed=0)
    metrics = router.fit(questions, labels)

    # leakage: R5.3 heldout already inside TextRouter.fit; also score R5.3 skill_validation as extra heldout
    val_recs = list(r53.get("skill_validation") or [])
    y_true = [r["capability"] for r in val_recs if r.get("capability") in PUBLIC_CAPS]
    y_pred = [router.predict_one(r["question"]) for r in val_recs if r.get("capability") in PUBLIC_CAPS]
    val_acc = mean(float(a == b) for a, b in zip(y_pred, y_true)) if y_true else 0.0

    split = load_split_r54(root)
    pred_map: dict[str, str] = {}
    leak_task_ids = []
    for key in ("acquisition_pool", "future_probe"):
        for rec in split.get(key) or []:
            q = rec["question"]
            if rec["task_id"] in q:
                leak_task_ids.append(rec["task_id"])
            pred_map[rec["task_id"]] = router.predict_one(q)

    vocab = set(router.vectorizer.get_feature_names_out().tolist()) if router.vectorizer is not None else set()
    suspicious = [t for t in vocab if t.startswith("hotpotqa:") or t.startswith("2wiki:") or t.startswith("musique:")]
    acq_pred = Counter(pred_map[r["task_id"]] for r in split["acquisition_pool"])
    fut_pred = Counter(pred_map[r["task_id"]] for r in split["future_probe"])

    router_dir = root / "router"
    router_dir.mkdir(parents=True, exist_ok=True)
    with (router_dir / "text_router.pkl").open("wb") as handle:
        pickle.dump({"vectorizer": router.vectorizer, "clf": router.clf, "labels": list(PUBLIC_CAPS)}, handle)

    payload = {
        "train_size": len(questions),
        "class_distribution": dict(counts),
        "r53_internal_val": metrics,
        "r53_skill_validation_accuracy": val_acc,
        "task_id_in_question": leak_task_ids,
        "vocabulary_task_id_tokens": suspicious,
        "r54_acq_predicted_distribution": dict(acq_pred),
        "r54_future_predicted_distribution": dict(fut_pred),
        "n_r54_predicted": len(pred_map),
        "trained_on": "R5.3 calibration + R5.3 acquisition only",
        "future_labels_used_to_fit": False,
    }
    write_json(root / "predictions.json", {"predicted": pred_map, "n": len(pred_map)})
    write_json(root / "router_metrics.json", payload)
    md = [
        "# ROUTER_AUDIT",
        "",
        "Deployable TF-IDF + multinomial LR. Training data = R5.3 calibration + R5.3 acquisition question text + capability labels.",
        "R5.4 future labels, task ids, dataset filename, gold answers, gold docs, and future outcomes were not used to fit.",
        "",
        json.dumps(payload, indent=2),
        "",
        "Predictions for R5.4 acquisition/future are stored in predictions.json. True capability is diagnostic-only.",
    ]
    (root / "ROUTER_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return FrozenTextRouter(router.vectorizer, router.clf)


def load_router(root: Path) -> FrozenTextRouter:
    path = root / "router" / "text_router.pkl"
    if not path.exists():
        return train_r54_router(root)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return FrozenTextRouter(payload["vectorizer"], payload["clf"], payload.get("labels") or PUBLIC_CAPS)


class E5Encoder:
    def __init__(self, model_path: str = E5_MODEL, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        vecs = []
        for start in range(0, len(texts), batch_size):
            chunk = [f"query: {t}" for t in texts[start : start + batch_size]]
            if not chunk:
                continue
            encoded = self.tokenizer(chunk, max_length=256, padding=True, truncation=True, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                out = self.model(**encoded, return_dict=True)
                mask = encoded["attention_mask"]
                hidden = out.last_hidden_state.masked_fill(~mask[..., None].bool(), 0.0)
                pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)[..., None]
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vecs.append(pooled.cpu().numpy().astype(np.float32))
        if not vecs:
            return np.zeros((0, 768), dtype=np.float32)
        return np.concatenate(vecs, axis=0)


def fit_clusters(root: Path) -> dict[str, Any]:
    from sklearn.cluster import KMeans

    split = load_split_r54(root)
    preds = load_predictions(root)
    acq = list(split.get("acquisition_pool") or [])
    encoder = E5Encoder()
    questions = [r["question"] for r in acq]
    vecs = encoder.encode(questions)
    ids = [r["task_id"] for r in acq]
    by_cap: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(ids):
        cap = preds.get(tid) or acq[i].get("capability") or "unknown"
        by_cap[cap].append(i)
    clusters: dict[str, dict[str, int]] = {}
    meta = {"k_per_cap": {}, "n_per_cap": {}, "random_state": KMEANS_RANDOM_STATE}
    for cap, idxs in by_cap.items():
        n = len(idxs)
        k = min(4, max(2, n // 20)) if n >= 2 else 1
        k = min(k, n)
        X = vecs[np.array(idxs)]
        if k <= 1:
            labels = np.zeros(n, dtype=int)
        else:
            km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X)
        meta["k_per_cap"][cap] = int(k)
        meta["n_per_cap"][cap] = int(n)
        for local_i, global_i in enumerate(idxs):
            clusters[ids[global_i]] = {"cluster": int(labels[local_i]), "cap": cap}
    emb_path = root / "acq_e5_embeddings.npz"
    np.savez_compressed(emb_path, ids=np.array(ids), vectors=vecs)
    write_json(root / "clusters.json", {"clusters": clusters, "meta": meta, "fitted_without_outcomes": True})
    return {"clusters": clusters, "meta": meta, "n": len(ids)}


def load_clusters(root: Path) -> tuple[dict[str, dict[str, int]], dict[str, np.ndarray]]:
    cpath = root / "clusters.json"
    epath = root / "acq_e5_embeddings.npz"
    if not cpath.exists():
        fit_clusters(root)
    payload = load_json(cpath)
    clusters = payload.get("clusters") or {}
    embeddings: dict[str, np.ndarray] = {}
    if epath.exists():
        data = np.load(epath, allow_pickle=True)
        ids = [str(x) for x in data["ids"].tolist()]
        vecs = data["vectors"]
        embeddings = {tid: vecs[i] for i, tid in enumerate(ids)}
    return clusters, embeddings


def load_base_rows(root: Path) -> dict[str, dict]:
    path = root / "base_eval" / "rows.jsonl"
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                tid = rec.get("task_id")
                if tid:
                    rows[str(tid)] = rec
        return rows
    for shard in sorted((root / "base_eval").glob("shard_*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                tid = rec.get("task_id")
                if tid:
                    rows[str(tid)] = rec
    return rows


def _load_jsonl_exp(path: Path) -> list[Experience]:
    rows: list[Experience] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            rows.append(Experience.model_validate(payload))
        except Exception:
            continue
    return rows


def count_only_store_predicted(
    acquired: Sequence[Experience],
    *,
    min_support: int,
    skill_dir: Path,
    budget: int | None = None,
) -> tuple[SkillStore, dict[str, int]]:
    import shutil

    skill_dir.mkdir(parents=True, exist_ok=True)
    store_path = skill_dir / ("store" if budget is None else f"store_b{budget}")
    if store_path.exists():
        shutil.rmtree(store_path)
    store = SkillStore(store_path)
    subset = list(acquired if budget is None else acquired[:budget])
    seen: dict[str, set[str]] = {cap: set() for cap in PUBLIC_CAPS}
    first: dict[str, int] = {}
    for i, exp in enumerate(subset, 1):
        cap = exp.capability if exp.capability in seen else None
        if not cap or not exp.task_id:
            continue
        seen[cap].add(str(exp.task_id))
        if cap not in first and len(seen[cap]) >= min_support:
            first[cap] = i
    for cap in PUBLIC_CAPS:
        n_unique = len(seen[cap])
        if n_unique < min_support:
            continue
        skill = count_only_public_skill(cap, n_support=n_unique, min_support=min_support, version=1)
        store.install(skill)
        write_json(skill_dir / f"count_only_{cap}.json", skill.model_dump(mode="json"))
    return store, first


def srt_store(rasa: RASAAcquisition, skill_dir: Path, budget: int) -> SkillStore:
    import shutil

    skill_dir.mkdir(parents=True, exist_ok=True)
    store_path = skill_dir / f"store_b{budget}"
    if store_path.exists():
        shutil.rmtree(store_path)
    store = SkillStore(store_path)
    for cap, rec in rasa.state.items():
        if rec.get("state") != STATE_ACTIVE:
            continue
        n_unique = len(rec.get("acquired_ids") or [])
        skill = count_only_public_skill(cap, n_support=max(n_unique, 1), min_support=1, version=1)
        skill.activation_reason = f"SRT ACTIVE; {rec.get('reason')}"
        store.install(skill)
        write_json(skill_dir / f"srt_{cap}.json", skill.model_dump(mode="json"))
    return store


def _seed_complete(out: Path) -> bool:
    if not (out / "DONE").exists():
        return False
    if not (out / "final_metrics.json").exists():
        return False
    path = out / "checkpoint_metrics.jsonl"
    if not path.exists():
        return False
    budgets = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            budgets.add(int(json.loads(line)["budget"]))
    # budget 0 always required; later checkpoints may be missing if early-stopped after writing terminal eval
    return 0 in budgets and (out / "final_metrics.json").exists()


def _try_seed_lock(out: Path) -> TextIO | None:
    lock_path = out / ".run.lock"
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} ts={time.time()}\n")
    fh.flush()
    return fh


def _release_seed_lock(fh: TextIO | None) -> None:
    if fh is None:
        return
    try:
        fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        fh.close()
    except OSError:
        pass


def run_one_seed(
    *,
    condition: str,
    seed: int,
    output_dir: Path,
    model: str,
    budget: int,
    gpu: str | None,
    resume: bool,
    retriever_url: str,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a (condition, seed) directory across GPUs; wait if another worker holds it."""
    root = Path(output_dir)
    out = root / "per_seed" / condition / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    lock_f: TextIO | None = None
    while True:
        if resume and _seed_complete(out):
            return {"status": "already_done", "out": str(out), "condition": condition, "seed": seed}
        lock_f = _try_seed_lock(out)
        if lock_f is not None:
            break
        print(f"[lock] {condition} seed={seed} held by another worker; waiting", flush=True)
        time.sleep(5)
    try:
        return _run_one_seed_unlocked(
            condition=condition,
            seed=seed,
            output_dir=output_dir,
            model=model,
            budget=budget,
            gpu=gpu,
            resume=resume,
            retriever_url=retriever_url,
            cache=cache,
        )
    finally:
        _release_seed_lock(lock_f)


def _run_one_seed_unlocked(
    *,
    condition: str,
    seed: int,
    output_dir: Path,
    model: str,
    budget: int,
    gpu: str | None,
    resume: bool,
    retriever_url: str,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    spec = CONDITIONS[condition]
    apply_gpu(gpu)
    root = Path(output_dir)
    out = root / "per_seed" / condition / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if resume and _seed_complete(out):
        return {"status": "already_done", "out": str(out), "condition": condition, "seed": seed}

    split = load_split_r54(root)
    predictions = load_predictions(root)
    clusters, embeddings = load_clusters(root)
    router = load_router(root)
    pool = records_to_tasks(split.get("acquisition_pool") or [], split="acquisition_pool", predictions=predictions)
    future = records_to_tasks(split.get("future_probe") or [], split="future_probe", predictions=predictions)
    by_id = {t.task_id: t for t in pool}
    world = world_from_tasks(list(pool) + list(future))
    base_rows = load_base_rows(root)
    if not base_rows:
        raise RuntimeError("base_eval rows missing; run --phase base_eval first")
    write_run_header(
        out,
        config={
            "script": "run_r5_4_condition.py",
            "condition": condition,
            "seed": seed,
            "model": model,
            "gpu": gpu,
            "budget": budget,
            "dispatch": "text_router",
            "activation": spec["activation"],
            "acquisition": spec["acquisition"],
            "retriever_url": retriever_url,
            "split_hash": sha256_file(root / "split_r5_4.json") if (root / "split_r5_4.json").exists() else "",
        },
    )

    acquired = _load_jsonl_exp(out / "trajectories.jsonl") if resume else []
    dummy_state = empty_agent(out / "dummy", model_name=model)
    diagnoser = CapabilityDiagnoser()
    rasa: RASAAcquisition | None = None
    policy: Any
    if spec["acquisition"] == "rasa" or condition in NEEDS_RASA_STATE:
        rasa = RASAAcquisition(
            seed=seed,
            predicted_cap=predictions,
            clusters=clusters,
            embeddings=embeddings,
            qonly=bool(spec["qonly"]),
            diversity=bool(spec["diversity"]) if spec["acquisition"] == "rasa" else False,
        )
        if spec["acquisition"] == "rasa":
            policy = rasa
        else:
            policy = make_acquisition(spec["acquisition"], seed=seed)
    else:
        policy = make_acquisition(spec["acquisition"], seed=seed)

    pending: list[str] = []
    t_round = 0
    state_path = out / "sampler_state.json"
    rasa_path = out / "rasa_state.json"
    if isinstance(policy, AceSkillPrioritizedAcquisition):
        policy.bind([t.task_id for t in pool])
        if resume and state_path.exists():
            payload = load_json(state_path)
            policy.load_state(payload)
            pending = list(payload.get("pending_round") or [])
            t_round = int(payload.get("t", 0))
    if rasa is not None and resume and rasa_path.exists():
        rasa.load_snapshot(load_json(rasa_path))

    cache = cache if cache is not None else {}
    checkpoint_path = out / "checkpoint_metrics.jsonl"
    done_evals: set[int] = set()
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_evals.add(int(json.loads(line)["budget"]))

    unique_ids = []
    seen_unique: set[str] = set()
    for exp in acquired:
        if exp.task_id and str(exp.task_id) not in seen_unique:
            seen_unique.add(str(exp.task_id))
            unique_ids.append(str(exp.task_id))

    need_llm = len(unique_ids) < budget and not (rasa is not None and spec["acquisition"] == "rasa" and rasa.all_terminal())
    need_eval = any(n not in done_evals for n in FULL_EVAL_AT if n <= budget)
    agent = cache.get("agent")
    base_state = cache.get("base_state")
    backend = cache.get("backend")
    env = cache.get("env")
    if (need_llm or need_eval) and agent is None:
        heartbeat(out, phase="load_model", acquired=len(unique_ids), budget=budget, condition=condition, seed=seed)
        agent, base_state, backend, env = build_public_agent(
            world, model_path=model, output_dir=out / "base_agent", retriever_url=retriever_url
        )
        cache.update({"agent": agent, "base_state": base_state, "backend": backend, "env": env})
    env = env or cache.get("env") or PublicSearchEnvironment(world, RetrieverClient(retriever_url))
    t0 = time.time()
    replay_count = 0
    tokens = int(sum(e.cost.tokens for e in acquired))
    tools = int(sum(e.cost.tool_calls for e in acquired))
    cost_path = out / "cost.jsonl"
    if resume and cost_path.exists():
        last_cost = None
        for line in cost_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last_cost = json.loads(line)
        if last_cost:
            replay_count = int(last_cost.get("diagnostic_replay_count") or replay_count)

    remaining = [t for t in pool if t.task_id not in seen_unique]
    recent = list(acquired)
    frozen_skills = {cap: fn() for cap, fn in PRIMITIVES.items()}
    for cap, skill in frozen_skills.items():
        skill.status = "active"
        skill.executable = True
        skill.runtime_config = {
            "primitive_library": "r5_3_public_v1",
            "fit_mode": "frozen_r53",
            "max_opens": 6,
            "max_hops": 4 if cap == "long_chain_search" else 2,
        }

    def save_rasa() -> None:
        if rasa is not None:
            write_json(rasa_path, rasa.snapshot())
        if isinstance(policy, AceSkillPrioritizedAcquisition):
            payload = policy.state_dict()
            payload["t"] = t_round
            payload["pending_round"] = pending
            write_json(state_path, payload)

    def maybe_eval(n_unique: int, *, force: bool = False) -> dict[str, Any] | None:
        targets = [b for b in FULL_EVAL_AT if b <= n_unique]
        if force and n_unique not in targets:
            targets.append(n_unique)
        last_row = None
        for n in sorted(set(targets)):
            if n in done_evals and not force:
                continue
            if n in done_evals:
                continue
            heartbeat(out, phase="eval", acquired=n, budget=budget, condition=condition, seed=seed)
            chunk = [e for e in acquired if e.task_id in set(unique_ids[:n])]
            # preserve unique order
            id_order = unique_ids[:n]
            chunk_by_id = {}
            ordered = []
            for e in acquired:
                tid = str(e.task_id)
                if tid in id_order and tid not in chunk_by_id:
                    chunk_by_id[tid] = e
            ordered = [chunk_by_id[i] for i in id_order if i in chunk_by_id]
            if spec["activation"] == "count_only":
                store, first = count_only_store_predicted(
                    ordered, min_support=MIN_SUPPORT, skill_dir=out / "skill_versions", budget=n
                )
            else:
                assert rasa is not None
                store = srt_store(rasa, out / "skill_versions", n)
                first = {
                    cap: rec.get("budget_to_terminal")
                    for cap, rec in rasa.state.items()
                    if rec.get("state") == STATE_ACTIVE and rec.get("budget_to_terminal") is not None
                }
            if n == 0:
                payload = eval_with_skills(
                    future, store=None, base_rows=base_rows, env=env, model=None, dispatch="text_router", router=router, out=out
                )
            else:
                payload = eval_with_skills(
                    future,
                    store=store,
                    base_rows=base_rows,
                    env=env,
                    model=backend,
                    dispatch="text_router",
                    router=router,
                    out=out,
                )
            dist = acquisition_distribution(ordered)
            posteriors = rasa.state if rasa is not None else {}
            row = {
                "budget": n,
                "condition": condition,
                "seed": seed,
                "macro": payload["macro"],
                "per_dataset": payload["per_dataset"],
                "per_capability": payload["per_capability"],
                "n_active_skills": payload["n_active_skills"],
                "n_skill_invocations": payload["n_skill_invocations"],
                "activation": spec["activation"],
                "first_skill_activation": dict(first),
                "rasa_states": {c: rec.get("state") for c, rec in (posteriors or {}).items()} if posteriors else {},
                "posteriors": {
                    c: {
                        k: rec.get(k)
                        for k in (
                            "alpha_q",
                            "beta_q",
                            "alpha_p",
                            "beta_p",
                            "mean_q",
                            "mean_p",
                            "mean_z",
                            "P_good",
                            "state",
                            "cluster_coverage",
                            "n_tested_base_failures",
                            "repair_count",
                        )
                    }
                    for c, rec in (posteriors or {}).items()
                }
                if posteriors
                else {},
                **dist,
            }
            append_jsonl(checkpoint_path, row)
            done_evals.add(n)
            write_json(out / f"eval_budget_{n}.json", {k: v for k, v in payload.items() if k != "rows"})
            last_row = row
        return last_row

    if 0 not in done_evals:
        maybe_eval(0)

    early_stop = False
    while len(unique_ids) < budget:
        if rasa is not None and spec["acquisition"] == "rasa" and rasa.all_terminal() and rasa.warm_done:
            early_stop = True
            break
        n = len(unique_ids)
        want = 1 if spec["acquisition"] == "rasa" else min(8, budget - n)
        profile = diagnoser.diagnose(recent) if recent else diagnoser.diagnose([])
        if pending:
            batch_ids = pending[:want]
            pending = pending[want:]
        else:
            cand_pool = remaining if spec["acquisition"] != "ace_skill_prioritized" else pool
            if spec["acquisition"] == "ace_skill_prioritized":
                props = policy.propose(dummy_state, profile, cand_pool, want, recent=recent, t=t_round)
            else:
                props = policy.propose(dummy_state, profile, cand_pool, want, recent=recent)
            batch_ids = [tid for tid in list(props[0].task_ids) if tid not in seen_unique][:want]
            if props[0].metadata.get("early_stop") and not batch_ids:
                early_stop = True
                break
        if not batch_ids:
            if remaining:
                batch_ids = [remaining[0].task_id]
            else:
                break
        pending = list(batch_ids)
        save_rasa()
        for tid in list(batch_ids):
            if len(unique_ids) >= budget:
                break
            if tid in seen_unique:
                pending = [x for x in pending if x != tid]
                remaining = [t for t in remaining if t.task_id != tid]
                continue
            task = by_id.get(tid) or next((t for t in remaining if t.task_id == tid), None)
            if task is None:
                if remaining:
                    task = remaining[0]
                else:
                    break
            heartbeat(
                out,
                phase="acquire",
                acquired=len(unique_ids),
                budget=budget,
                condition=condition,
                seed=seed,
                task_id=task.task_id,
            )
            if agent is None:
                raise RuntimeError("LLM agent required for acquisition")
            agent.skill_store = None
            agent.use_executable_skill = False
            exp = agent.run_task(task, state=base_state)
            exp.run_id = "acquire"
            pred_cap = predictions.get(task.task_id) or task.capability
            exp.capability = pred_cap
            exp.capability_context = {
                **(exp.capability_context or {}),
                "dataset": (task.metadata or {}).get("dataset"),
                "predicted_capability": pred_cap,
                "true_capability": (task.metadata or {}).get("true_capability"),
            }
            row = score_public(exp, task)
            base_success = bool(row.get("em"))
            exp.success = base_success
            acquired.append(exp)
            recent.append(exp)
            tokens += int(exp.cost.tokens)
            tools += int(exp.cost.tool_calls)
            unique_ids.append(str(task.task_id))
            seen_unique.add(str(task.task_id))
            remaining = [t for t in remaining if t.task_id != task.task_id]
            if rasa is not None:
                rasa.observe_base(task.task_id, base_success, unique_budget=len(unique_ids))
            if isinstance(policy, AceSkillPrioritizedAcquisition):
                policy.observe(task.task_id, 1.0 if base_success else 0.0, t_round)

            repaired = None
            if (not base_success) and condition in NEEDS_REPLAY and rasa is not None:
                cap = str(pred_cap)
                rec = rasa.state.get(cap) or {}
                if rec.get("state") not in {STATE_ACTIVE, STATE_PARKED}:
                    skill = frozen_skills.get(cap)
                    if skill is not None:
                        heartbeat(
                            out,
                            phase="skill_replay",
                            acquired=len(unique_ids),
                            condition=condition,
                            seed=seed,
                            task_id=task.task_id,
                        )
                        replay = run_public_skill_on_task(skill, env, task, model=backend)
                        replay.run_id = "skill_replay"
                        replay.capability = cap
                        replay_row = score_public(replay, task)
                        repaired = bool(replay_row.get("em"))
                        replay_count += 1
                        tokens += int(replay.cost.tokens)
                        tools += int(replay.cost.tool_calls)
                        rasa.observe_repair(task.task_id, repaired)
                        append_jsonl(out / "skill_replays.jsonl", json.loads(replay.model_dump_json()))
            if rasa is not None:
                rasa.apply_readiness(unique_budget=len(unique_ids), use_srt=spec["activation"] == "srt")
            append_jsonl(out / "trajectories.jsonl", json.loads(exp.model_dump_json()))
            append_jsonl(
                out / "selected_experiences.jsonl",
                {
                    "step": len(unique_ids),
                    "task_id": task.task_id,
                    "predicted_capability": pred_cap,
                    "true_capability": (task.metadata or {}).get("true_capability"),
                    "dataset": (task.metadata or {}).get("dataset"),
                    "base_success": base_success,
                    "skill_repaired": repaired,
                    "tokens": exp.cost.tokens,
                    "tool_calls": exp.cost.tool_calls,
                },
            )
            append_jsonl(
                cost_path,
                {
                    "step": len(unique_ids),
                    "tokens": tokens,
                    "tool_calls": tools,
                    "diagnostic_replay_count": replay_count,
                    "wall_seconds": time.time() - t0,
                },
            )
            pending = [x for x in pending if x != tid]
            save_rasa()
            if len(unique_ids) in FULL_EVAL_AT:
                maybe_eval(len(unique_ids))
        t_round += 1
        pending = []
        save_rasa()

    if early_stop:
        maybe_eval(len(unique_ids), force=True)
    for n in FULL_EVAL_AT:
        if n <= len(unique_ids) and n not in done_evals:
            maybe_eval(n)
        elif n > len(unique_ids) and early_stop and n not in done_evals:
            # reuse terminal agent: copy last eval with budget label
            rows = [json.loads(l) for l in checkpoint_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if rows:
                last = dict(rows[-1])
                last["budget"] = n
                last["early_stop_copy"] = True
                append_jsonl(checkpoint_path, last)
                done_evals.add(n)

    rows = [json.loads(l) for l in checkpoint_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    zero = next((r for r in rows if int(r["budget"]) == 0), None)
    last = next((r for r in rows if int(r["budget"]) == budget), rows[-1] if rows else None)
    if last is None or zero is None:
        raise RuntimeError(f"incomplete evals for {condition} seed={seed}")
    base_macro = float(zero["macro"])
    dist = acquisition_distribution(acquired)
    rasa_states = rasa.state if rasa is not None else {}
    n_terminal = sum(1 for rec in rasa_states.values() if rec.get("state") in {STATE_ACTIVE, STATE_PARKED})
    collapse = False
    if rasa_states:
        counts = Counter()
        for exp in acquired:
            counts[str(exp.capability)] += 1
        if counts and counts.most_common(1)[0][1] == len(acquired) == budget:
            only = counts.most_common(1)[0][0]
            collapse = rasa_states.get(only, {}).get("state") == STATE_UNRESOLVED
    first_active = {
        cap: rec.get("budget_to_terminal")
        for cap, rec in rasa_states.items()
        if rec.get("state") == STATE_ACTIVE
    }
    summary = {
        "condition": condition,
        "seed": seed,
        "n_acquired_unique": len(unique_ids),
        "n_acquired_rollouts": len(acquired),
        "early_stop": early_stop,
        "base_future_macro": base_macro,
        "future_macro": last["macro"],
        "delta_future": last["macro"] - base_macro,
        "per_dataset": last["per_dataset"],
        "per_capability": last["per_capability"],
        "n_active_skills": last["n_active_skills"],
        "tokens": tokens,
        "tool_calls": tools,
        "diagnostic_replay_count": replay_count,
        "total_executor_episodes": len(acquired) + replay_count,
        "wall_seconds": time.time() - t0,
        "dispatch": "text_router",
        "activation": spec["activation"],
        "acquisition": spec["acquisition"],
        "n_terminal_caps": n_terminal,
        "n_unresolved_caps": sum(1 for rec in rasa_states.values() if rec.get("state") == STATE_UNRESOLVED),
        "collapse_unresolved_single_cap": collapse,
        "budget_to_first_ACTIVE": min([v for v in first_active.values() if v is not None], default=None),
        "budget_to_C2_ACTIVE": first_active.get("comparison_alignment"),
        "rasa_states": {c: rec.get("state") for c, rec in rasa_states.items()},
        "posteriors": last.get("posteriors") or {},
        **dist,
        "protocol": "r5.4 RASA; deployable text router; frozen public skills; no R5.3 future reuse",
    }
    write_json(out / "final_metrics.json", summary)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    heartbeat(out, phase="done", acquired=len(unique_ids), budget=budget, condition=condition, seed=seed)
    return summary
