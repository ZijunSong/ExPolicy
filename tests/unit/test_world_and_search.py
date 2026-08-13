"""Unit tests for Helion world + BM25 (no LLM)."""

from __future__ import annotations

from collections import Counter

from expolicy.benchmarks.search_benchmark import SearchBenchmark
from expolicy.core.types import DataSplit
from expolicy.environments.search.bm25 import BM25Index
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import build_world, world_stats


def test_world_is_large_and_split() -> None:
    world = build_world(13)
    stats = world_stats(world)
    assert stats["n_documents"] >= 200
    assert stats["n_tasks"] >= 200
    splits = Counter(t.split for t in world.tasks.values())
    for name in ("deployment", "acquisition", "validation", "future", "retention"):
        assert splits[name] > 0
    # future tasks must not share ids with deployment
    dep = {t.task_id for t in world.tasks.values() if t.split == DataSplit.DEPLOYMENT.value}
    fut = {t.task_id for t in world.tasks.values() if t.split == DataSplit.FUTURE.value}
    assert dep.isdisjoint(fut)


def test_bm25_retrieves_gold_for_single_hop() -> None:
    world = build_world(13)
    index = BM25Index.from_world(world)
    hits_ok = 0
    n = 0
    for task in world.tasks.values():
        if task.capability != "single_hop":
            continue
        n += 1
        if n > 40:
            break
        hits = index.search(task.question, top_k=8)
        hit_ids = {h.document_id for h in hits}
        if set(task.gold_doc_ids) & hit_ids:
            hits_ok += 1
    assert n >= 20
    assert hits_ok / min(n, 40) >= 0.6


def test_search_open_find_tools() -> None:
    env = SearchEnvironment.from_seed(13)
    task = next(iter(env.world.tasks.values()))
    env.reset(task)
    search = env.step("search", {"query": task.question})
    assert search.ok
    assert search.payload["hits"]
    doc_id = search.payload["hits"][0]["document_id"]
    opened = env.step("open", {"document_id": doc_id})
    assert opened.ok
    assert "text" in opened.payload
    found = env.step("find", {"document_id": doc_id, "pattern": opened.payload["text"].split()[3]})
    assert found.ok
    unknown = env.step("open", {"document_id": "doc_does_not_exist"})
    assert not unknown.ok


def test_benchmark_future_guard() -> None:
    bench = SearchBenchmark.build(13)
    future_ids = [t.task_id for t in bench.tasks(split="future")]
    try:
        bench.future_guard(future_ids[:1], future_ids)
        raised = False
    except Exception:
        raised = True
    assert raised
    bench.future_guard([], future_ids)


def test_world_deterministic() -> None:
    a = build_world(13)
    b = build_world(13)
    assert set(a.tasks) == set(b.tasks)
    assert a.tasks[next(iter(a.tasks))].gold_answer == b.tasks[next(iter(b.tasks))].gold_answer
