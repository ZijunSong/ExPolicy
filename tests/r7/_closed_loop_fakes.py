"""Shared fakes for R7 CPU tests. No gold-as-hit."""

from __future__ import annotations

from types import SimpleNamespace

from expolicy.closed_loop.constants import C1, C2, C3
from expolicy.environments.search.world import Task


class FakeGen:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = 8
        self.prompt_tokens = 4
        self.completion_tokens = 4
        self.latency = 0.001
        self.raw = {}


class FakeModel:
    name = "fake-qwen"

    def __init__(self) -> None:
        self.n = 0

    def generate(self, messages, **kwargs):
        seq = [
            "<tool>search</tool>\n<arg>entity keywords</arg>",
            "<tool>open</tool>\n<arg>doc_1</arg>",
            "<tool>answer</tool>\n<arg>Paris</arg>",
        ]
        text = seq[min(self.n % 3, 2)]
        self.n += 1
        return FakeGen(text)


class FakeClient:
    def search(self, query, topk=5):
        return [
            {
                "document_id": "doc_1",
                "title": "Paris",
                "text": "Paris is the capital of France. Bill Gates co-founded Microsoft. The Seine river.",
                "score": 1.0,
                "rank": 0,
            }
        ]

    def open_id(self, document_id):
        return self.search("open")[0]


def make_task(i: int, cap: str, split: str = "calibration") -> Task:
    return Task(
        task_id=f"{split}:{cap}:{i}",
        question=f"What is the capital of France item {i} {cap}?",
        gold_answer="Paris",
        capability=cap,
        stage=0,
        gold_doc_ids=[],
        distractor_doc_ids=[],
        hops=2 if cap != C3 else 3,
        split=split,
        aliases=["Paris"],
        metadata={"dataset": "hotpotqa"},
    )


def tiny_split() -> dict:
    def bucket(name, n_per=3):
        rows = []
        for cap in (C1, C2, C3):
            for i in range(n_per):
                t = make_task(i, cap, name)
                rows.append(
                    {
                        "task_id": t.task_id,
                        "dataset": "hotpotqa",
                        "official_split": "train",
                        "question": t.question,
                        "gold_answer": t.gold_answer,
                        "aliases": t.aliases,
                        "type": "bridge",
                        "hops": t.hops,
                        "supporting_titles": [],
                        "raw_id": str(i),
                        "capability": cap,
                    }
                )
        return rows

    return {
        "split_seed": 20260819,
        "calibration": bucket("calibration", 2),
        "acquisition_pool": bucket("acquisition_pool", 6),
        "shadow_reward_probe": bucket("shadow_reward_probe", 2),
        "retention_probe": bucket("retention_probe", 1)[:4],
        "future_test": bucket("future_test", 1),
    }
