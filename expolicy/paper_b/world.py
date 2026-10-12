"""Deterministic local tasks for the feasibility harness. Not an official benchmark."""

from __future__ import annotations

from pathlib import Path

from expolicy.paper_b.constants import (
    FAMILY_COMPARE,
    FAMILY_RECOVERY,
    FAMILY_RETENTION,
    SPLIT_CROSS,
    SPLIT_NEAR,
    SPLIT_RETENTION,
    SPLIT_TRAIN,
    SPLIT_UNRELATED,
    SPLIT_VAL,
)
from expolicy.paper_b.models import Document, Task
from expolicy.r9_lab.io_util import read_jsonl

_TEMPLATES = {
    FAMILY_COMPARE: {"near": "cmp_which_first", "cross": "cmp_released_earlier"},
    FAMILY_RECOVERY: {"near": "rec_who_directed", "cross": "rec_movie_director"},
    FAMILY_RETENTION: {"near": "ret_capital", "cross": "ret_capital"},
}


def build_world(counts: dict[str, dict[str, int]], *, seed: int) -> list[Task]:
    """Build disjoint entity groups. `seed` is recorded by the caller; generation itself is indexed."""
    del seed  # the index is the split identity; do not resample from it
    tasks: list[Task] = []
    for split, families in counts.items():
        for family, count in families.items():
            for index in range(int(count)):
                tasks.append(_task(split, family, index))
    tasks.sort(key=lambda task: task.task_id)
    return tasks


def load_tasks_jsonl(path: Path) -> list[Task]:
    tasks = []
    for row in read_jsonl(path):
        corpus = tuple(Document(terms=tuple(doc["terms"]), text=str(doc["text"])) for doc in row["corpus"])
        tasks.append(
            Task(
                task_id=str(row["task_id"]),
                split=str(row["split"]),
                source_group=str(row["source_group"]),
                entity_key=str(row["entity_key"]),
                db_object=str(row["db_object"]),
                family=str(row["family"]),
                question=str(row["question"]),
                gold_answer=str(row["gold_answer"]),
                corpus=corpus,
            )
        )
    return tasks


def _task(split: str, family: str, index: int) -> Task:
    band = "cross" if split == SPLIT_CROSS else "near"
    source_group = _TEMPLATES[family][band]
    tag = f"{split[:3]}{index:04d}"
    if family == FAMILY_COMPARE:
        left = f"Fa{tag}"
        right = f"Fb{tag}"
        year_left = 1980 + index
        year_right = 1991 + index
        if index % 2:
            year_left, year_right = year_right, year_left
        gold = left if year_left < year_right else right
        if source_group == "cmp_which_first":
            question = f"Which film came out first, {left} or {right}?"
        else:
            question = f"Which was released earlier, {left} or {right}?"
        corpus = (
            Document((left.casefold(), "release"), f"{left} was released in {year_left}."),
            Document((right.casefold(), "release"), f"{right} was released in {year_right}."),
        )
        entity_key = "|".join(sorted((left, right)))
    elif family == FAMILY_RECOVERY:
        movie = f"Mv{tag}"
        director = f"Dr{tag}"
        if source_group == "rec_who_directed":
            question = f"Who directed the film {movie}?"
        else:
            question = f"Name the director of the movie {movie}."
        corpus = (Document((movie.casefold(), "biography"), f"{movie} is a film directed by {director}."),)
        gold = director
        entity_key = movie
    else:
        city = f"Ct{tag}"
        capital = f"Cp{tag}"
        question = f"What is the capital of {city}?"
        corpus = (Document(("capital", city.casefold()), f"The capital of {city} is {capital}."),)
        gold = capital
        entity_key = city
        source_group = "ret_capital"
    return Task(
        task_id=f"{split}-{_code(family)}-{index:04d}",
        split=split,
        source_group=source_group,
        entity_key=entity_key,
        db_object=f"page:{split}:{entity_key}",
        family=family,
        question=question,
        gold_answer=gold,
        corpus=corpus,
    )


def _code(family: str) -> str:
    return {
        FAMILY_COMPARE: "cmp",
        FAMILY_RECOVERY: "rec",
        FAMILY_RETENTION: "ret",
    }[family]


def pilot_counts() -> dict[str, dict[str, int]]:
    target = {FAMILY_COMPARE: 48, FAMILY_RECOVERY: 48}
    return {
        SPLIT_TRAIN: dict(target),
        SPLIT_VAL: {FAMILY_COMPARE: 16, FAMILY_RECOVERY: 16, FAMILY_RETENTION: 16},
        SPLIT_NEAR: {FAMILY_COMPARE: 50, FAMILY_RECOVERY: 50},
        SPLIT_CROSS: {FAMILY_COMPARE: 50, FAMILY_RECOVERY: 50},
        SPLIT_RETENTION: {FAMILY_RETENTION: 100},
        SPLIT_UNRELATED: {FAMILY_RETENTION: 16},
    }


def feasibility_counts() -> dict[str, dict[str, int]]:
    return {
        SPLIT_TRAIN: {FAMILY_COMPARE: 4, FAMILY_RECOVERY: 4},
        SPLIT_VAL: {FAMILY_COMPARE: 3, FAMILY_RECOVERY: 3, FAMILY_RETENTION: 2},
        SPLIT_NEAR: {FAMILY_COMPARE: 4, FAMILY_RECOVERY: 4},
        SPLIT_CROSS: {FAMILY_COMPARE: 4, FAMILY_RECOVERY: 4},
        SPLIT_RETENTION: {FAMILY_RETENTION: 6},
        SPLIT_UNRELATED: {FAMILY_RETENTION: 4},
    }
