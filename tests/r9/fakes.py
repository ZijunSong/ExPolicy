"""Shared fakes for R9 CPU tests. No GPU, no docker, no gold-as-observation."""

from __future__ import annotations

from pathlib import Path

from expolicy.r9_lab.benchmark.adapter import FakeLabEnv, FakeTask
from expolicy.r9_lab.benchmark.split import TaskRecord, build_synthetic_catalog, construct_split, write_split
from expolicy.r9_lab.llm_client import FakeLLMClient
from expolicy.r9_lab.state.persistent_state import PersistentState


GOLD_SQL = "SELECT UNIQUE_R9_GOLD_TOKEN_XYZ FROM secret_table"
GOLD_PLAN = "GOLD_ACTION_PLAN_DO_NOT_PROMPT"


def make_fake_task(env: str, i: int, *, split: str = "calibration") -> FakeTask:
    return FakeTask(
        task_id=f"{env}:{split}:{i}",
        env=env,
        instruction=f"[{env}] solve synthetic task {split} #{i}",
        skill_list=[f"{env}_skill_{i % 4}"],
        difficulty=i % 5,
        gold={"gold_sql": GOLD_SQL, "gold_action_plan": GOLD_PLAN, "success_token": "OK"},
        success_token="OK",
    )


def make_env_and_client(tasks: list[FakeTask]) -> tuple[FakeLabEnv, FakeLLMClient]:
    env_name = tasks[0].env if tasks else "db"
    return FakeLabEnv(tasks), FakeLLMClient(env=env_name, success_token="OK")


def write_synthetic_split(tmp: Path) -> dict:
    catalog = build_synthetic_catalog(n_per_env=600)
    payload = construct_split(catalog, synthetic=True)
    write_split(tmp / "split.json", payload)
    return payload


def fresh_state(tmp: Path) -> PersistentState:
    return PersistentState.create(tmp / "state")
