"""Official LAB session loop, parquet converter, vLLM launch cmd — fakes only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from expolicy.r9_lab.benchmark.adapter import insert_lab_path, parse_agent_response
from expolicy.r9_lab.benchmark.lab_data import (
    LabDataError,
    convert_source_dir,
    parquet_to_entry_dict,
    validate_entry,
)
from expolicy.r9_lab.benchmark.official_env import (
    OfficialLabEnv,
    OfficialLabInfraError,
    chat_history_to_openai_messages,
    construct_official_task,
    run_official_episode,
    summarize_episodes,
)
from expolicy.r9_lab.constants import DTYPE, MAX_MODEL_LEN, MAX_ROUND, vllm_port_for_model
from expolicy.r9_lab.llm_client import FakeLLMClient
from expolicy.r9_lab.state.persistent_state import PersistentState
from tests.r9.fakes import GOLD_SQL, make_fake_task


def _types():
    insert_lab_path()
    from src.typings import Role, SampleStatus, SessionEvaluationOutcome, TaskName

    return Role, SampleStatus, SessionEvaluationOutcome, TaskName


class FakeOfficialTask:
    def __init__(self, env: str = "db"):
        Role, SampleStatus, SessionEvaluationOutcome, TaskName = _types()
        self.env = env
        self.task_name = TaskName({"db": "db_bench", "os": "os_interaction", "kg": "knowledge_graph"}[env])
        self.max_round = MAX_ROUND[env]
        self.current_round = 0
        self.gold_sql = GOLD_SQL
        self._Outcome = SessionEvaluationOutcome
        self._Role = Role
        self._Status = SampleStatus

    def reset(self, session) -> None:
        session.sample_status = self._Status.RUNNING
        session.chat_history.inject(
            {
                "role": self._Role.USER,
                "content": "REQUIREMENT PROMPT. Now, I will give you the question that you need to solve.",
            }
        )
        session.chat_history.inject({"role": self._Role.AGENT, "content": "OK."})
        session.chat_history.inject({"role": self._Role.USER, "content": "How many rows are in the table?"})
        self.current_round = 0

    def interact(self, session) -> None:
        if self.current_round >= self.max_round:
            session.sample_status = self._Status.TASK_LIMIT_REACHED
            session.task_output = {"answer": None}
            return
        self.current_round += 1
        last = session.chat_history.get_item_deep_copy(-1).content
        parsed = parse_agent_response(self.env, last, parsers={})
        if parsed.action == "finish":
            session.task_output = {"answer": parsed.content or last}
            session.sample_status = self._Status.COMPLETED
            return
        if parsed.action == "execute":
            session.chat_history.inject({"role": self._Role.USER, "content": "[]"})
            return
        session.sample_status = self._Status.AGENT_VALIDATION_FAILED
        session.task_output = {"answer": None}

    def complete(self, session) -> None:
        blob = ""
        for i in range(session.chat_history.get_value_length()):
            blob += session.chat_history.get_item_deep_copy(i).content
        ok = "OK" in blob
        session.evaluation_record.outcome = self._Outcome.from_bool(ok)

    def release(self) -> None:
        return None


def test_official_episode_injects_prefix_into_first_user_only(tmp_path: Path):
    task = FakeOfficialTask("db")
    client = FakeLLMClient(env="db", success_token="OK")
    state = PersistentState.create(tmp_path / "s")
    state.skill_markdown = "SKILL_PREFIX_UNIQUE_R9"
    state.save()
    result = run_official_episode(task, "0", client, env="db", state=state, gold={"gold_sql": GOLD_SQL})
    assert result.infra_failure is False
    assert result.success is True
    assert client.calls
    first_user = next(m for m in client.calls[0] if m["role"] == "user")
    assert first_user["content"].startswith("Skill document:\nSKILL_PREFIX_UNIQUE_R9")
    assert "How many rows are in the table?" in "\n".join(m["content"] for m in client.calls[0] if m["role"] == "user")
    assert "REQUIREMENT PROMPT" in first_user["content"]
    assert GOLD_SQL not in str(result.messages)
    assert GOLD_SQL not in str(client.calls)


def test_official_episode_messages_are_openai_roles():
    task = FakeOfficialTask("db")
    client = FakeLLMClient(env="db")
    run_official_episode(task, "1", client, env="db")
    roles = {m["role"] for m in client.calls[0]}
    assert "system" in roles
    assert "assistant" in roles or any(m["role"] == "user" for m in client.calls[0])
    assert "agent" not in {m["role"] for m in client.calls[0]}


def test_parquet_converts_to_db_entry_dict(tmp_path: Path):
    src = tmp_path / "db_bench"
    src.mkdir()
    df = pd.DataFrame(
        [
            {
                "sample_index": "7",
                "instruction": "count employees",
                "skill_list": json.dumps(["select"]),
                "answer_info": json.dumps(
                    {"md5": None, "direct": [[3]], "sql": "SELECT COUNT(*) FROM emp"}
                ),
                "table_info": json.dumps(
                    {
                        "name": "emp",
                        "row_list": [[1, 2], [3, 4], [5, 6]],
                        "column_info_list": [{"name": "id", "type": "int"}, {"name": "n", "type": "int"}],
                    }
                ),
            }
        ]
    )
    parquet = src / "train-00000-of-00001.parquet"
    df.to_parquet(parquet)
    dest = tmp_path / "entry_dict.json"
    info = convert_source_dir("db", src, dest)
    payload = json.loads(dest.read_text())
    assert "7" in payload
    assert payload["7"]["instruction"] == "count employees"
    assert payload["7"]["answer_info"]["sql"] == "SELECT COUNT(*) FROM emp"
    assert payload["7"]["skill_list"] == ["select"]
    assert info["kind"] == "parquet"


def test_parquet_parses_python_repr_nested_fields(tmp_path: Path):
    src = tmp_path / "db_bench"
    src.mkdir()
    df = pd.DataFrame(
        [
            {
                "sample_index": 0,
                "instruction": "insert a row",
                "skill_list": "['insert']",
                "answer_info": "{'direct': None, 'md5': 'abc', 'sql': 'INSERT INTO t VALUES (1)'}",
                "table_info": "{'name': 't', 'row_list': [(1,)], 'column_info_list': [{'name': 'id', 'type': 'INT'}]}",
            }
        ]
    )
    df.to_parquet(src / "train.parquet")
    dest = tmp_path / "entry_dict.json"
    convert_source_dir("db", src, dest)
    payload = json.loads(dest.read_text())
    assert payload["0"]["skill_list"] == ["insert"]
    assert payload["0"]["answer_info"]["sql"].startswith("INSERT")
    assert payload["0"]["table_info"]["name"] == "t"


def test_converter_refuses_to_invent_gold(tmp_path: Path):
    src = tmp_path / "db_bench"
    src.mkdir()
    df = pd.DataFrame([{"sample_index": "0", "instruction": "hi", "skill_list": ["select"]}])
    df.to_parquet(src / "x.parquet")
    with pytest.raises(LabDataError, match="invent gold|missing"):
        convert_source_dir("db", src, tmp_path / "entry_dict.json")


def test_construct_official_task_raises_infra_without_assets(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("expolicy.r9_lab.benchmark.official_env.LAB_ROOT", tmp_path)
    with pytest.raises(OfficialLabInfraError):
        construct_official_task("db")


def test_vllm_launch_cmd_uses_frozen_constants():
    from pathlib import Path as P

    import importlib.util

    path = P("/data/ppnm/expolicy/scripts/r9/serve_vllm.py")
    spec = importlib.util.spec_from_file_location("serve_vllm_r9", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    cmd = mod.build_vllm_cmd(model_path="/m", served_name="q0", port=18100)
    assert "--generation-config" in cmd and "vllm" in cmd
    assert "--dtype" in cmd and DTYPE in cmd
    assert "--max-model-len" in cmd and str(MAX_MODEL_LEN) in cmd
    assert vllm_port_for_model("Q1") == 18101


def test_model_screen_dry_run_and_fake_official_path(tmp_path: Path, monkeypatch):
    import importlib.util

    from expolicy.r9_lab.benchmark.split import construct_split, build_synthetic_catalog, write_split
    from expolicy.r9_lab.constants import MODEL_LADDER

    payload = construct_split(build_synthetic_catalog(n_per_env=600), synthetic=True)
    write_split(tmp_path / "split.json", payload)
    path = Path("/data/ppnm/expolicy/scripts/r9/02_model_screen.py")
    spec = importlib.util.spec_from_file_location("r9_02", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    rc = mod.main(["--dry-run", "--output-root", str(tmp_path), "--model-id", "Q0", "--env", "db"])
    assert rc == 0
    csv_text = (tmp_path / "R9_MODEL_SCREEN.csv").read_text()
    assert "Q0" in csv_text

    class Wrap:
        def __init__(self, env: str):
            self.env = env
            self.task = FakeOfficialTask(env)

        def run_episode(self, sample_index, client, **kwargs):
            return run_official_episode(self.task, sample_index, client, env=self.env, **kwargs)

        def close(self) -> None:
            return None

    recs = [
        type("R", (), {"sample_index": "0", "task_id": "db:0", "instruction": "q"})()
    ]
    stats = mod.run_screen_cell(
        spec=MODEL_LADDER[0],
        env="db",
        records=recs,
        port=18100,
        gpu="4",
        output_root=tmp_path,
        env_ctor=Wrap,
        client=FakeLLMClient(env="db"),
    )
    assert stats["n"] == 1
    assert stats["infra_failure_rate"] == 0.0
    assert stats.get("faked_success") is not True


def test_summarize_does_not_treat_empty_as_success():
    s = summarize_episodes([])
    assert s["n"] == 0
    assert s["n_success"] == 0
