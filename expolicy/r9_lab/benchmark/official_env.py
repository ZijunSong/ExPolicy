"""Official LifelongAgentBench Task session loop. Never patch third_party.

Uses OUR VLLMClient for inference so Q2/Q3 can pass enable_thinking=False
without touching LAB LanguageModelAgent.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Protocol

from expolicy.r9_lab.benchmark.adapter import insert_lab_path, parse_agent_response
from expolicy.r9_lab.benchmark.verifier import assert_no_gold_in_prompt, prompt_contains_gold
from expolicy.r9_lab.constants import (
    GOLD_FIELD_NAMES,
    LAB_CHAT_HISTORY_REL,
    LAB_DATA_REL,
    LAB_KG_ONTOLOGY_REL,
    LAB_OS_COMMAND_TIMEOUT,
    LAB_ROOT,
    LAB_SPARQL_URL,
    LAB_SYSTEM_PROMPT,
    LAB_TASK_NAME,
    MAX_ROUND,
)
from expolicy.r9_lab.runner import EpisodeResult, inject_state_prefix
from expolicy.r9_lab.state.persistent_state import PersistentState


class OfficialLabInfraError(RuntimeError):
    """Docker / MySQL / SPARQL / official assets missing. Do not fake success."""


class LabTaskProtocol(Protocol):
    task_name: Any
    max_round: int

    def reset(self, session: Any) -> None: ...
    def interact(self, session: Any) -> None: ...
    def complete(self, session: Any) -> None: ...
    def release(self) -> None: ...


def _lab_types() -> Any:
    insert_lab_path()
    from src.typings import (  # noqa: WPS433
        ChatHistoryItem,
        Role,
        SampleStatus,
        Session,
        SessionEvaluationOutcome,
        TaskName,
    )

    return {
        "ChatHistoryItem": ChatHistoryItem,
        "Role": Role,
        "SampleStatus": SampleStatus,
        "Session": Session,
        "SessionEvaluationOutcome": SessionEvaluationOutcome,
        "TaskName": TaskName,
    }


def openai_role_dict() -> dict[Any, str]:
    types = _lab_types()
    Role = types["Role"]
    return {Role.USER: "user", Role.AGENT: "assistant"}


def chat_history_to_openai_messages(
    chat_history: Any,
    *,
    role_dict: Mapping[Any, str] | None = None,
    system_prompt: str = LAB_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    role_dict = role_dict or openai_role_dict()
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for idx in range(chat_history.get_value_length()):
        item = chat_history.get_item_deep_copy(idx)
        messages.append({"role": role_dict[item.role], "content": item.content})
    return messages


def _first_user_index(chat_history: Any, Role: Any) -> int | None:
    for idx in range(chat_history.get_value_length()):
        if chat_history.get_item_deep_copy(idx).role == Role.USER:
            return idx
    return None


def _last_user_text(chat_history: Any, Role: Any) -> str:
    text = ""
    for idx in range(chat_history.get_value_length()):
        item = chat_history.get_item_deep_copy(idx)
        if item.role == Role.USER:
            text = item.content
    return text


def inject_prefix_into_first_user(session: Any, prefix: str) -> None:
    """Prepend retrieved state into the first USER message. Do not delete official text."""
    if not prefix.strip():
        return
    types = _lab_types()
    Role = types["Role"]
    ChatHistoryItem = types["ChatHistoryItem"]
    idx = _first_user_index(session.chat_history, Role)
    if idx is None:
        return
    item = session.chat_history.get_item_deep_copy(idx)
    if prefix in item.content:
        return
    session.chat_history.set(
        idx,
        ChatHistoryItem(role=Role.USER, content=prefix.rstrip() + "\n\n" + item.content),
    )


def _assert_messages_have_no_gold(messages: list[dict[str, str]], gold: Any | None) -> None:
    blob = "\n".join(m.get("content") or "" for m in messages)
    for banned in GOLD_FIELD_NAMES:
        # Field names like "gold" may appear in English requirements; only fail on payload leak.
        pass
    if gold is not None:
        hits = prompt_contains_gold(messages, gold)
        if hits:
            raise AssertionError(f"gold/reference leaked into LLM prompt: {hits[:3]}")
        assert_no_gold_in_prompt(messages, gold)
    # Hard-stop if ground-truth SQL-looking gold strings were injected by us.
    if "UNIQUE_R9_GOLD_TOKEN_XYZ" in blob:
        raise AssertionError("test gold token leaked into LLM prompt")


def run_official_episode(
    task: LabTaskProtocol,
    sample_index: Any,
    client: Any,
    *,
    env: str,
    state: PersistentState | None = None,
    gold: Any | None = None,
    task_id: str | None = None,
    instruction_hint: str = "",
) -> EpisodeResult:
    """Official Session + reset / inference / interact / complete, using OUR client."""
    types = _lab_types()
    Session = types["Session"]
    Role = types["Role"]
    ChatHistoryItem = types["ChatHistoryItem"]
    SampleStatus = types["SampleStatus"]
    SessionEvaluationOutcome = types["SessionEvaluationOutcome"]

    t0 = time.time()
    tokens = 0
    reasoning_tokens = 0
    invalid_actions = 0
    parser_valid_steps = 0
    steps = 0
    infra_failure = False
    messages_acc: list[dict[str, str]] = []
    trajectory: list[dict[str, Any]] = []
    instruction = instruction_hint
    success = False

    session = Session(task_name=task.task_name, sample_index=sample_index)
    try:
        task.reset(session)
        query = _last_user_text(session.chat_history, Role) or instruction_hint
        instruction = query or instruction
        prefix = inject_state_prefix(state, query)
        inject_prefix_into_first_user(session, prefix)
        role_dict = openai_role_dict()
        while session.sample_status == SampleStatus.RUNNING:
            messages = chat_history_to_openai_messages(session.chat_history, role_dict=role_dict)
            _assert_messages_have_no_gold(messages, gold)
            messages_acc = messages
            result = client.chat(messages)
            tokens += int(getattr(result, "total_tokens", 0) or 0)
            reasoning_tokens += int(getattr(result, "reasoning_tokens", 0) or 0)
            agent_text = result.text or ""
            session.chat_history.inject(ChatHistoryItem(role=Role.AGENT, content=agent_text))
            parsed = parse_agent_response(env, agent_text, parsers={})
            steps += 1
            if parsed.parser_valid:
                parser_valid_steps += 1
            else:
                invalid_actions += 1
            trajectory.append({"agent": agent_text, "parsed": parsed.action, "content": parsed.content})
            task.interact(session)
        task.complete(session)
        success = session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
        if session.sample_status in {
            SampleStatus.TASK_ENVIRONMENT_ERROR,
            SampleStatus.TASK_UNKNOWN_ERROR,
        }:
            infra_failure = True
    except OfficialLabInfraError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        msg = str(exc).lower()
        dockerish = any(
            token in msg or token in name.lower()
            for token in ("docker", "mysql", "sparql", "connection refused", "errno 111", "container")
        )
        if dockerish:
            raise OfficialLabInfraError(
                f"LAB environment unavailable for env={env} sample={sample_index}: {type(exc).__name__}: {exc}"
            ) from exc
        infra_failure = True

    return EpisodeResult(
        task_id=task_id or f"{env}:{sample_index}",
        env=env,
        success=bool(success) and not infra_failure,
        steps=steps,
        invalid_actions=invalid_actions,
        parser_valid=parser_valid_steps == steps and steps > 0,
        tokens=tokens,
        reasoning_tokens=reasoning_tokens,
        wall_time=time.time() - t0,
        infra_failure=infra_failure,
        messages=list(messages_acc),
        trajectory=trajectory,
        instruction=instruction,
    )


def _require_file(path: Any, what: str) -> None:
    from pathlib import Path

    if not Path(path).exists():
        raise OfficialLabInfraError(f"missing {what}: {path}")


def construct_official_task(env: str) -> LabTaskProtocol:
    """Instantiate official Task. Raises OfficialLabInfraError if docker/data missing."""
    insert_lab_path()
    types = _lab_types()
    TaskName = types["TaskName"]
    data_path = LAB_ROOT / LAB_DATA_REL[env]
    chat_path = LAB_ROOT / LAB_CHAT_HISTORY_REL[env]
    _require_file(data_path, "official entry_dict.json")
    _require_file(chat_path, "official chat_history_items JSON")
    try:
        from src.factories.chat_history_item.online.chat_history_item_factory import (  # noqa: WPS433
            ChatHistoryItemFactory,
        )

        factory = ChatHistoryItemFactory(str(chat_path))
        task_name = TaskName(LAB_TASK_NAME[env])
        if env == "db":
            from src.tasks.instance.db_bench.task import DBBench  # noqa: WPS433

            return DBBench(
                task_name=task_name,
                chat_history_item_factory=factory,
                data_file_path=str(data_path),
                max_round=MAX_ROUND["db"],
            )
        if env == "os":
            from src.tasks.instance.os_interaction.task import OSInteraction  # noqa: WPS433

            return OSInteraction(
                task_name=task_name,
                chat_history_item_factory=factory,
                data_file_path=str(data_path),
                max_round=MAX_ROUND["os"],
                command_execution_timeout=LAB_OS_COMMAND_TIMEOUT,
            )
        if env == "kg":
            ontology = LAB_ROOT / LAB_KG_ONTOLOGY_REL
            if not ontology.exists():
                raise OfficialLabInfraError(f"missing KG ontology dir: {ontology}")
            from src.tasks.instance.knowledge_graph.task import KnowledgeGraph  # noqa: WPS433

            return KnowledgeGraph(
                task_name=task_name,
                chat_history_item_factory=factory,
                sparql_url=LAB_SPARQL_URL,
                ontology_dir_path=str(ontology),
                data_file_path=str(data_path),
                max_round=MAX_ROUND["kg"],
            )
        raise OfficialLabInfraError(f"unknown env {env}")
    except OfficialLabInfraError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise OfficialLabInfraError(
            f"Failed to construct official LAB Task for env={env} "
            f"(docker/mysql/SPARQL/data). {type(exc).__name__}: {exc}"
        ) from exc


class OfficialLabEnv:
    def __init__(self, env: str, *, task: LabTaskProtocol | None = None):
        self.env = env
        self._owns_task = task is None
        self.task = task if task is not None else construct_official_task(env)

    def run_episode(
        self,
        sample_index: Any,
        client: Any,
        *,
        state: PersistentState | None = None,
        gold: Any | None = None,
        task_id: str | None = None,
        instruction_hint: str = "",
    ) -> EpisodeResult:
        return run_official_episode(
            self.task,
            sample_index,
            client,
            env=self.env,
            state=state,
            gold=gold,
            task_id=task_id,
            instruction_hint=instruction_hint,
        )

    def close(self) -> None:
        if self._owns_task:
            try:
                self.task.release()
            except Exception:
                pass


def summarize_episodes(results: list[EpisodeResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "n": 0,
            "success": float("nan"),
            "n_success": 0,
            "n_failure": 0,
            "parser_valid_rate": float("nan"),
            "infra_failure_rate": float("nan"),
            "mean_steps": float("nan"),
            "invalid_action_rate": float("nan"),
            "tokens": 0,
            "reasoning_tokens": 0,
            "wall_time": 0.0,
        }
    n_success = sum(1 for r in results if r.success)
    n_infra = sum(1 for r in results if r.infra_failure)
    n_parser = sum(1 for r in results if r.parser_valid)
    n_invalid = sum(r.invalid_actions for r in results)
    n_steps = sum(r.steps for r in results)
    return {
        "n": n,
        "success": n_success / n,
        "n_success": n_success,
        "n_failure": n - n_success,
        "parser_valid_rate": n_parser / n,
        "infra_failure_rate": n_infra / n,
        "mean_steps": (n_steps / n) if n else 0.0,
        "invalid_action_rate": (n_invalid / n_steps) if n_steps else 0.0,
        "tokens": sum(r.tokens for r in results),
        "reasoning_tokens": sum(r.reasoning_tokens for r in results),
        "wall_time": sum(r.wall_time for r in results),
    }
