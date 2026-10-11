"""Wrap official LAB Task APIs. Never patch third_party.

If LAB / docker is unavailable, FakeLabEnv is used only by tests.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from expolicy.r9_lab.benchmark.verifier import assert_no_gold_in_prompt, strip_gold
from expolicy.r9_lab.constants import GOLD_FIELD_NAMES, LAB_ROOT, MAX_ROUND


def insert_lab_path() -> None:
    lab = str(LAB_ROOT)
    if lab not in sys.path:
        sys.path.insert(0, lab)


def try_import_official_parsers() -> dict[str, Callable[[str], Any]]:
    """Import official *_parse_agent_response if the LAB package is importable."""
    parsers: dict[str, Callable[[str], Any]] = {}
    try:
        insert_lab_path()
        from src.tasks.instance.db_bench.task import DBBench
        from src.tasks.instance.knowledge_graph.task import KnowledgeGraph
        from src.tasks.instance.os_interaction.task import OSInteraction

        parsers["db"] = DBBench._parse_agent_response
        parsers["os"] = OSInteraction._parse_agent_response
        parsers["kg"] = KnowledgeGraph._parse_agent_response
    except Exception:
        parsers = {}
    return parsers


@dataclass
class ParsedAction:
    action: str  # execute | finish | invalid
    content: Optional[str]
    finish_reason: Optional[str]
    parser_valid: bool


def _fallback_parse(env: str, text: str) -> ParsedAction:
    """Regex copies of official LAB patterns (read-only; not a third_party patch)."""
    if env == "db":
        m = re.search(r"Action: (Operation|Answer)", text)
        if m is None:
            return ParsedAction("invalid", None, "no Action", False)
        if m.group(1) == "Operation":
            sql = re.search(r"```sql\n([\s\S]*?)\n```", text)
            if sql is None:
                return ParsedAction("invalid", None, "no sql", False)
            return ParsedAction("execute", sql.group(1).strip().replace("\n", " "), None, True)
        ans = re.search(r"\nFinal Answer:(.*)", text)
        if ans is None:
            return ParsedAction("invalid", None, "no Final Answer", False)
        return ParsedAction("finish", ans.group(1).strip(), None, True)
    if env == "os":
        m = re.search(r"Act:\s*(.+)", text)
        if m is None:
            return ParsedAction("invalid", None, "no Act", False)
        action_str = m.group(1)
        if action_str.lower().startswith("bash"):
            blocks = re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)
            return ParsedAction("execute", "\n\n".join(blocks), None, True)
        if action_str.lower().startswith("finish"):
            return ParsedAction("finish", None, None, True)
        return ParsedAction("invalid", None, "invalid Act", False)
    if env == "kg":
        fin = re.search(r"Final [Aa]nswer:\s*(?:[Vv]ar(?:iable)?\s*)?#(\d+)", text)
        if fin:
            return ParsedAction("finish", fin.group(1), None, True)
        act = re.search(r"Action: (\w+)\((.+?)\)", text)
        if act is None:
            return ParsedAction("invalid", None, "no KG action", False)
        return ParsedAction("execute", f"{act.group(1)}({act.group(2)})", None, True)
    return ParsedAction("invalid", None, f"unknown env {env}", False)


def parse_agent_response(env: str, text: str, parsers: Mapping[str, Callable[[str], Any]] | None = None) -> ParsedAction:
    parsers = parsers if parsers is not None else try_import_official_parsers()
    fn = parsers.get(env)
    if fn is not None:
        try:
            result = fn(text)
            action = str(getattr(result, "action", result.get("action") if isinstance(result, dict) else "invalid"))
            action = action.split(".")[-1].lower() if "." in action else action.lower()
            content = getattr(result, "content", result.get("content") if isinstance(result, dict) else None)
            reason = getattr(result, "finish_reason", result.get("finish_reason") if isinstance(result, dict) else None)
            valid = action in {"execute", "finish"}
            return ParsedAction(action if valid else "invalid", content, reason, valid)
        except Exception:
            pass
    return _fallback_parse(env, text)


@dataclass
class FakeTask:
    task_id: str
    env: str
    instruction: str
    skill_list: list[str] = field(default_factory=list)
    difficulty: int = 0
    gold: dict[str, Any] = field(default_factory=dict)
    success_token: str = "OK"

    def public_view(self) -> dict[str, Any]:
        return strip_gold(
            {
                "task_id": self.task_id,
                "env": self.env,
                "instruction": self.instruction,
                "skill_list": self.skill_list,
                "difficulty": self.difficulty,
            }
        )


class FakeLabEnv:
    """Interactive fake used only by tests when LAB/docker is unavailable.

    Gold lives in a private store and is never returned by reset()/observe().
    """

    def __init__(self, tasks: Sequence[FakeTask] | Mapping[str, FakeTask]):
        if isinstance(tasks, Mapping):
            self.tasks = dict(tasks)
        else:
            self.tasks = {t.task_id: t for t in tasks}
        self._gold: dict[str, dict[str, Any]] = {t.task_id: dict(t.gold) for t in self.tasks.values()}
        self.current: FakeTask | None = None
        self.round = 0
        self.done = False
        self.success = False
        self.invalid_actions = 0
        self.parser_valid_steps = 0
        self.steps = 0
        self.last_obs = ""
        self.trajectory: list[dict[str, Any]] = []
        self.infra_failure = False

    @property
    def max_round(self) -> int:
        env = self.current.env if self.current else "db"
        return MAX_ROUND.get(env, 3)

    def reset(self, task_id: str) -> dict[str, Any]:
        self.current = self.tasks[task_id]
        self.round = 0
        self.done = False
        self.success = False
        self.invalid_actions = 0
        self.parser_valid_steps = 0
        self.steps = 0
        self.trajectory = []
        self.infra_failure = False
        self.last_obs = self.current.instruction
        obs = {"role": "user", "content": self.current.instruction, "task_id": task_id, "env": self.current.env}
        for banned in GOLD_FIELD_NAMES:
            assert banned not in obs
        return obs

    def interact(self, agent_text: str) -> dict[str, Any]:
        assert self.current is not None
        parsed = parse_agent_response(self.current.env, agent_text, parsers={})
        self.steps += 1
        self.round += 1
        if parsed.parser_valid:
            self.parser_valid_steps += 1
        else:
            self.invalid_actions += 1
        self.trajectory.append({"agent": agent_text, "parsed": parsed.action, "content": parsed.content})
        gold = self._gold.get(self.current.task_id) or {}
        token = str(gold.get("success_token") or self.current.success_token)
        if parsed.action == "finish":
            content = parsed.content or agent_text
            self.success = token in str(content) or token in agent_text
            self.done = True
        elif parsed.action == "execute":
            self.last_obs = f"executed: {parsed.content}"
            if token and token in str(parsed.content):
                self.success = True
            if self.round >= self.max_round:
                self.done = True
        else:
            self.last_obs = "invalid action"
            if self.round >= self.max_round:
                self.done = True
        return {"observation": self.last_obs, "done": self.done, "parser_valid": parsed.parser_valid}

    def complete(self) -> dict[str, Any]:
        self.done = True
        return {
            "success": bool(self.success),
            "steps": self.steps,
            "invalid_actions": self.invalid_actions,
            "parser_valid": self.parser_valid_steps == self.steps and self.steps > 0,
            "parser_valid_rate": (self.parser_valid_steps / self.steps) if self.steps else 0.0,
            "infra_failure": self.infra_failure,
        }

    def gold_for(self, task_id: str) -> dict[str, Any]:
        return dict(self._gold.get(task_id) or {})
