"""Shadow-probe evaluation for Exact AVA.

For the frozen CountOnly operator, A(S,B) depends only on which predicted
capabilities reach K unique evidences. Shadow / future metrics are therefore
a function of the activation mask and can be cached.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from expolicy.assimilation.carriers.skill_store import SkillStore
from expolicy.environments.public_search.capability_map import PUBLIC_CAPS
from expolicy.environments.public_search.skills import count_only_public_skill
from expolicy.experiments.common import write_json
from expolicy.experiments.r5_3 import eval_with_skills, heartbeat, macro_from_rows
from expolicy.state.isolated_clone import MIN_SUPPORT, caps_from_mask_key, mask_key_from_caps

ALL_MASKS: tuple[str, ...] = (
    "none",
    "bridge_decomposition",
    "comparison_alignment",
    "long_chain_search",
    "bridge_decomposition+comparison_alignment",
    "bridge_decomposition+long_chain_search",
    "comparison_alignment+long_chain_search",
    "bridge_decomposition+comparison_alignment+long_chain_search",
)


def store_for_mask(mask_key: str, dest: Path, *, min_support: int = MIN_SUPPORT) -> SkillStore | None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    caps = caps_from_mask_key(mask_key)
    if not caps:
        return None
    store = SkillStore(dest)
    for cap in caps:
        skill = count_only_public_skill(cap, n_support=max(min_support, 8), min_support=min_support, version=1)
        store.install(skill)
    return store


def mask_eval_path(root: Path, split_name: str, mask_key: str) -> Path:
    safe = mask_key.replace("+", "__")
    return Path(root) / "mask_eval" / split_name / f"{safe}.json"


def load_mask_summary(root: Path, split_name: str, mask_key: str) -> dict[str, Any] | None:
    path = mask_eval_path(root, split_name, mask_key)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def shadow_value(
    *,
    before_mask: str,
    after_mask: str,
    root: Path,
    split_name: str = "shadow_probe",
) -> float:
    before = load_mask_summary(root, split_name, before_mask)
    after = load_mask_summary(root, split_name, after_mask)
    if before is None or after is None:
        raise FileNotFoundError(f"mask eval missing for {before_mask} or {after_mask} on {split_name}")
    return float(after["macro"]) - float(before["macro"])


def task_shard(task_id: str, n_shards: int, shard: int) -> bool:
    if n_shards <= 1:
        return True
    h = int(hashlib.md5(task_id.encode("utf-8")).hexdigest(), 16)
    return h % n_shards == shard


def eval_mask_on_tasks(
    tasks: Sequence,
    *,
    mask_key: str,
    base_rows: Mapping[str, dict],
    env,
    model,
    router,
    out: Path,
    resume_path: Path | None = None,
    shard: int = 0,
    n_shards: int = 1,
) -> dict[str, Any]:
    from expolicy.logging.event_logger import append_jsonl

    store_dir = out / "stores" / mask_key.replace("+", "__")
    store = store_for_mask(mask_key, store_dir)
    selected = [t for t in tasks if task_shard(t.task_id, n_shards, shard)]
    shard_path = resume_path or (out / f"shard_{shard}.jsonl")
    done: dict[str, dict] = {}
    if shard_path.exists():
        for line in shard_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[str(rec["task_id"])] = rec
    remaining = [t for t in selected if t.task_id not in done]
    for i, task in enumerate(remaining):
        heartbeat(out, phase="mask_eval", mask=mask_key, shard=shard, i=i, n=len(remaining), task_id=task.task_id)
        payload = eval_with_skills(
            [task],
            store=store,
            base_rows=dict(base_rows),
            env=env,
            model=model if store is not None else None,
            dispatch="text_router",
            router=router,
            out=None,
        )
        row = payload["rows"][0]
        append_jsonl(shard_path, row)
        done[task.task_id] = row
    rows = [done[t.task_id] for t in selected if t.task_id in done]
    summary = macro_from_rows(rows)
    summary.update(
        {
            "mask": mask_key,
            "split": str(out.name),
            "n_skill_invocations": sum(1 for r in rows if str(r.get("dispatch_source") or "").startswith("skill:")),
            "n_base_fallback": sum(1 for r in rows if r.get("dispatch_source") == "base_llm"),
            "n_active_skills": 0 if store is None else len(caps_from_mask_key(mask_key)),
            "shard": shard,
            "n_shards": n_shards,
        }
    )
    write_json(out / f"summary_shard_{shard}.json", {k: v for k, v in summary.items() if k != "rows"})
    return summary
