"""Index historical R9 cells without copying them into the A-v1 matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from expolicy.r9_lab.constants import A3_SKILLOPT
from expolicy.r9_lab.io_util import read_json


def inventory_legacy_matrix(matrix_root: Path) -> dict[str, Any]:
    cells = []
    if not matrix_root.exists():
        return {"n_cells": 0, "cells": [], "splice_into_a_v1": False}
    for done in sorted(matrix_root.glob("*/*/DONE.json")):
        if done.parent.parent.name.startswith("_"):
            continue
        payload = read_json(done)
        action = str(payload.get("action") or done.parent.name)
        has_items = any(isinstance(payload.get(key), list) for key in ("items", "task_rows", "per_task"))
        cells.append(
            {
                "batch_id": payload.get("batch_id") or done.parent.parent.name,
                "action": action,
                "path": str(done),
                "has_per_task_rows": has_items,
                "protocol_id": payload.get("protocol_id"),
                "in_a_v1_action_space": action != A3_SKILLOPT,
                "reusable_as_a_v1_row": False,
                "reason": "legacy aggregate cell; per-task pairing and A-v1 config hash are absent",
            }
        )
    return {
        "n_cells": len(cells),
        "n_skillopt": sum(1 for cell in cells if cell["action"] == A3_SKILLOPT),
        "n_a_v1_actions": sum(1 for cell in cells if cell["in_a_v1_action_space"]),
        "cells": cells,
        "splice_into_a_v1": False,
    }
