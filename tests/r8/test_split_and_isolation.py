from __future__ import annotations

import json
from pathlib import Path

from expolicy.r8.family import family_from_gamefile

ROOT = Path("/data/ppnm/expolicy")
SPLIT = ROOT / "outputs" / "r8_backend_validation" / "split" / "split.json"


def test_family_from_gamefile() -> None:
    assert (
        family_from_gamefile(
            "json_2.1.1/train/pick_heat_then_place_in_recep-Potato-None-Fridge-2/trial_X/game.tw-pddl"
        )
        == "pick_heat_then_place_in_recep"
    )
    assert family_from_gamefile("look_at_obj_in_light-CD-None-DeskLamp-304/trial") == "look_at_obj_in_light"


def test_clone_isolation_script() -> None:
    import subprocess
    import sys

    rc = subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "r8" / "audit_counterfactual_clone.py")],
    )
    assert rc == 0


def test_frozen_split_if_present() -> None:
    if not SPLIT.exists():
        return
    payload = json.loads(SPLIT.read_text())
    future = {x["gamefile"] for x in payload["buckets"]["future_test"]}
    skillopt = SPLIT.parent / "skillopt"
    for name in ("train", "val", "test"):
        items = json.loads((skillopt / name / "items.json").read_text())
        leaked = {it["gamefile"] for it in items} & future
        assert not leaked, leaked
    ids = []
    for bucket, rows in payload["buckets"].items():
        ids.append((bucket, {x["id"] for x in rows}))
    for i, (a, sa) in enumerate(ids):
        for b, sb in ids[i + 1 :]:
            assert not (sa & sb), (a, b, sa & sb)
