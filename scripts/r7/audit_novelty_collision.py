#!/usr/bin/env python3
"""Novelty collision audit. Experience-policy-as-such is already occupied."""

from __future__ import annotations

import argparse
from pathlib import Path

from expolicy.r7.constants import R7_ROOT

TABLE = """# NOVELTY_COLLISION

As of 2026-08-15, the following public prior art already occupies naive "experience policy" claims.
R7 **must not** treat "a policy decides how to use experience" as novelty.

| Work | Occupied space | Restriction on ExPolicy |
|---|---|---|
| Ace-Skill, arXiv:2605.08887 | prioritized experience sampling + skill organization | “主动采更有价值的数据” itself is not new |
| SkillOS, arXiv:2605.06614 | learned skill curator; insert/update/delete SkillRepo | skill curation policy is not new |
| COVE, arXiv:2608.01234 | harness-side vs parametric-side routing + stage-aware trigger | carrier routing alone is not novelty |
| ExpWeaver, arXiv:2605.07164 | agent decides when to use experience at runtime | “when to use experience” is studied |
| MemCon, arXiv:2607.13591 | learned policy for retrieve / inject / consolidate / forget | memory control policy occupied |
| OPD-Evolver, arXiv:2606.17628 | select/use/write/maintain experience + OPD internalization | holistic memory-management policy |
| SESA, arXiv:2607.29468 | task distribution ↔ skill memory co-evolution | “experience changes later training distribution” exists |
| SkillOpt, arXiv:2605.23904 | rollout-driven held-out gated skill optimization | trainable external skill state exists |
| SkillMentor, arXiv:2607.27360 | blind-spot diagnosis → corrective skills | “learn what to practice” exists |
| HASP, arXiv:2605.17734 | experience → executable intervention program | executable skill is not novelty |
| OEL, arXiv:2603.16856 | deployment experience → OPCD internalization | experience → parameter exists |
| OPCD, arXiv:2602.12275 | on-policy context distillation | parametric internalization is not new |
| PATH-Bench, arXiv:2608.01149 | path-dependent lifelong experience + selective use | path-dependent experience value is named |

## Allowed stronger novelty (R7 target)

Learned Experience Lifecycle Policy: one transferable policy jointly decides
(A) acquire what → (B) transform/assimilate into which substrate → (C) validate/promote/retire when,
optimizing delayed future capability gain, forgetting, and compute cost.

ExPolicy-v2 must show these stage decisions are **coupled**, and that a joint policy beats the
strongest stage-wise composition (Ace-Skill + SkillOpt + SEU/ExpWeaver, plus SkillOS / COVE / OPD-Evolver).

If that extra gain is absent, **do not** keep “Experience Policy” as the paper's main contribution.

## This round's non-claims

- Interventional skills are a **validity fix**, not novelty (HASP-adjacent).
- SkillOpt-style assimilator is a **strong baseline backend**, not novelty.
- Frontier acquisition remaining as a Helion-only heuristic is **not** a paper.
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(R7_ROOT / "audit"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "NOVELTY_COLLISION.md").write_text(TABLE, encoding="utf-8")
    print(f"wrote {out / 'NOVELTY_COLLISION.md'}", flush=True)


if __name__ == "__main__":
    main()
