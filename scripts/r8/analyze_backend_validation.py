#!/usr/bin/env python3
"""Score R8-B backend validation and emit Gate H / foundation decisions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expolicy.r8.constants import (
    FAMILIES,
    FAMILY_GAIN_MIN,
    FOUNDATION_MAX,
    FOUNDATION_MIN,
    ORACLE_VS_FIXED_MIN,
)


def _load_done(jobs: Path) -> list[dict]:
    rows = []
    for done in sorted(jobs.glob("*/DONE")):
        try:
            rows.append(json.loads(done.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def _macro(rates: dict[str, float | None]) -> float | None:
    vals = [rates[f] for f in FAMILIES if rates.get(f) is not None]
    if len(vals) != len(FAMILIES):
        return None
    return sum(vals) / len(vals)


def _mean_over_seeds(items: list[dict]) -> dict[str, float | None]:
    acc: dict[str, list[float]] = defaultdict(list)
    for item in items:
        rates = (item.get("parsed") or {}).get("family_rates") or {}
        if not rates and item.get("future"):
            # memrl future may only have macro
            continue
        for fam, val in rates.items():
            if val is not None:
                acc[fam].append(float(val))
    return {fam: (sum(acc[fam]) / len(acc[fam]) if acc.get(fam) else None) for fam in FAMILIES}


def analyze(input_dir: Path, out_prefix: Path) -> dict:
    jobs = input_dir / "jobs"
    rows = _load_done(jobs)
    foundation = next((r for r in rows if r.get("condition") == "foundation_no_learning"), None)
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("smoke") or row.get("condition") == "foundation_no_learning":
            continue
        cond = row.get("condition")
        if cond:
            by_cond[cond].append(row)

    none_rates = _mean_over_seeds(by_cond.get("none", []))
    mem_rates = _mean_over_seeds(by_cond.get("memrl", []))
    skill_rates = _mean_over_seeds(by_cond.get("skillopt", []))

    # MemRL future macro fallback if family rates missing.
    def _fallback_macro(cond: str, parsed: dict[str, float | None]) -> float | None:
        m = _macro(parsed)
        if m is not None:
            return m
        vals = []
        for row in by_cond.get(cond, []):
            if cond == "memrl":
                fut = row.get("future") or {}
                if fut.get("success") is not None:
                    vals.append(float(fut["success"]))
            else:
                suc = (row.get("parsed") or {}).get("summary", {}).get("success")
                if suc is not None:
                    vals.append(float(suc))
        return sum(vals) / len(vals) if vals else None

    none_macro = _fallback_macro("none", none_rates)
    mem_macro = _fallback_macro("memrl", mem_rates)
    skill_macro = _fallback_macro("skillopt", skill_rates)

    mem_gain = {}
    skill_gain = {}
    best_backend = {}
    for fam in FAMILIES:
        n = none_rates.get(fam)
        m = mem_rates.get(fam)
        s = skill_rates.get(fam)
        mem_gain[fam] = (m - n) if (m is not None and n is not None) else None
        skill_gain[fam] = (s - n) if (s is not None and n is not None) else None
        cands = []
        if n is not None:
            cands.append(("none", n))
        if m is not None:
            cands.append(("memrl", m))
        if s is not None:
            cands.append(("skillopt", s))
        best_backend[fam] = max(cands, key=lambda x: x[1])[0] if cands else None

    mem_ok = any(v is not None and v >= FAMILY_GAIN_MIN for v in mem_gain.values())
    skill_ok = any(v is not None and v >= FAMILY_GAIN_MIN for v in skill_gain.values())
    winners = {b for b in best_backend.values() if b and b != "none"}
    hetero_ok = len(winners) >= 2

    oracle_macro_vals = []
    fixed_scores = {
        "none": none_macro,
        "memrl": mem_macro,
        "skillopt": skill_macro,
    }
    best_fixed = None
    best_fixed_name = None
    for name, val in fixed_scores.items():
        if val is None:
            continue
        if best_fixed is None or val > best_fixed:
            best_fixed = val
            best_fixed_name = name
    for fam in FAMILIES:
        cands = [v for v in (none_rates.get(fam), mem_rates.get(fam), skill_rates.get(fam)) if v is not None]
        if cands:
            oracle_macro_vals.append(max(cands))
    oracle_macro = sum(oracle_macro_vals) / len(oracle_macro_vals) if len(oracle_macro_vals) == len(FAMILIES) else None
    oracle_gap = (oracle_macro - best_fixed) if (oracle_macro is not None and best_fixed is not None) else None
    oracle_ok = oracle_gap is not None and oracle_gap >= ORACLE_VS_FIXED_MIN

    foundation_success = None
    if foundation:
        foundation_success = (foundation.get("parsed") or {}).get("summary", {}).get("success")
        rates = (foundation.get("parsed") or {}).get("family_rates") or {}
        if foundation_success is None:
            foundation_success = _macro(rates)

    foundation_pass = (
        foundation_success is not None
        and FOUNDATION_MIN <= float(foundation_success) <= FOUNDATION_MAX
    )
    gate_h = bool(mem_ok and skill_ok and hetero_ok and oracle_ok)
    decision = "HETEROGENEOUS_ASSIMILATION_PASS" if gate_h else "HETEROGENEOUS_ASSIMILATION_GATE_FAIL"
    if foundation_success is not None and not foundation_pass:
        decision = "MODEL_FOUNDATION_GATE_FAIL"

    report = {
        "decision": decision,
        "foundation_success": foundation_success,
        "foundation_pass": foundation_pass,
        "none_family": none_rates,
        "memrl_family": mem_rates,
        "skillopt_family": skill_rates,
        "memrl_gain_vs_none": mem_gain,
        "skillopt_gain_vs_none": skill_gain,
        "best_backend_per_family": best_backend,
        "macros": {"none": none_macro, "memrl": mem_macro, "skillopt": skill_macro, "oracle_per_family": oracle_macro},
        "best_fixed": {"name": best_fixed_name, "macro": best_fixed},
        "oracle_minus_best_fixed": oracle_gap,
        "checks": {
            "memrl_family_gain_ge_05": mem_ok,
            "skillopt_family_gain_ge_05": skill_ok,
            "two_families_disagree_on_best": hetero_ok,
            "oracle_vs_fixed_ge_03": oracle_ok,
        },
        "n_jobs": len(rows),
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(out_prefix) + ".json") if out_prefix.suffix == "" else out_prefix.with_suffix(".json")
    md_path = json_path.with_suffix(".md")
    if out_prefix.name == "BACKEND_GATE" or out_prefix.name.endswith("BACKEND_GATE"):
        json_path = Path(str(out_prefix) + ".json")
        md_path = Path(str(out_prefix) + ".md")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# R8 Backend Validation Gate",
        "",
        f"**Decision: `{decision}`**",
        "",
        f"- Foundation no-learning success: {foundation_success} (band [{FOUNDATION_MIN}, {FOUNDATION_MAX}])",
        f"- Macros: none={none_macro} memrl={mem_macro} skillopt={skill_macro} oracle_per_family={oracle_macro}",
        f"- Best fixed: {best_fixed_name}={best_fixed}; oracle−fixed={oracle_gap}",
        "",
        "## Checks",
        "",
    ]
    for k, v in report["checks"].items():
        lines.append(f"- `{k}`: **{v}**")
    lines += ["", "## Best backend per family", ""]
    for fam, b in best_backend.items():
        lines.append(f"- {fam}: {b} (none={none_rates.get(fam)} mem={mem_rates.get(fam)} skill={skill_rates.get(fam)})")
    lines += [
        "",
        "If FAIL: do **not** train ExPolicy. Do not rewrite the project as a SkillOpt scheduler.",
        "Next substrate candidates: WebShop / ScienceWorld / LifelongAgentBench.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    (input_dir / "BACKEND_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    if decision == "MODEL_FOUNDATION_GATE_FAIL":
        (input_dir / "MODEL_FOUNDATION_GATE_FAIL.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "outputs" / "r8_backend_validation")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "r8_backend_validation" / "BACKEND_GATE")
    args = p.parse_args()
    print(json.dumps(analyze(args.input, args.out), indent=2))


if __name__ == "__main__":
    main()
