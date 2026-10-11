#!/usr/bin/env python3
"""Analyze R4 carrier matrix and write GO_NO_GO_R4.md."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from expolicy.experiments.common import ROOT, mean, write_json
from expolicy.experiments.r_common import add_common_args, apply_gpu

DEFAULT_OUT = str(ROOT / "outputs" / "r4_batch_carrier_matrix")


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    args = p.parse_args()
    apply_gpu(args.gpu)
    out = Path(args.output_dir)
    matrix_path = out / "carrier_matrix.json"
    if not matrix_path.exists():
        raise SystemExit(f"missing {matrix_path}")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    carriers = ["episodic", "semantic", "skill", "weight"]

    n = len(matrix)
    n_pos = 0
    unique_best: dict[str, int] = defaultdict(int)
    mean_u: dict[str, list[float]] = defaultdict(list)
    oracle = []
    fixed = defaultdict(list)
    for row in matrix:
        utils = {
            c: cell["U_capability"]
            for c, cell in (row.get("carriers") or {}).items()
            if cell.get("status") == "ok"
        }
        if not utils:
            continue
        if max(utils.values()) > 0:
            n_pos += 1
        best_u = max(utils.values())
        bests = [c for c, v in utils.items() if v == best_u]
        if len(bests) == 1:
            unique_best[bests[0]] += 1
        oracle.append(best_u)
        for c in carriers:
            if c in utils:
                mean_u[c].append(utils[c])
                fixed[c].append(utils[c])

    mean_by_c = {c: mean(vs) for c, vs in mean_u.items()}
    global_best = max(mean_by_c, key=mean_by_c.get) if mean_by_c else None
    n_change = sum(1 for c, k in unique_best.items() if c != global_best)
    oracle_mean = mean(oracle)
    best_fixed_mean = mean_by_c.get(global_best, 0.0) if global_best else 0.0
    oracle_better = oracle_mean > best_fixed_mean + 1e-9
    grain = 1.0 / 16.0
    amplitude_ok = abs(oracle_mean - best_fixed_mean) > grain or any(
        abs((row.get("carriers") or {}).get(c, {}).get("U_capability") or 0) > grain
        for row in matrix
        for c in carriers
    )

    a = n_pos / max(n, 1) >= 0.5
    b = len(unique_best) >= 2 and n_change > 0
    c = oracle_better
    d = amplitude_ok
    go = a and b and c and d

    lines = [
        "# GO / NO-GO R4",
        "",
        f"n_batches = {n}",
        f"batches with some carrier U_capability > 0: {n_pos}/{n} ({n_pos / max(n, 1):.2f})",
        f"mean U_capability by carrier: {json.dumps(mean_by_c, indent=2)}",
        f"unique best counts: {dict(unique_best)}",
        f"global best fixed carrier: {global_best} (mean U={best_fixed_mean:.4f})",
        f"hindsight oracle mean U: {oracle_mean:.4f}",
        "",
        "## Gates",
        f"- A. >=50% batches have some U_capability > 0: **{'PASS' if a else 'FAIL'}**",
        f"- B. best carrier varies (not one global ruler): **{'PASS' if b else 'FAIL'}**",
        f"- C. oracle > best fixed: **{'PASS' if c else 'FAIL'}** ({oracle_mean:.4f} vs {best_fixed_mean:.4f})",
        f"- D. amplitude > 1/16 probe grain: **{'PASS' if d else 'FAIL'}**",
        "",
        f"## Decision: **{'GO' if go else 'NO-GO'}**",
        "",
        "GO means experience can be absorbed and carrier choice materially changes realized capability gain. Then R5 is allowed.",
        "NO-GO means pause Adaptive Assimilation; do not train a learned router; do not scale to E5–E8.",
        "",
        "U_capability = Δnear + 0.5 Δfar − 0.5 max(0, −Δretention). Token cost is reported separately as U_efficiency and is not used for this gate.",
    ]
    (out / "GO_NO_GO_R4.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out / "GO_NO_GO_R4.json",
        {
            "go": go,
            "gates": {"A": a, "B": b, "C": c, "D": d},
            "n": n,
            "n_pos": n_pos,
            "mean_u": mean_by_c,
            "unique_best": dict(unique_best),
            "oracle_mean": oracle_mean,
            "best_fixed": global_best,
            "best_fixed_mean": best_fixed_mean,
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
