#!/usr/bin/env python3
"""2-round x batch=2 closed-loop smoke. Fail if posterior does not change."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from expolicy.closed_loop.constants import R7_ROOT
from expolicy.closed_loop.io_utils import read_json, write_json
from expolicy.closed_loop.runner.config import R7Config
from expolicy.closed_loop.runner.episode import run_r7_episode
from expolicy.experiments.r_common import apply_gpu


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "5"))
    p.add_argument("--retriever-url", default=os.environ.get("EXPOLICY_RETRIEVER_URL", "http://127.0.0.1:18080"))
    p.add_argument("--output-dir", default=str(R7_ROOT / "smoke"))
    args = p.parse_args()
    apply_gpu(args.gpu)
    out = Path(args.output_dir)
    cfg = R7Config(
        condition="expolicy",
        seed=0,
        rounds=2,
        batch_size=2,
        budget=4,
        retriever_url=args.retriever_url,
        full_eval_at=(0,),
        smoke=True,
    )
    run_r7_episode(cfg, output_root=out, resume=True)
    seed_dir = out / "expolicy" / "seed0"
    r0 = read_json(seed_dir / "round_00" / "policy_after.json")
    r1 = read_json(seed_dir / "round_01" / "policy_before.json")
    u0 = read_json(seed_dir / "round_00" / "policy_update.json")
    u1 = read_json(seed_dir / "round_01" / "policy_update.json")
    if r0 != r1:
        # round_01 policy_before must equal round_00 policy_after
        pass
    if read_json(seed_dir / "round_00" / "policy_before.json") == r0:
        raise SystemExit("SMOKE FAIL: policy checkpoint unchanged after round 0")
    if float(u0["acquisition"].get("parameter_delta_norm") or 0) <= 0:
        raise SystemExit("SMOKE FAIL: acquisition posterior delta is 0")
    if float(u0["assimilation"].get("parameter_delta_norm") or 0) <= 0:
        raise SystemExit("SMOKE FAIL: assimilation posterior delta is 0")
    if r0 != read_json(seed_dir / "round_01" / "policy_before.json"):
        raise SystemExit("SMOKE FAIL: round 1 did not read updated posterior")
    trace = [
        "# R7 Smoke Trace",
        "",
        "S0 → acquisition → 2 real trajectories → assimilation → persistent change → shadow → reward → posterior update → S1 → next acquisition",
        "",
        f"- round0 acq delta: {u0['acquisition'].get('parameter_delta_norm')}",
        f"- round0 assim delta: {u0['assimilation'].get('parameter_delta_norm')}",
        f"- round1 acq delta: {u1['acquisition'].get('parameter_delta_norm')}",
        f"- round0 acq action: {read_json(seed_dir / 'round_00' / 'acquisition_decision.json').get('capability')}",
        f"- round1 acq action: {read_json(seed_dir / 'round_01' / 'acquisition_decision.json').get('capability')}",
        f"- round0 assim action: {read_json(seed_dir / 'round_00' / 'assimilation_decision.json').get('action')}",
        "",
        "SMOKE PASS",
        "",
    ]
    (out / "SMOKE_TRACE.md").parent.mkdir(parents=True, exist_ok=True)
    # smoke output lives under outputs/r7_closed_loop/smoke and also condition dir
    smoke_dir = Path(args.output_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "SMOKE_TRACE.md").write_text("\n".join(trace), encoding="utf-8")
    write_json(smoke_dir / "SMOKE_PASS.json", {"pass": True, "seed_dir": str(seed_dir)})
    print("SMOKE PASS", seed_dir)


if __name__ == "__main__":
    main()
