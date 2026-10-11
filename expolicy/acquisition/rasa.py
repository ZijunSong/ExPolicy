"""RASA: Remediability-Aware Sequential Acquisition.

Thompson sampling over z = q * p, with frozen semantic-cluster diversity.
Does not read future labels, gold docs, or true capability names for ranking.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from expolicy.acquisition.base import AcquisitionPolicy
from expolicy.assimilation.readiness_gate import (
    PRIOR_ALPHA,
    PRIOR_BETA,
    STATE_ACTIVE,
    STATE_PARKED,
    STATE_UNRESOLVED,
    apply_gate,
    sample_z,
)
from expolicy.core.capability import CapabilityProfile
from expolicy.core.experience import AcquisitionProposal, Experience
from expolicy.core.state import PersistentAgentState
from expolicy.environments.public_search.capability_map import PUBLIC_CAPS
from expolicy.environments.search.world import Task

WARM_PER_CAP = 2
WARM_TOTAL = 6
N_MC = 4096


def _empty_cap() -> dict[str, Any]:
    return {
        "alpha_q": PRIOR_ALPHA,
        "beta_q": PRIOR_BETA,
        "alpha_p": PRIOR_ALPHA,
        "beta_p": PRIOR_BETA,
        "n_tested_base": 0,
        "n_base_success": 0,
        "n_tested_base_failures": 0,
        "repair_count": 0,
        "n_skill_replay": 0,
        "cluster_coverage": 0,
        "tested_failure_ids": [],
        "acquired_ids": [],
        "failure_cluster_ids": [],
        "state": STATE_UNRESOLVED,
        "P_good": None,
        "reason": "init",
        "budget_to_terminal": None,
        "mean_q": 0.5,
        "mean_p": 0.5,
        "mean_z": 0.25,
    }


class RASAAcquisition(AcquisitionPolicy):
    name = "rasa"

    def __init__(
        self,
        seed: int = 0,
        *,
        predicted_cap: Mapping[str, str] | None = None,
        clusters: Mapping[str, Mapping[str, int]] | None = None,
        embeddings: Mapping[str, Sequence[float]] | None = None,
        qonly: bool = False,
        diversity: bool = True,
        caps: Sequence[str] = PUBLIC_CAPS,
    ) -> None:
        self.seed = int(seed)
        self.qonly = bool(qonly)
        self.diversity = bool(diversity)
        self.caps = list(caps)
        self.predicted_cap = {str(k): str(v) for k, v in dict(predicted_cap or {}).items()}
        self.clusters = {str(tid): dict(v) for tid, v in dict(clusters or {}).items()}
        self.embeddings = {str(tid): np.asarray(vec, dtype=np.float32) for tid, vec in dict(embeddings or {}).items()}
        self.rng = np.random.RandomState(self.seed)
        self.state: dict[str, dict[str, Any]] = {c: _empty_cap() for c in self.caps}
        self.n_unique = 0
        self.warm_done = False
        self._warm_queue: list[str] = []

    def cap_of(self, task: Task | str) -> str:
        tid = task.task_id if isinstance(task, Task) else str(task)
        if tid in self.predicted_cap:
            return self.predicted_cap[tid]
        if isinstance(task, Task):
            pred = (task.metadata or {}).get("predicted_capability")
            if pred:
                return str(pred)
            return str(task.capability or "")
        return ""

    def cluster_of(self, task_id: str) -> int | None:
        rec = self.clusters.get(str(task_id)) or {}
        if "cluster" not in rec:
            return None
        return int(rec["cluster"])

    def unresolved(self) -> list[str]:
        return [c for c in self.caps if self.state[c]["state"] == STATE_UNRESOLVED]

    def all_terminal(self) -> bool:
        return not self.unresolved()

    def _acquired_set(self) -> set[str]:
        out: set[str] = set()
        for rec in self.state.values():
            out.update(str(x) for x in rec.get("acquired_ids") or [])
        return out

    def bind_warm_start(self, pool: Sequence[Task]) -> list[str]:
        if self._warm_queue or self.warm_done:
            return list(self._warm_queue)
        chosen: list[str] = []
        acquired = self._acquired_set()
        for cap in self.caps:
            if len(chosen) >= WARM_TOTAL:
                break
            cands = [
                t
                for t in pool
                if self.cap_of(t) == cap and t.task_id not in acquired and t.task_id not in chosen
            ]
            if len(cands) < WARM_PER_CAP:
                raise RuntimeError(f"RASA warm-start: predicted cap {cap} has {len(cands)} remaining tasks, need {WARM_PER_CAP}")
            picked: list[Task] = []
            used_clusters: set[int] = set()
            if self.diversity:
                by_cl: dict[int, list[Task]] = {}
                for t in cands:
                    cl = self.cluster_of(t.task_id)
                    if cl is None:
                        cl = -1
                    by_cl.setdefault(cl, []).append(t)
                # prefer distinct clusters
                for cl in sorted(by_cl):
                    if len(picked) >= WARM_PER_CAP:
                        break
                    picked.append(by_cl[cl][0])
                    used_clusters.add(cl)
                if len(picked) < WARM_PER_CAP:
                    for t in cands:
                        if t not in picked:
                            picked.append(t)
                        if len(picked) >= WARM_PER_CAP:
                            break
            else:
                order = list(cands)
                self.rng.shuffle(order)
                picked = order[:WARM_PER_CAP]
            chosen.extend(t.task_id for t in picked[:WARM_PER_CAP])
        self._warm_queue = chosen[:WARM_TOTAL]
        return list(self._warm_queue)

    def _least_covered_cluster(self, cap: str, cands: Sequence[Task]) -> list[Task]:
        covered = {int(x) for x in (self.state[cap].get("failure_cluster_ids") or [])}
        scores = []
        for t in cands:
            cl = self.cluster_of(t.task_id)
            cl = -1 if cl is None else int(cl)
            cover_rank = 1 if cl in covered else 0
            n_in = sum(1 for s in cands if (self.cluster_of(s.task_id) if self.cluster_of(s.task_id) is not None else -1) == cl)
            scores.append((cover_rank, n_in, cl, t))
        # uncovered first, then smaller clusters
        scores.sort(key=lambda x: (x[0], x[1], x[2]))
        if not scores:
            return []
        best_cl = scores[0][2]
        return [t for t in cands if (self.cluster_of(t.task_id) if self.cluster_of(t.task_id) is not None else -1) == best_cl]

    def _most_dissimilar(self, cap: str, cands: Sequence[Task]) -> Task:
        acquired = [str(x) for x in self.state[cap].get("acquired_ids") or []]
        refs = [self.embeddings[i] for i in acquired if i in self.embeddings]
        if not refs:
            return cands[0]
        mat = np.stack(refs, axis=0)
        best_t = cands[0]
        best_d = -1.0
        for t in cands:
            vec = self.embeddings.get(t.task_id)
            if vec is None:
                d = 0.0
            else:
                sims = mat @ vec
                d = float(1.0 - float(np.max(sims)))
            if d > best_d:
                best_d = d
                best_t = t
        return best_t

    def _select_in_cap(self, cap: str, pool: Sequence[Task]) -> Task | None:
        acquired = self._acquired_set()
        cands = [t for t in pool if self.cap_of(t) == cap and t.task_id not in acquired]
        if not cands:
            return None
        if not self.diversity:
            idx = int(self.rng.randint(0, len(cands)))
            return cands[idx]
        subset = self._least_covered_cluster(cap, cands) or list(cands)
        return self._most_dissimilar(cap, subset)

    def _thompson_cap(self) -> str | None:
        open_caps = self.unresolved()
        if not open_caps:
            return None
        best_c = open_caps[0]
        best_z = -1.0
        for cap in open_caps:
            rec = self.state[cap]
            z = sample_z(
                rec["alpha_q"],
                rec["beta_q"],
                rec["alpha_p"],
                rec["beta_p"],
                n=1,
                rng=self.rng,
                qonly=self.qonly,
            )[0]
            if float(z) > best_z:
                best_z = float(z)
                best_c = cap
        return best_c

    def propose(
        self,
        agent_state: PersistentAgentState,
        capability_profile: CapabilityProfile,
        candidate_pool: Sequence[Task],
        budget: int,
        *,
        recent: Sequence[Experience] | None = None,
    ) -> list[AcquisitionProposal]:
        if self.all_terminal():
            return [
                AcquisitionProposal(
                    mode=self.name,
                    task_ids=[],
                    rationale="all capabilities terminal; early stop",
                    metadata={"early_stop": True},
                )
            ]
        want = max(1, int(budget))
        chosen: list[str] = []
        scores: dict[str, float] = {}
        if not self.warm_done:
            queue = self.bind_warm_start(candidate_pool)
            while queue and len(chosen) < want:
                tid = queue.pop(0)
                self._warm_queue = queue
                chosen.append(tid)
                scores[tid] = 1.0
            if not queue:
                self.warm_done = True
        while len(chosen) < want and not self.all_terminal():
            cap = self._thompson_cap()
            if cap is None:
                break
            task = self._select_in_cap(cap, candidate_pool)
            if task is None:
                # no remaining tasks for this cap: park it so we do not spin
                rec = self.state[cap]
                if rec["state"] == STATE_UNRESOLVED:
                    rec["state"] = STATE_PARKED
                    rec["reason"] = "no_remaining_tasks"
                continue
            chosen.append(task.task_id)
            rec = self.state[cap]
            scores[task.task_id] = float(rec.get("mean_z") or 0.0)
        return [
            AcquisitionProposal(
                mode="rasa_qonly" if self.qonly else self.name,
                task_ids=chosen,
                scores=scores,
                rationale="warm-start then Thompson over repairable deficit z=q*p (or q-only)",
                metadata={
                    "qonly": self.qonly,
                    "diversity": self.diversity,
                    "unresolved": self.unresolved(),
                    "early_stop": self.all_terminal() and not chosen,
                },
            )
        ]

    def observe_base(self, task_id: str, success: bool, *, unique_budget: int) -> dict[str, Any]:
        cap = self.predicted_cap.get(str(task_id), "")
        if cap not in self.state:
            return {}
        rec = self.state[cap]
        if str(task_id) not in rec["acquired_ids"]:
            rec["acquired_ids"].append(str(task_id))
            self.n_unique += 1
        rec["n_tested_base"] = int(rec["n_tested_base"]) + 1
        if success:
            rec["beta_q"] = float(rec["beta_q"]) + 1.0
            rec["n_base_success"] = int(rec["n_base_success"]) + 1
        else:
            rec["alpha_q"] = float(rec["alpha_q"]) + 1.0
            rec["n_tested_base_failures"] = int(rec["n_tested_base_failures"]) + 1
            rec["tested_failure_ids"].append(str(task_id))
            cl = self.cluster_of(str(task_id))
            if cl is not None and int(cl) not in rec["failure_cluster_ids"]:
                rec["failure_cluster_ids"].append(int(cl))
            rec["cluster_coverage"] = len(rec["failure_cluster_ids"])
        return rec

    def observe_repair(self, task_id: str, repaired: bool) -> dict[str, Any]:
        cap = self.predicted_cap.get(str(task_id), "")
        if cap not in self.state:
            return {}
        rec = self.state[cap]
        rec["n_skill_replay"] = int(rec["n_skill_replay"]) + 1
        if repaired:
            rec["alpha_p"] = float(rec["alpha_p"]) + 1.0
            rec["repair_count"] = int(rec["repair_count"]) + 1
        else:
            rec["beta_p"] = float(rec["beta_p"]) + 1.0
        return rec

    def apply_readiness(self, *, unique_budget: int, use_srt: bool) -> dict[str, Any]:
        if not use_srt:
            return {c: dict(self.state[c]) for c in self.caps}
        for cap in self.caps:
            rec = self.state[cap]
            if rec["state"] in {STATE_ACTIVE, STATE_PARKED}:
                continue
            updated = apply_gate(rec)
            if updated["state"] in {STATE_ACTIVE, STATE_PARKED} and rec.get("budget_to_terminal") is None:
                updated["budget_to_terminal"] = int(unique_budget)
            self.state[cap] = updated
        return {c: dict(self.state[c]) for c in self.caps}

    def snapshot(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "qonly": self.qonly,
            "diversity": self.diversity,
            "n_unique": self.n_unique,
            "warm_done": self.warm_done,
            "warm_queue": list(self._warm_queue),
            "state": self.state,
            "rng_state": self.rng.get_state()[1].tolist(),
            "rng_pos": int(self.rng.get_state()[2]),
        }

    def load_snapshot(self, payload: Mapping[str, Any]) -> None:
        self.qonly = bool(payload.get("qonly", self.qonly))
        self.diversity = bool(payload.get("diversity", self.diversity))
        self.n_unique = int(payload.get("n_unique") or 0)
        self.warm_done = bool(payload.get("warm_done"))
        self._warm_queue = list(payload.get("warm_queue") or [])
        raw = dict(payload.get("state") or {})
        for cap in self.caps:
            if cap in raw:
                rec = _empty_cap()
                rec.update(raw[cap])
                self.state[cap] = rec
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            pos = int(payload.get("rng_pos") or 624)
            self.rng.set_state(("MT19937", np.array(rng_state, dtype=np.uint32), pos, 0, 0.0))
