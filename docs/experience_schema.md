# Experience schema

All interactions serialize through Pydantic models in `expolicy/core/experience.py`.

## Experience

Required fields: `experience_id`, `task_id`, `timestamp`, `capability_context`, `initial_state`, `observations`, `actions`, `tool_calls`, `outcome`, `reward`, `success`, `failure_type`, `evidence`, `provenance`, `uncertainty`, `cost` (tokens, latency, environment_steps, api_cost, gpu_seconds, training_tokens, tool_calls), `parent_experience_id`, `branch_origin`.

## Derived objects

- `ExperienceVerification` — validity, evidence/consistency/provenance scores, rejection reasons
- `ExperienceAbstraction` — preconditions, situation, decision_point, action_pattern, evidence, outcome, failure_mode, correction, reusable_principle, applicability_scope, not_applicable_when, confidence, provenance
- `ExperienceUtility` — realized future/transfer/retention gains and costs
- `AssimilationDecision` — carrier, confidence, expected_utility, expected_interference, update_spec
- `UtilityRecord` — policy-training row (no future gold answers)

## Lifecycle

`acquired → verified → abstracted → evaluated → routed → used → re-evaluated → merged | superseded | deprecated`

Illegal transitions raise `InvalidExperienceError`. Invalid experiences cannot enter weight update.

## Leakage

Acquisition and the Experience Policy may see agent state, past trajectories, capability profiles, candidate metadata, past utility, and budget. They must not see held-out answers, future rewards, or oracle utilities (oracle is analysis-only).
