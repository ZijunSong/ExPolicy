"""Locked settings for the OS content-attribution round.

This is a new stage written after the calibration pilot. It does not change
D-v1 or D-v1.1 thresholds, and it does not authorize a router or a second model.
"""

from __future__ import annotations

PROTOCOL = {
    "stage_id": "D-content-os-v1",
    "parent_protocol": "D-v1.1",
    "formal_claim": False,
    "updater": "EPISODIC_REPLAY",
    "delta_reference": 0.02,
    "retention_weight": 0.5,
    "environment": "os",
    "inference_seed": 0,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "task_seed": 20261014,
    "order_seed": 20261014,
    "n_future": 32,
    "n_retention": 16,
    "history_future": 8,
    "history_retention": 8,
    "repeats": 2,
    "conditions": ["NO_UPDATE", "REAL", "EMPTY", "MATCHED_WRONG"],
    "batches": ["os-B1", "os-B2"],
    "prior_occupied_episodes": 1705,
    "cumulative_cap": 6000,
    "phase_episode_cap": 1400,
    "phase_token_cap": 12_000_000,
    "donor_episode_cap": 64,
    "retry_pool": 128,
    "length_tolerance": 0.20,
    "near_copy_jaccard": 0.85,
    "min_benefiting_future_tasks": 3,
    "min_benefiting_families": 2,
    "single_task_share": 0.80,
    "neutral_instruction": "Memory slot.",
    "neutral_action": "Noted.",
    "primary_contrasts": ["REAL-NO_UPDATE", "REAL-EMPTY", "REAL-MATCHED_WRONG"],
    "served_name": "qwen3-14b-r9",
    "port": 18102,
}
