"""Emit MemRL / SkillOpt YAML and env configs that only override I/O + model endpoints."""

from __future__ import annotations

from pathlib import Path

import yaml

from expolicy.r8.constants import ALFWORLD_DATA, MEMRL_ROOT, MODEL_NAME, SKILLOPT_ROOT


def memrl_env_yaml(*, train_view: Path, eval_view: Path) -> dict:
    return {
        "dataset": {
            "data_path": str(train_view),
            "eval_id_data_path": str(eval_view),
            "eval_ood_data_path": str(eval_view),
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": str(ALFWORLD_DATA / "logic" / "alfred.pddl"),
            "grammar": str(ALFWORLD_DATA / "logic" / "alfred.twl2"),
        },
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "controller": {"type": "oracle", "debug": False, "load_receps": True},
        "general": {
            "random_seed": 42,
            "use_cuda": False,
            "visdom": False,
            "task": "alfred",
            "training_method": "dagger",
            "save_path": "./training/",
            "observation_pool_capacity": 3,
            "hide_init_receptacles": False,
            "training": {
                "batch_size": 10,
                "max_episode": 50000,
                "smoothing_eps": 0.1,
                "optimizer": {"learning_rate": 0.001, "clip_grad_norm": 5},
            },
            "evaluate": {"run_eval": True, "batch_size": 10, "env": {"type": "AlfredTWEnv"}},
            "checkpoint": {
                "report_frequency": 1000,
                "experiment_tag": "r8",
                "load_pretrained": False,
                "load_from_tag": "",
            },
        },
        "dagger": {
            "action_space": "generation",
            "max_target_length": 20,
            "beam_width": 10,
            "generate_top_k": 5,
            "training": {"max_nb_steps_per_episode": 50},
            "fraction_assist": {
                "fraction_assist_anneal_episodes": 50000,
                "fraction_assist_anneal_from": 1.0,
                "fraction_assist_anneal_to": 0.01,
            },
            "fraction_random": {
                "fraction_random_anneal_episodes": 0,
                "fraction_random_anneal_from": 0.0,
                "fraction_random_anneal_to": 0.0,
            },
            "replay": {
                "replay_memory_capacity": 500000,
                "update_per_k_game_steps": 5,
                "replay_batch_size": 64,
                "replay_sample_history_length": 4,
                "replay_sample_update_from": 2,
            },
        },
    }


def memrl_run_yaml(
    *,
    env_yaml: Path,
    llm_base: str,
    embed_base: str,
    seed: int,
    out_dir: Path,
    num_sections: int,
    batch_size: int,
    max_steps: int,
    dataset_ratio: float = 1.0,
    valid_interval: int = 0,
    experiment_name: str = "r8_memrl",
) -> dict:
    return {
        "llm": {
            "provider": "openai",
            "api_key": "EMPTY",
            "base_url": llm_base,
            "model": MODEL_NAME,
            "temperature": 0,
            "max_tokens": 256,
        },
        "embedding": {
            "provider": "openai",
            "api_key": "EMPTY",
            "base_url": embed_base,
            "model": "text-embedding-3-large",
            "max_text_len": 4096,
        },
        "memory": {
            "build_strategy": "proceduralization",
            "retrieve_strategy": "query",
            "update_strategy": "adjustment",
            "k_retrieve": 5,
            "max_keywords": 8,
            "confidence_threshold": 0.0,
            "memory_confidence": 100.0,
            "add_similarity_threshold": 0.90,
            "mos_config_path": "configs/mos_config_final.json",
            "user_id": "memrl_user",
            "sim_norm_mean": 0.5187,
            "sim_norm_std": 0.1203,
        },
        "environment": {
            "alfworld_config_path": str(env_yaml),
            "alfworld_env_type": "AlfredTWEnv",
        },
        "experiment": {
            "random_seed": seed,
            "enable_value_driven": True,
            "experiment_name": experiment_name,
            "mode": "train",
            "num_sections": num_sections,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "dataset_ratio": dataset_ratio,
            "few_shot_path": str(MEMRL_ROOT / "configs" / "alfworld" / "alfworld_examples.json"),
            "baseline_mode": "none",
            "baseline_k": 10,
            "valid_interval": valid_interval,
            "test_interval": 0,
            "output_dir": str(out_dir),
            "save_trajectories": True,
            "save_memories": True,
            "ckpt_eval_enabled": False,
            "ckpt_eval_path": "",
            "ckpt_resume_enabled": False,
            "ckpt_resume_path": "",
            "ckpt_resume_epoch": None,
        },
        "rl_config": {
            "epsilon": 0,
            "tau": 0.62,
            "alpha": 0.3,
            "gamma": 0.0,
            "q_init_pos": 0,
            "q_init_neg": 0,
            "success_reward": 1.0,
            "failure_reward": -1.0,
            "sim_threshold": 0.62,
            "topk": 3,
            "novelty_threshold": 0.85,
            "recency_boost": 0.0,
            "reward_merge_gain": 0.1,
            "q_min_threshold": -10,
            "weight_sim": 0.5,
            "weight_q": 0.5,
        },
    }


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


SKILLOPT_INITIAL = str(SKILLOPT_ROOT / "skillopt" / "envs" / "alfworld" / "skills" / "initial.md")
