"""Official MemRL ALFWorld wiring with R8 split/model overrides (no vendor source edits)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

from expolicy.r8.constants import MEMRL_ROOT
from expolicy.r8.family import family_from_gamefile

if str(MEMRL_ROOT) not in sys.path:
    sys.path.insert(0, str(MEMRL_ROOT))


def run_memrl_official(*, config_yaml: Path, env_yaml: Path, work_dir: Path) -> dict:
    """Clone of run/run_alfworld.py main(), except env yaml comes from R8."""
    from memrl.agent.memp_agent import MempAgent
    from memrl.configs.config import MempConfig
    from memrl.providers.embedding import OpenAIEmbedder
    from memrl.providers.llm import OpenAILLM
    from memrl.run.alfworld_rl_runner import AlfworldRunner
    from memrl.service.memory_service import MemoryService
    from memrl.service.strategies import (
        BuildStrategy,
        RetrieveStrategy,
        StrategyConfiguration,
        UpdateStrategy,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = MempConfig.from_yaml(str(config_yaml))
    os.environ["OPENAI_API_KEY"] = str(cfg.llm.api_key)
    os.environ["OPENAI_API_BASE"] = str(cfg.llm.base_url)
    os.environ["OPENAI_BASE_URL"] = str(cfg.llm.base_url)
    os.environ.setdefault("MEMOS_DISABLE_NACOS", "1")
    log_dir = work_dir / "local_cache"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(work_dir / "memrl.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    llm = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_temperature=cfg.llm.temperature,
        default_max_tokens=cfg.llm.max_tokens,
        token_log_dir=str(log_dir),
    )
    embedder = OpenAIEmbedder(
        api_key=cfg.embedding.api_key,
        base_url=cfg.embedding.base_url,
        model=cfg.embedding.model,
        max_text_len=getattr(cfg.embedding, "max_text_len", 4096),
        token_log_dir=str(log_dir),
    )
    temp_dir = tempfile.mkdtemp(prefix="memp_r8_", dir=str(work_dir))
    mos_config = {
        "chat_model": {
            "backend": "openai",
            "config": {
                "model_name_or_path": cfg.llm.model,
                "api_key": cfg.llm.api_key,
                "api_base": cfg.llm.base_url,
            },
        },
        "mem_reader": {
            "backend": "simple_struct",
            "config": {
                "llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": cfg.llm.model,
                        "api_key": cfg.llm.api_key,
                        "api_base": cfg.llm.base_url,
                    },
                },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "model_name_or_path": cfg.embedding.model,
                            "api_key": cfg.embedding.api_key,
                            "base_url": cfg.embedding.base_url,
                            "embedding_dims": 3072,
                        },
                    },
                "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
            },
        },
        "user_manager": {"backend": "sqlite", "config": {"db_path": str(Path(temp_dir) / "users.db")}},
        "top_k": 5,
    }
    mos_path = Path(temp_dir) / "mos_config.json"
    mos_path.write_text(json.dumps(mos_config), encoding="utf-8")
    logging.info("constructing MemoryService mos=%s", mos_path)
    memory = MemoryService(
        mos_config_path=str(mos_path),
        llm_provider=llm,
        embedding_provider=embedder,
        strategy_config=StrategyConfiguration(
            BuildStrategy(cfg.memory.build_strategy),
            RetrieveStrategy(cfg.memory.retrieve_strategy),
            UpdateStrategy(cfg.memory.update_strategy),
        ),
        user_id=f"r8_{work_dir.name}",
        num_workers=cfg.experiment.batch_size,
        max_keywords=cfg.memory.max_keywords,
        add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
        enable_value_driven=cfg.experiment.enable_value_driven,
        rl_config=cfg.rl_config,
        db_max_concurrency=4,
        sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", None),
        sim_norm_std=getattr(cfg.memory, "sim_norm_std", None),
    )
    logging.info("MemoryService ready")
    few_shot_path = Path(cfg.experiment.few_shot_path)
    if not few_shot_path.is_absolute():
        few_shot_path = MEMRL_ROOT / few_shot_path
    few_shot = json.loads(few_shot_path.read_text(encoding="utf-8"))
    agent = MempAgent(llm_provider=llm, few_shot_examples=few_shot)
    runner = AlfworldRunner(
        agent=agent,
        root=str(MEMRL_ROOT),
        env_config=str(env_yaml),
        memory_service=memory,
        exp_name=cfg.experiment.experiment_name,
        ck_dir=str(log_dir),
        random_seed=cfg.experiment.random_seed,
        num_section=cfg.experiment.num_sections,
        batch_size=cfg.experiment.batch_size,
        max_steps=cfg.experiment.max_steps,
        rl_config=cfg.rl_config,
        bon=cfg.experiment.bon,
        retrieve_k=cfg.memory.k_retrieve,
        mode=cfg.experiment.mode,
        valid_interval=cfg.experiment.valid_interval,
        test_interval=0,
        dataset_ratio=cfg.experiment.dataset_ratio,
    )
    try:
        runner.run()
        return summarize_memrl_log(getattr(runner, "results_log", []) or [])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def evaluate_memrl_checkpoint(*, config_yaml: Path, env_yaml: Path, work_dir: Path, snapshot: Path | None) -> dict:
    """Reload MemoryService if snapshot exists, then eval-only on env yaml eval split."""
    from memrl.agent.memp_agent import MempAgent
    from memrl.configs.config import MempConfig
    from memrl.providers.embedding import OpenAIEmbedder
    from memrl.providers.llm import OpenAILLM
    from memrl.run.alfworld_rl_runner import AlfworldRunner
    from memrl.service.memory_service import MemoryService
    from memrl.service.strategies import (
        BuildStrategy,
        RetrieveStrategy,
        StrategyConfiguration,
        UpdateStrategy,
    )

    cfg = MempConfig.from_yaml(str(config_yaml))
    log_dir = work_dir / "eval_cache"
    log_dir.mkdir(parents=True, exist_ok=True)
    llm = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_temperature=cfg.llm.temperature,
        default_max_tokens=cfg.llm.max_tokens,
        token_log_dir=str(log_dir),
    )
    embedder = OpenAIEmbedder(
        api_key=cfg.embedding.api_key,
        base_url=cfg.embedding.base_url,
        model=cfg.embedding.model,
        max_text_len=getattr(cfg.embedding, "max_text_len", 4096),
        token_log_dir=str(log_dir),
    )
    temp_dir = tempfile.mkdtemp(prefix="memp_r8_eval_", dir=str(work_dir))
    mos_path = Path(temp_dir) / "mos_config.json"
    mos_path.write_text(
        json.dumps(
            {
                "chat_model": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": cfg.llm.model,
                        "api_key": cfg.llm.api_key,
                        "api_base": cfg.llm.base_url,
                    },
                },
                "user_manager": {"backend": "sqlite", "config": {"db_path": str(Path(temp_dir) / "users.db")}},
                "top_k": 5,
            }
        ),
        encoding="utf-8",
    )
    memory = MemoryService(
        mos_config_path=str(mos_path),
        llm_provider=llm,
        embedding_provider=embedder,
        strategy_config=StrategyConfiguration(
            BuildStrategy(cfg.memory.build_strategy),
            RetrieveStrategy(cfg.memory.retrieve_strategy),
            UpdateStrategy(cfg.memory.update_strategy),
        ),
        user_id=f"r8eval_{work_dir.name}",
        num_workers=cfg.experiment.batch_size,
        enable_value_driven=cfg.experiment.enable_value_driven,
        rl_config=cfg.rl_config,
        db_max_concurrency=4,
    )
    if snapshot and snapshot.exists():
        memory.load_checkpoint_snapshot(str(snapshot))
    few_shot_path = Path(cfg.experiment.few_shot_path)
    if not few_shot_path.is_absolute():
        few_shot_path = MEMRL_ROOT / few_shot_path
    agent = MempAgent(
        llm_provider=llm,
        few_shot_examples=json.loads(few_shot_path.read_text(encoding="utf-8")),
    )
    runner = AlfworldRunner(
        agent=agent,
        root=str(MEMRL_ROOT),
        env_config=str(env_yaml),
        memory_service=memory,
        exp_name=cfg.experiment.experiment_name + "_eval",
        ck_dir=str(log_dir),
        random_seed=cfg.experiment.random_seed,
        num_section=1,
        batch_size=cfg.experiment.batch_size,
        max_steps=cfg.experiment.max_steps,
        rl_config=cfg.rl_config,
        retrieve_k=cfg.memory.k_retrieve,
        mode="test",
        valid_interval=0,
        test_interval=0,
        dataset_ratio=1.0,
    )
    try:
        rate = runner._evaluate(runner.valid_game_files, "r8_future_or_calib", after_section=0)
        rows = [r for r in (runner.results_log or []) if r.get("mode") == "r8_future_or_calib"]
        return {"success": float(rate or 0.0), "log": rows}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def summarize_memrl_log(rows: list[dict]) -> dict:
    evals = [r for r in rows if str(r.get("mode", "")).startswith("eval") or "future" in str(r.get("mode", ""))]
    trains = [r for r in rows if r.get("mode") == "train_cumulative"]
    return {
        "n_log": len(rows),
        "last_train_success": trains[-1]["success"] if trains else None,
        "evals": evals,
    }


def family_rates_from_trajs(trajs: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[bool]] = {}
    for traj in trajs:
        fam = family_from_gamefile(str(traj.get("gamefile") or ""))
        if fam is None:
            continue
        buckets.setdefault(fam, []).append(bool(traj.get("success")))
    return {fam: (sum(v) / len(v) if v else 0.0) for fam, v in buckets.items()}
