"""Typer CLI for ExPolicy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from expolicy.cli.inspect import inspect_agent, inspect_experience

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command("setup")
def setup_cmd(
    seed: int = 13,
    world_path: Path = Path("data/world/helion.json"),
) -> None:
    from expolicy.environments.search.world import build_world, world_stats

    world = build_world(seed)
    world.save(world_path)
    stats = world_stats(world)
    console.print(json.dumps(stats, indent=2))
    console.print(f"[green]wrote[/green] {world_path}")


@app.command("run-agent")
def run_agent_cmd(
    config: Path = typer.Option(..., "--config", exists=True),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    from expolicy.cli.run import run_experiment

    summary = run_experiment(config, limit=limit)
    console.print(json.dumps(summary, indent=2, default=str))


@app.command("collect-experience")
def collect_cmd(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    from expolicy.cli.run import run_experiment

    summary = run_experiment(config, collect_only=True)
    console.print(json.dumps(summary, indent=2, default=str))


@app.command("acquire")
def acquire_cmd(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    from expolicy.cli.run import run_experiment

    summary = run_experiment(config)
    console.print(json.dumps(summary, indent=2, default=str))


@app.command("assimilate")
def assimilate_cmd(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    from expolicy.cli.run import run_experiment

    summary = run_experiment(config)
    console.print(json.dumps(summary, indent=2, default=str))


@app.command("evolve")
def evolve_cmd(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    from expolicy.cli.run import run_evolve

    summary = run_evolve(config)
    console.print(json.dumps(summary, indent=2, default=str))


@app.command("evaluate")
def evaluate_cmd(
    checkpoint: Path = typer.Argument(..., exists=True),
    config: Optional[Path] = typer.Option(None, "--config"),
    split: str = "validation",
    stage: int = 0,
    limit: int = 50,
) -> None:
    from expolicy.cli.evaluate import evaluate_checkpoint

    summary = evaluate_checkpoint(checkpoint, config=config, split=split, stage=stage, limit=limit)
    console.print(json.dumps(summary, indent=2, default=str))


@app.command("inspect-agent")
def inspect_agent_cmd(path: Path = typer.Argument(..., exists=True)) -> None:
    info = inspect_agent(path)
    console.print(json.dumps(info, indent=2, default=str))


@app.command("inspect-experience")
def inspect_experience_cmd(
    experience_id: str,
    run_dir: Path = typer.Option(Path("outputs"), "--run-dir"),
) -> None:
    info = inspect_experience(run_dir, experience_id)
    console.print(json.dumps(info, indent=2, default=str))


@app.command("train-policy")
def train_policy_cmd(
    buffer: Path = typer.Option(..., "--buffer", exists=True),
    out: Path = typer.Option(Path("outputs/experience_policy.npz"), "--out"),
    steps: int = 200,
) -> None:
    from expolicy.cli.train import train_policy

    summary = train_policy(buffer, out, steps=steps)
    console.print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    app()
