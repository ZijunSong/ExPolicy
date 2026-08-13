"""Smoke: CLI help and world setup."""

from __future__ import annotations

from typer.testing import CliRunner

from expolicy.cli.main import app


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-agent" in result.stdout
    assert "evolve" in result.stdout
    assert "inspect-agent" in result.stdout
