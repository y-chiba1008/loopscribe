"""cli の単体テスト。"""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from loopscribe.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "loopscribe" in result.stdout
    assert "0.1.0" in result.stdout


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "start" in result.stdout
    assert "download-model" in result.stdout
    assert "devices" in result.stdout


def test_start_rejects_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    result = runner.invoke(app, ["start"])
    assert result.exit_code == 1
    assert "Windows" in result.stdout


def test_devices_rejects_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    result = runner.invoke(app, ["devices"])
    assert result.exit_code == 1
    assert "Windows" in result.stdout


def test_start_help_shows_options() -> None:
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code == 0
    assert "--device" in result.stdout
    assert "--output-dir" in result.stdout


def test_download_model_help() -> None:
    result = runner.invoke(app, ["download-model", "--help"])
    assert result.exit_code == 0
