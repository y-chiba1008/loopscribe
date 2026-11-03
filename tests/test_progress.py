"""progress の単体テスト。"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from loopscribe.progress import (
    ProgressReporter,
    format_duration,
    lag_seconds,
    model_loading_status,
    progress_ratio,
)


def test_format_duration() -> None:
    assert format_duration(0) == "00:00:00"
    assert format_duration(12.9) == "00:00:12"
    assert format_duration(3661) == "01:01:01"
    assert format_duration(-1.0) == "00:00:00"


def test_lag_and_ratio() -> None:
    assert lag_seconds(10.0, 15.0) == 5.0
    assert lag_seconds(15.0, 10.0) == 0.0
    assert progress_ratio(0.0, 0.0) == 0.0
    assert progress_ratio(5.0, 10.0) == 0.5
    assert progress_ratio(12.0, 10.0) == 1.0


def test_progress_reporter_updates_without_tty() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    with ProgressReporter(console, enabled=False) as reporter:
        assert reporter.enabled is False
        reporter.update(12.0, 30.0)
        reporter.update(30.0, 30.0)


def test_progress_reporter_enabled_flag() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=True, width=120, highlight=False)
    with ProgressReporter(console, enabled=True) as reporter:
        assert reporter.enabled is True
        reporter.update(90.0, 120.0)


def test_model_loading_status_non_tty() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    with model_loading_status(console, model_name="tiny"):
        pass
    text = stream.getvalue()
    assert "モデルを読み込み中: tiny" in text
    assert "モデルの読み込みが完了しました" in text
