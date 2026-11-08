"""progress の単体テスト。"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from loopscribe.progress import (
    ProgressReporter,
    RecordingReporter,
    format_duration,
    format_size,
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


def test_format_size() -> None:
    assert format_size(0) == "0.0MB"
    assert format_size(44) == "0.0MB"
    assert format_size(115 * 1024 * 1024) == "115.0MB"
    assert format_size(-1) == "0.0MB"


def test_recording_reporter_disabled_without_tty() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    with RecordingReporter(console) as reporter:
        assert reporter.enabled is False
        reporter.update(3.0, 96_044)
    assert stream.getvalue() == ""


def test_recording_reporter_shows_time_and_size() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=True, width=120, highlight=False)
    with RecordingReporter(console) as reporter:
        assert reporter.enabled is True
        reporter.update(3661.0, 2 * 1024 * 1024)
    text = stream.getvalue()
    assert "録音 01:01:01" in text
    assert "2.0MB" in text


def test_model_loading_status_non_tty() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    with model_loading_status(console, model_name="tiny"):
        pass
    text = stream.getvalue()
    assert "モデルを読み込み中: tiny" in text
    assert "モデルの読み込みが完了しました" in text
