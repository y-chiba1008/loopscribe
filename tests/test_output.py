"""output の単体テスト。"""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from loopscribe.errors import InvalidOutputNameError
from loopscribe.output import (
    AbsoluteSegment,
    ConsoleSink,
    MultiSink,
    TxtSink,
    format_header,
    format_line,
    format_timestamp,
    normalize_output_name,
    unique_output_path,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("meeting", "meeting"),
        ("meeting.txt", "meeting"),
        ("meeting.wav", "meeting"),
        ("meeting.TXT", "meeting"),
        ("meeting.wav.txt", "meeting.wav"),
        ("meeting.mp3", "meeting.mp3"),
        ("会議 11月", "会議 11月"),
    ],
)
def test_normalize_output_name(name: str, expected: str) -> None:
    assert normalize_output_name(name) == expected


@pytest.mark.parametrize("name", ["notes/meeting", "notes\\meeting", "../meeting", "C:\\x"])
def test_normalize_output_name_rejects_path_separators(name: str) -> None:
    with pytest.raises(InvalidOutputNameError, match="-o"):
        normalize_output_name(name)


@pytest.mark.parametrize("name", ["", "  ", ".txt", ".wav"])
def test_normalize_output_name_rejects_empty(name: str) -> None:
    with pytest.raises(InvalidOutputNameError, match="空"):
        normalize_output_name(name)


def test_unique_output_path_without_conflict(tmp_path: Path) -> None:
    assert unique_output_path(tmp_path, "meeting", ".txt") == tmp_path / "meeting.txt"


def test_unique_output_path_adds_number(tmp_path: Path) -> None:
    (tmp_path / "meeting.txt").write_text("")
    assert unique_output_path(tmp_path, "meeting", ".txt") == tmp_path / "meeting_2.txt"
    (tmp_path / "meeting_2.txt").write_text("")
    assert unique_output_path(tmp_path, "meeting", ".txt") == tmp_path / "meeting_3.txt"


def test_txt_sink_does_not_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "meeting.txt"
    path.write_text("既存", encoding="utf-8")
    with pytest.raises(FileExistsError):
        TxtSink(path, started_at=datetime(2026, 1, 1), model_name="small")
    assert path.read_text(encoding="utf-8") == "既存"


def test_unique_output_path_ignores_other_suffix(tmp_path: Path) -> None:
    (tmp_path / "meeting.wav").write_text("")
    assert unique_output_path(tmp_path, "meeting", ".txt") == tmp_path / "meeting.txt"


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(12.9) == "00:00:12"
    assert format_timestamp(3661) == "01:01:01"
    assert format_timestamp(3600 * 100) == "100:00:00"


def test_format_timestamp_rejects_negative() -> None:
    with pytest.raises(ValueError):
        format_timestamp(-0.1)


def test_format_line() -> None:
    assert format_line(41.2, "次の話題ですが…") == "[00:00:41] 次の話題ですが…"


def test_format_header() -> None:
    started = datetime(2026, 11, 2, 12, 34, 56)
    assert (
        format_header(started, model_name="small")
        == "# loopscribe 開始: 2026-11-02 12:34:56 / モデル: small"
    )


def test_format_header_with_input() -> None:
    started = datetime(2026, 11, 2, 12, 34, 56)
    assert (
        format_header(started, model_name="small", input_name="meeting.wav")
        == "# loopscribe 入力: meeting.wav / 処理開始: 2026-11-02 12:34:56 / モデル: small"
    )


def test_console_sink_prints_line() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    sink = ConsoleSink(console)
    sink.write(AbsoluteSegment(start=12.0, end=14.0, text="  こんにちは  "))
    sink.close()
    assert "[00:00:12] こんにちは" in stream.getvalue()


def test_console_sink_skips_blank() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    sink = ConsoleSink(console)
    sink.write(AbsoluteSegment(start=0.0, end=1.0, text="   "))
    assert stream.getvalue() == ""


def test_txt_sink_writes_header_and_flushed_lines(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    started = datetime(2026, 11, 2, 12, 34, 56)
    sink = TxtSink(path, started_at=started, model_name="small")
    sink.write(AbsoluteSegment(start=12.0, end=14.0, text="こんにちは"))
    sink.write(AbsoluteSegment(start=41.0, end=45.0, text="次の話題"))
    sink.close()

    text = path.read_text(encoding="utf-8")
    assert text == (
        "# loopscribe 開始: 2026-11-02 12:34:56 / モデル: small\n"
        "[00:00:12] こんにちは\n"
        "[00:00:41] 次の話題\n"
    )


def test_multi_sink_forwards(tmp_path: Path) -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    path = tmp_path / "out.txt"
    multi = MultiSink(
        [
            ConsoleSink(console),
            TxtSink(path, started_at=datetime(2026, 1, 1, 0, 0, 0), model_name="tiny"),
        ]
    )
    multi.write(AbsoluteSegment(start=5.0, end=6.0, text="テスト"))
    multi.close()
    assert "[00:00:05] テスト" in stream.getvalue()
    assert "[00:00:05] テスト" in path.read_text(encoding="utf-8")
