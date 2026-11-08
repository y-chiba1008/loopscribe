"""文字起こし結果の出力(コンソールと txt)。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, TextIO

from rich.console import Console

from loopscribe.constants import OUTPUT_ENCODING


@dataclass(frozen=True)
class AbsoluteSegment:
    """録音開始からの絶対時刻を持つセグメント。"""

    start: float
    end: float
    text: str


class Sink(Protocol):
    """セグメントを受け取る出力先。"""

    def write(self, segment: AbsoluteSegment) -> None: ...

    def close(self) -> None: ...


def format_timestamp(seconds: float) -> str:
    """経過秒を [HH:MM:SS] にする。時間は 100 以上でもよい。"""
    if seconds < 0:
        raise ValueError("seconds は 0 以上であること")
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_line(start_seconds: float, text: str) -> str:
    """本文1行。コンソールと txt で共通。"""
    return f"[{format_timestamp(start_seconds)}] {text}"


def format_header(
    started_at: datetime,
    *,
    model_name: str,
) -> str:
    """txt 先頭のヘッダ行。"""
    stamp = started_at.strftime("%Y-%m-%d %H:%M:%S")
    return f"# loopscribe 開始: {stamp} / モデル: {model_name}"


class ConsoleSink:
    """rich.Console へ本文を出す Sink。"""

    def __init__(self, console: Console) -> None:
        self._console = console

    def write(self, segment: AbsoluteSegment) -> None:
        text = segment.text.strip()
        if not text:
            return
        self._console.print(format_line(segment.start, text))

    def close(self) -> None:
        return


class TxtSink:
    """UTF-8 の txt へ1セグメントごとに flush する Sink。"""

    def __init__(
        self,
        path: Path | str,
        *,
        started_at: datetime,
        model_name: str,
        encoding: str = OUTPUT_ENCODING,
    ) -> None:
        self._path = Path(path)
        self._file: TextIO = self._path.open("w", encoding=encoding, newline="\n")
        self._file.write(format_header(started_at, model_name=model_name) + "\n")
        self._file.flush()

    def write(self, segment: AbsoluteSegment) -> None:
        text = segment.text.strip()
        if not text:
            return
        self._file.write(format_line(segment.start, text) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class MultiSink:
    """複数の Sink へ同じセグメントを渡す。"""

    def __init__(self, sinks: list[Sink]) -> None:
        self._sinks = list(sinks)

    def write(self, segment: AbsoluteSegment) -> None:
        for sink in self._sinks:
            sink.write(segment)

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()
