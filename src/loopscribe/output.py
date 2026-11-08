"""文字起こし結果の出力(コンソールと txt)。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, TextIO

from rich.console import Console

from loopscribe.constants import OUTPUT_ENCODING, OUTPUT_NAME_STRIP_SUFFIXES
from loopscribe.errors import InvalidOutputNameError

_PATH_SEPARATORS = ("/", "\\")


def normalize_output_name(
    name: str,
    *,
    strip_suffixes: tuple[str, ...] = OUTPUT_NAME_STRIP_SUFFIXES,
) -> str:
    """`-n/--output-name` の値を、拡張子なしの名前にする。

    末尾の拡張子(`strip_suffixes`。大文字小文字は区別しない)は1つだけ取り除く。
    パスの区切り文字を含む名前と、空の名前は `InvalidOutputNameError`。
    """
    if any(sep in name for sep in _PATH_SEPARATORS):
        raise InvalidOutputNameError(
            f"出力名にパスの区切り文字は使えません(保存先は -o で指定してください): {name}"
        )
    stem = name
    lower = name.lower()
    for suffix in strip_suffixes:
        if lower.endswith(suffix.lower()):
            stem = name[: -len(suffix)]
            break
    if not stem.strip():
        raise InvalidOutputNameError(f"出力名が空です: {name!r}")
    return stem


def unique_output_path(output_dir: Path, stem: str, suffix: str) -> Path:
    """`output_dir/stem+suffix` を返す。同名のファイルがあれば `_2`、`_3` … を付ける。"""
    path = output_dir / f"{stem}{suffix}"
    number = 2
    while path.exists():
        path = output_dir / f"{stem}_{number}{suffix}"
        number += 1
    return path


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
    """UTF-8 の txt へ1セグメントごとに flush する Sink。

    既存のファイルは上書きしない(あれば `FileExistsError`)。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        started_at: datetime,
        model_name: str,
        encoding: str = OUTPUT_ENCODING,
    ) -> None:
        self._path = Path(path)
        self._file: TextIO = self._path.open("x", encoding=encoding, newline="\n")
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
