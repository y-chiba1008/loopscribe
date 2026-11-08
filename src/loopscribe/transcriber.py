"""文字起こしのプロトコルと、faster-whisper の結果の変換(モデル共通)。

モデルごとの推論は `loopscribe.models` に置く。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Word:
    """チャンク内の相対時刻を持つ語。text は前後の空白を含めてそのまま持つ。"""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    """チャンク内の相対時刻を持つセグメント。

    words は担当チャンクの判定に使う(空なら、セグメントの開始時刻で判定する)。
    """

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


class Transcriber(Protocol):
    """float32 / 16kHz / モノラルの音声を文字起こしする。"""

    def transcribe(self, samples: np.ndarray) -> list[Segment]: ...


class FakeTranscriber:
    """テスト用の偽実装。呼び出し順に用意したセグメント列を返す。"""

    def __init__(
        self,
        responses: list[list[Segment]] | None = None,
        *,
        error_on_call: int | None = None,
    ) -> None:
        self._responses = [list(items) for items in (responses or [])]
        self._error_on_call = error_on_call
        self.calls: list[np.ndarray] = []

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        self.calls.append(np.asarray(samples, dtype=np.float32).copy())
        call_index = len(self.calls) - 1
        if self._error_on_call is not None and call_index == self._error_on_call:
            raise RuntimeError("fake transcription failure")
        if call_index < len(self._responses):
            return list(self._responses[call_index])
        return []


def as_mono_float32(samples: np.ndarray) -> np.ndarray:
    """推論に渡す前に、1次元(モノラル)であることを確かめて float32 にする。"""
    if samples.ndim != 1:
        raise ValueError("samples は1次元(モノラル)であること")
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32, copy=False)
    return samples


def to_segments(raw_segments: Iterable[Any]) -> list[Segment]:
    """faster-whisper のセグメント列を Segment に変換する。本文が空のセグメントは除く。"""
    result: list[Segment] = []
    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue
        words = tuple(
            Word(start=float(w.start), end=float(w.end), text=w.word) for w in seg.words or ()
        )
        result.append(Segment(start=float(seg.start), end=float(seg.end), text=text, words=words))
    return result
