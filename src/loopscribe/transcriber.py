"""文字起こしのプロトコルと faster-whisper 実装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from faster_whisper import WhisperModel

from loopscribe.constants import (
    BEAM_SIZE,
    COMPUTE_TYPE,
    CONDITION_ON_PREVIOUS_TEXT,
    CPU_THREADS,
    LANGUAGE,
    MODEL_NAME,
    VAD_FILTER,
)


@dataclass(frozen=True)
class Segment:
    """チャンク内の相対時刻を持つセグメント。"""

    start: float
    end: float
    text: str


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


class FasterWhisperTranscriber:
    """faster-whisper による Transcriber 実装。"""

    def __init__(
        self,
        model_name_or_path: str = MODEL_NAME,
        *,
        device: str = "cpu",
        compute_type: str = COMPUTE_TYPE,
        cpu_threads: int = CPU_THREADS,
        language: str = LANGUAGE,
        beam_size: int = BEAM_SIZE,
        vad_filter: bool = VAD_FILTER,
        condition_on_previous_text: bool = CONDITION_ON_PREVIOUS_TEXT,
    ) -> None:
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._condition_on_previous_text = condition_on_previous_text
        self._model = WhisperModel(
            model_name_or_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        if samples.ndim != 1:
            raise ValueError("samples は1次元(モノラル)であること")
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)

        raw_segments, _info = self._model.transcribe(
            samples,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            condition_on_previous_text=self._condition_on_previous_text,
        )
        result: list[Segment] = []
        for seg in raw_segments:
            text = seg.text.strip()
            if not text:
                continue
            result.append(Segment(start=float(seg.start), end=float(seg.end), text=text))
        return result
