"""Whisper `small`: 30秒チャンクを WhisperModel.transcribe でそのまま推論する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from loopscribe.constants import CPU_THREADS, LANGUAGE
from loopscribe.models.base import ModelSpec
from loopscribe.transcriber import Segment, as_mono_float32, to_segments

REPO_ID = "small"
COMPUTE_TYPE = "int8"
# 5にしてもCERの改善が小さく、RTFが上がる(決定ログ#17)
BEAM_SIZE = 1
# オフにすると取りこぼしが増える(決定ログ#17)
VAD_FILTER = True
# チャンクをまたいだ誤りの伝播を防ぐ
CONDITION_ON_PREVIOUS_TEXT = False


class SmallTranscriber:
    """`small` の Transcriber 実装。"""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        cpu_threads: int = CPU_THREADS,
        compute_type: str = COMPUTE_TYPE,
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
            str(model_name_or_path),
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        raw_segments, _info = self._model.transcribe(
            as_mono_float32(samples),
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            condition_on_previous_text=self._condition_on_previous_text,
            word_timestamps=True,
        )
        return to_segments(raw_segments)


class SmallModel(ModelSpec):
    name = "small"
    repo_id = REPO_ID

    def load(self, model_dir: Path, *, cpu_threads: int = CPU_THREADS) -> SmallTranscriber:
        return SmallTranscriber(model_dir, cpu_threads=cpu_threads)
