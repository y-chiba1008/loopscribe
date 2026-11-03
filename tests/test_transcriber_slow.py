"""実モデルを使う slow テスト。既定ではスキップ(`uv run pytest -m slow`)。"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from rich.console import Console

from loopscribe.audio.wavfile import WavFileSource
from loopscribe.constants import (
    BEAM_SIZE,
    COMPUTE_TYPE,
    CONDITION_ON_PREVIOUS_TEXT,
    CPU_THREADS,
    LANGUAGE,
    MODEL_NAME,
    VAD_FILTER,
)
from loopscribe.model import ensure_model, is_model_available
from loopscribe.output import AbsoluteSegment
from loopscribe.pipeline import PipelineProgress, TranscriptionPipeline
from loopscribe.transcriber import FasterWhisperTranscriber
from tests.cer import character_error_rate

# 検証時の small CER は約 0.19。環境差を見て余裕を持たせる。
_MAX_CER = 0.40

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_CANDIDATES = (
    (
        _REPO_ROOT / "tests" / "data" / "sample.wav",
        _REPO_ROOT / "tests" / "data" / "sample.txt",
    ),
    (
        _REPO_ROOT / "scripts" / "data" / "sample.wav",
        _REPO_ROOT / "scripts" / "data" / "sample.txt",
    ),
)


def _resolve_sample() -> tuple[Path, Path]:
    for wav, txt in _SAMPLE_CANDIDATES:
        if wav.is_file() and txt.is_file():
            return wav, txt
    pytest.skip(
        "slow テスト用の sample.wav / sample.txt がありません。"
        " tests/data/ に置くか、scripts/prepare_bench_audio.py を実行してください。"
    )


@pytest.mark.slow
def test_pipeline_with_faster_whisper_cer_on_sample() -> None:
    """WavFileSource → pipeline → 実モデルの CER が閾値以下であること。"""
    if not is_model_available(MODEL_NAME):
        pytest.skip(f"モデル '{MODEL_NAME}' が未取得です。download-model 相当で取得してください。")

    wav_path, txt_path = _resolve_sample()
    reference = txt_path.read_text(encoding="utf-8")
    model_path = ensure_model(MODEL_NAME)
    transcriber = FasterWhisperTranscriber(
        str(model_path),
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=VAD_FILTER,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
    )

    texts: list[str] = []

    class CollectSink:
        def write(self, segment: AbsoluteSegment) -> None:
            texts.append(segment.text)

        def close(self) -> None:
            return

    console = Console(force_terminal=False)
    stop = threading.Event()
    source = WavFileSource(wav_path)
    progress = PipelineProgress()
    pipeline = TranscriptionPipeline(
        source,
        transcriber,
        CollectSink(),
        stop_event=stop,
        console=console,
        progress=progress,
        wait_timeout=0.2,
    )
    code = pipeline.run()
    assert code == 0

    hypothesis = "".join(texts)
    cer = character_error_rate(reference, hypothesis)
    assert cer <= _MAX_CER, (
        f"CER {cer:.3f} が閾値 {_MAX_CER} を超えました"
        f" (segments={len(texts)}, hyp_head={hypothesis[:80]!r})"
    )
