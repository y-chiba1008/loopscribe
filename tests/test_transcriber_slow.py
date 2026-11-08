"""実モデルを使う slow テスト。既定ではスキップ(`uv run pytest -m slow`)。

モデルは自動では取得しない(kotoba-whisper は約1.5GBある)。未取得のモデルはスキップする。
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from rich.console import Console

from loopscribe.audio.wavfile import WavFileSource
from loopscribe.models import MODELS
from loopscribe.output import AbsoluteSegment
from loopscribe.pipeline import PipelineProgress, TranscriptionPipeline
from tests.cer import character_error_rate

# 検証時の CER は small で約 0.19、kotoba で約 0.18。環境差を見て余裕を持たせる。
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
@pytest.mark.parametrize("model_name", list(MODELS))
def test_pipeline_with_faster_whisper_cer_on_sample(model_name: str) -> None:
    """WavFileSource → pipeline → 実モデルの CER が閾値以下であること。"""
    spec = MODELS[model_name]
    if not spec.is_available():
        pytest.skip(
            f"モデル '{model_name}' が未取得です。"
            f"`loopscribe download-model --model {model_name}` で取得してください。"
        )

    wav_path, txt_path = _resolve_sample()
    reference = txt_path.read_text(encoding="utf-8")
    transcriber = spec.load(spec.ensure())

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
