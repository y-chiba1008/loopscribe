"""Transcriber の単体テスト(実モデルなし)。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from loopscribe.transcriber import FasterWhisperTranscriber, Segment


def test_segment_fields() -> None:
    seg = Segment(start=1.0, end=2.5, text="こんにちは")
    assert seg.start == 1.0
    assert seg.end == 2.5
    assert seg.text == "こんにちは"


def test_faster_whisper_transcriber_maps_segments() -> None:
    raw = [
        SimpleNamespace(start=0.0, end=1.0, text="  こんにちは  "),
        SimpleNamespace(start=1.5, end=2.0, text="   "),
        SimpleNamespace(start=2.0, end=3.0, text="世界"),
    ]
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter(raw), SimpleNamespace())

    with patch("loopscribe.transcriber.WhisperModel", return_value=mock_model):
        tr = FasterWhisperTranscriber(
            "small",
            language="ja",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

    samples = np.zeros(16_000, dtype=np.float32)
    segments = tr.transcribe(samples)

    assert segments == [
        Segment(start=0.0, end=1.0, text="こんにちは"),
        Segment(start=2.0, end=3.0, text="世界"),
    ]
    kwargs = mock_model.transcribe.call_args.kwargs
    assert kwargs["language"] == "ja"
    assert kwargs["beam_size"] == 1
    assert kwargs["vad_filter"] is True
    assert kwargs["condition_on_previous_text"] is False


def test_faster_whisper_transcriber_rejects_non_mono() -> None:
    mock_model = MagicMock()
    with patch("loopscribe.transcriber.WhisperModel", return_value=mock_model):
        tr = FasterWhisperTranscriber("small")
    with pytest.raises(ValueError, match="1次元"):
        tr.transcribe(np.zeros((2, 100), dtype=np.float32))


def test_faster_whisper_transcriber_casts_dtype() -> None:
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter(()), SimpleNamespace())
    with patch("loopscribe.transcriber.WhisperModel", return_value=mock_model):
        tr = FasterWhisperTranscriber("small")

    samples = np.zeros(100, dtype=np.float64)
    assert tr.transcribe(samples) == []
    passed = mock_model.transcribe.call_args.args[0]
    assert passed.dtype == np.float32
