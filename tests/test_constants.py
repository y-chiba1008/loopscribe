"""constants の単体テスト(設計資料の確定値)。"""

from __future__ import annotations

import importlib

import pytest

from loopscribe import constants


def test_language_constant() -> None:
    assert constants.LANGUAGE == "ja"


def test_model_constants_are_not_in_constants() -> None:
    for name in ("MODEL_NAME", "COMPUTE_TYPE", "BEAM_SIZE", "VAD_FILTER"):
        assert not hasattr(constants, name)


@pytest.mark.parametrize(
    ("cpu_count", "expected"),
    [(8, 4), (12, 6), (1, 1), (None, 1)],
)
def test_cpu_threads_is_half_of_logical_cores(
    monkeypatch: pytest.MonkeyPatch, cpu_count: int | None, expected: int
) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: cpu_count)
    try:
        threads = importlib.reload(constants).CPU_THREADS
        assert threads == expected
    finally:
        monkeypatch.undo()
        importlib.reload(constants)


def test_audio_and_chunk_constants() -> None:
    assert constants.SAMPLE_RATE == 16_000
    assert constants.CHUNK_SECONDS == 30.0
    assert constants.OVERLAP_SECONDS == 3.0
    assert constants.MIN_TAIL_SECONDS == 1.0
    assert constants.WASAPI_FRAMES_PER_BUFFER == 512
    assert constants.WASAPI_QUEUE_POLL_SECONDS == 0.2


def test_ui_and_output_constants() -> None:
    assert constants.LAG_WARN_SECONDS == 600.0
    assert constants.BUFFER_WARN_MB == 2048.0
    assert constants.UI_REFRESH_SECONDS == 0.5
    assert constants.SIGNAL_POLL_SECONDS == 0.2
    assert constants.OUTPUT_FILENAME_FORMAT == "loopscribe_%Y%m%d_%H%M%S.txt"
    assert constants.OUTPUT_ENCODING == "utf-8"
