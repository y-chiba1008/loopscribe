"""constants の単体テスト(設計資料の確定値)。"""

from __future__ import annotations

from loopscribe import constants


def test_model_and_inference_constants() -> None:
    assert constants.MODEL_NAME == "small"
    assert constants.COMPUTE_TYPE == "int8"
    assert constants.CPU_THREADS == 4
    assert constants.LANGUAGE == "ja"
    assert constants.BEAM_SIZE == 1
    assert constants.VAD_FILTER is True
    assert constants.CONDITION_ON_PREVIOUS_TEXT is False


def test_audio_and_chunk_constants() -> None:
    assert constants.SAMPLE_RATE == 16_000
    assert constants.CHUNK_SECONDS == 30.0
    assert constants.OVERLAP_SECONDS == 3.0
    assert constants.MIN_TAIL_SECONDS == 1.0


def test_ui_and_output_constants() -> None:
    assert constants.LAG_WARN_SECONDS == 600.0
    assert constants.BUFFER_WARN_MB == 2048.0
    assert constants.UI_REFRESH_SECONDS == 0.5
    assert constants.SIGNAL_POLL_SECONDS == 0.2
    assert constants.OUTPUT_FILENAME_FORMAT == "loopscribe_%Y%m%d_%H%M%S.txt"
    assert constants.OUTPUT_ENCODING == "utf-8"
