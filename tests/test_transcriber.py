"""transcriber の共通処理の単体テスト(実モデルなし)。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from loopscribe.transcriber import Segment, Word, as_mono_float32, to_segments


def test_segment_fields() -> None:
    seg = Segment(start=1.0, end=2.5, text="こんにちは")
    assert seg.start == 1.0
    assert seg.end == 2.5
    assert seg.text == "こんにちは"
    assert seg.words == ()


def test_to_segments_maps_and_skips_empty() -> None:
    raw = [
        SimpleNamespace(
            start=0.0,
            end=1.0,
            text="  こんにちは  ",
            words=[
                SimpleNamespace(start=0.0, end=0.4, word=" こん", probability=0.9),
                SimpleNamespace(start=0.4, end=1.0, word="にちは", probability=0.9),
            ],
        ),
        SimpleNamespace(start=1.5, end=2.0, text="   ", words=[]),
        SimpleNamespace(start=2.0, end=3.0, text="世界", words=None),
    ]
    assert to_segments(iter(raw)) == [
        Segment(
            start=0.0,
            end=1.0,
            text="こんにちは",
            words=(Word(start=0.0, end=0.4, text=" こん"), Word(start=0.4, end=1.0, text="にちは")),
        ),
        Segment(start=2.0, end=3.0, text="世界"),
    ]


def test_as_mono_float32_rejects_non_mono() -> None:
    with pytest.raises(ValueError, match="1次元"):
        as_mono_float32(np.zeros((2, 100), dtype=np.float32))


def test_as_mono_float32_casts_dtype() -> None:
    assert as_mono_float32(np.zeros(100, dtype=np.float64)).dtype == np.float32
