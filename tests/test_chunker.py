"""chunker の単体テスト。"""

from __future__ import annotations

import math

import pytest

from loopscribe.chunker import (
    ChunkSpan,
    chunk_ready_at,
    chunk_span,
    is_chunk_ready,
    ownership_bounds,
    owns_segment,
    step_seconds,
    tail_span_after_stop,
)


def test_step_seconds() -> None:
    assert step_seconds(30.0, 3.0) == 27.0


def test_step_seconds_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError):
        step_seconds(3.0, 3.0)


def test_chunk_span_full() -> None:
    assert chunk_span(0, chunk_seconds=30.0, overlap_seconds=3.0) == ChunkSpan(0, 0.0, 30.0)
    span = chunk_span(1, chunk_seconds=30.0, overlap_seconds=3.0)
    assert span.start == 27.0
    assert span.end == 57.0


def test_chunk_span_with_clipped_end() -> None:
    span = chunk_span(1, chunk_seconds=30.0, overlap_seconds=3.0, end=40.0)
    assert span.start == 27.0
    assert span.end == 40.0


def test_chunk_ready() -> None:
    assert chunk_ready_at(0, chunk_seconds=30.0, overlap_seconds=3.0) == 30.0
    assert chunk_ready_at(1, chunk_seconds=30.0, overlap_seconds=3.0) == 57.0
    assert is_chunk_ready(0, 29.9, chunk_seconds=30.0, overlap_seconds=3.0) is False
    assert is_chunk_ready(0, 30.0, chunk_seconds=30.0, overlap_seconds=3.0) is True


def test_ownership_middle_of_overlap() -> None:
    # チャンク0: [0, 28.5), チャンク1(非最終): [28.5, 55.5)
    b0 = ownership_bounds(0, is_last=False, chunk_seconds=30.0, overlap_seconds=3.0)
    assert b0.lo == 0.0
    assert b0.hi == 28.5
    b1 = ownership_bounds(1, is_last=False, chunk_seconds=30.0, overlap_seconds=3.0)
    assert b1.lo == 28.5
    assert b1.hi == 55.5


def test_ownership_last_has_infinite_hi() -> None:
    b = ownership_bounds(1, is_last=True, chunk_seconds=30.0, overlap_seconds=3.0)
    assert b.lo == 28.5
    assert math.isinf(b.hi)


def test_owns_segment_no_overlap_or_gap() -> None:
    # 境界 28.5 は後続チャンクが担当(半開区間)
    assert owns_segment(28.499, 0, is_last=False, chunk_seconds=30.0, overlap_seconds=3.0)
    assert not owns_segment(28.5, 0, is_last=False, chunk_seconds=30.0, overlap_seconds=3.0)
    assert owns_segment(28.5, 1, is_last=False, chunk_seconds=30.0, overlap_seconds=3.0)


def test_tail_span_when_long_enough() -> None:
    # チャンク0処理後、録音40秒 → 端数 [27, 40]
    tail = tail_span_after_stop(
        40.0,
        1,
        chunk_seconds=30.0,
        overlap_seconds=3.0,
        min_tail_seconds=1.0,
    )
    assert tail is not None
    assert tail.index == 1
    assert tail.start == 27.0
    assert tail.end == 40.0


def test_tail_span_skipped_when_shorter_than_min() -> None:
    # チャンク0処理後、録音 27.5 秒 → 残り 0.5 < 1.0
    assert (
        tail_span_after_stop(
            27.5,
            1,
            chunk_seconds=30.0,
            overlap_seconds=3.0,
            min_tail_seconds=1.0,
        )
        is None
    )


def test_short_chunk_params_for_tests() -> None:
    # 数秒の wav 向けに定数を小さく渡せる
    span = chunk_span(0, chunk_seconds=4.0, overlap_seconds=1.0)
    assert span.end == 4.0
    assert is_chunk_ready(1, 7.0, chunk_seconds=4.0, overlap_seconds=1.0)
    assert not is_chunk_ready(1, 6.9, chunk_seconds=4.0, overlap_seconds=1.0)
