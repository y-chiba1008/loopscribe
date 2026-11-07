"""チャンク区間と担当境界の計算。

設定値は引数で受け取り、既定値として定数を使う(グローバル参照はしない)。
"""

from __future__ import annotations

from dataclasses import dataclass

from loopscribe.constants import CHUNK_SECONDS, MIN_TAIL_SECONDS, OVERLAP_SECONDS


@dataclass(frozen=True)
class ChunkSpan:
    """1つのチャンクがカバーする録音上の区間(秒)。"""

    index: int
    start: float
    end: float


@dataclass(frozen=True)
class OwnershipBounds:
    """チャンクが採用する語(語が無いセグメントはセグメント)の絶対開始時刻の範囲 [lo, hi)。"""

    lo: float
    hi: float

    def contains(self, absolute_start: float) -> bool:
        return self.lo <= absolute_start < self.hi


def step_seconds(chunk_seconds: float, overlap_seconds: float) -> float:
    """チャンク開始同士の間隔(秒)。"""
    if chunk_seconds <= overlap_seconds:
        raise ValueError("chunk_seconds は overlap_seconds より大きくすること")
    return chunk_seconds - overlap_seconds


def chunk_span(
    index: int,
    *,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
    end: float | None = None,
) -> ChunkSpan:
    """チャンク index の区間を返す。

    end を渡すと終端をそこで打ち切る(停止時の端数チャンク用)。
    """
    if index < 0:
        raise ValueError("index は 0 以上であること")
    start = index * step_seconds(chunk_seconds, overlap_seconds)
    full_end = start + chunk_seconds
    actual_end = full_end if end is None else end
    if actual_end < start:
        raise ValueError("end は start 以上であること")
    return ChunkSpan(index=index, start=start, end=actual_end)


def chunk_ready_at(
    index: int,
    *,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> float:
    """チャンク index の処理を始めてよい録音秒数(終端に達した時点)。"""
    return chunk_span(index, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds).end


def is_chunk_ready(
    index: int,
    recorded_seconds: float,
    *,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> bool:
    """録音がチャンク index の終端に達していれば True。"""
    return recorded_seconds >= chunk_ready_at(
        index, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds
    )


def ownership_bounds(
    index: int,
    *,
    is_last: bool,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> OwnershipBounds:
    """チャンク index が採用する絶対開始時刻の範囲。

    重なり区間の真ん中を境界にする。先頭チャンクの下限は 0、
    最終チャンクの上限は無限大。
    """
    step = step_seconds(chunk_seconds, overlap_seconds)
    start = index * step
    half = overlap_seconds / 2.0
    lo = 0.0 if index == 0 else start + half
    hi = float("inf") if is_last else (index + 1) * step + half
    return OwnershipBounds(lo=lo, hi=hi)


def owns_segment(
    absolute_start: float,
    index: int,
    *,
    is_last: bool,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> bool:
    """絶対開始時刻がチャンク index の担当なら True。"""
    bounds = ownership_bounds(
        index,
        is_last=is_last,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    return bounds.contains(absolute_start)


def tail_span_after_stop(
    recorded_seconds: float,
    next_index: int,
    *,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
    min_tail_seconds: float = MIN_TAIL_SECONDS,
) -> ChunkSpan | None:
    """録音停止後に処理する端数チャンクを返す。

    まだカバーされていない末尾が min_tail_seconds 以上あれば、
    [next_index の開始, recorded_seconds] を返す。それ未満なら None。
    None のとき、直前のチャンクを最終チャンク(is_last=True)として扱うのは呼び出し側の責務。
    """
    if next_index < 0:
        raise ValueError("next_index は 0 以上であること")
    start = next_index * step_seconds(chunk_seconds, overlap_seconds)
    remaining = recorded_seconds - start
    if remaining < min_tail_seconds:
        return None
    return ChunkSpan(index=next_index, start=start, end=recorded_seconds)
