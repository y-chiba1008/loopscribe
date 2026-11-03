"""モノラル化と 16kHz へのリサンプル。"""

from __future__ import annotations

import numpy as np
import soxr
from numpy.typing import NDArray

from loopscribe.constants import SAMPLE_RATE


class AudioConverter:
    """マルチチャンネル float32 を、モノラル int16 / 16kHz に変換する。

    ブロック境界の不連続を避けるため、状態付きの soxr.ResampleStream を使う。
    """

    def __init__(
        self,
        in_rate: int,
        in_channels: int,
        *,
        out_rate: int = SAMPLE_RATE,
    ) -> None:
        if in_rate <= 0:
            raise ValueError("in_rate は 1 以上であること")
        if in_channels <= 0:
            raise ValueError("in_channels は 1 以上であること")
        if out_rate <= 0:
            raise ValueError("out_rate は 1 以上であること")
        self._in_channels = in_channels
        self._out_rate = out_rate
        self._stream = soxr.ResampleStream(
            in_rate,
            out_rate,
            num_channels=1,
            dtype="float32",
        )

    @property
    def out_rate(self) -> int:
        return self._out_rate

    def convert(
        self,
        block: NDArray[np.floating],
        *,
        last: bool = False,
    ) -> NDArray[np.int16]:
        """1ブロックを変換する。入力は (frames,) または (frames, channels)。"""
        mono = _to_mono_float32(block, self._in_channels)
        resampled = self._stream.resample_chunk(mono, last=last)
        if resampled.size == 0:
            return np.empty(0, dtype=np.int16)
        return _float_to_int16(resampled)


def _to_mono_float32(
    block: NDArray[np.floating],
    expected_channels: int,
) -> NDArray[np.float32]:
    if block.size == 0:
        return np.empty(0, dtype=np.float32)
    if block.ndim == 1:
        if expected_channels != 1:
            raise ValueError(f"1次元入力は channels=1 のときのみ(expected={expected_channels})")
        return np.ascontiguousarray(block, dtype=np.float32)
    if block.ndim != 2:
        raise ValueError("block は1次元または2次元であること")
    if block.shape[1] != expected_channels:
        raise ValueError(
            f"channels が一致しない: got={block.shape[1]}, expected={expected_channels}"
        )
    if expected_channels == 1:
        return np.ascontiguousarray(block[:, 0], dtype=np.float32)
    return np.ascontiguousarray(block.mean(axis=1), dtype=np.float32)


def _float_to_int16(samples: NDArray[np.floating]) -> NDArray[np.int16]:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)
