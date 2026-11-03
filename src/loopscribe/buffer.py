"""スレッドセーフな音声バッファ(int16 / 16kHz / モノラル)。"""

from __future__ import annotations

import threading

import numpy as np
from numpy.typing import NDArray


class AudioBuffer:
    """録音サンプルを無制限に保持するバッファ。

    サンプル位置は録音開始からの絶対インデックス。処理済み区間は
    `trim_before` で解放する(次チャンクの重なり分は残す)。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._data = np.empty(0, dtype=np.int16)
        self._origin = 0
        self._total = 0
        self._closed = False

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def append(self, samples: NDArray[np.int16]) -> None:
        """末尾にサンプルを追記する。close 後は ValueError。"""
        if samples.dtype != np.int16:
            raise TypeError("samples は dtype=int16 であること")
        if samples.ndim != 1:
            raise ValueError("samples は1次元(モノラル)であること")
        with self._cond:
            if self._closed:
                raise ValueError("close 済みの AudioBuffer には追記できない")
            if samples.size == 0:
                return
            self._data = np.concatenate((self._data, samples))
            self._total += int(samples.size)
            self._cond.notify_all()

    def read(self, start: int, end: int) -> NDArray[np.int16]:
        """絶対サンプル位置 [start, end) をコピーして返す。"""
        if start < 0 or end < start:
            raise ValueError("start/end が不正")
        with self._lock:
            if start < self._origin:
                raise ValueError(f"start={start} は trim 済み(origin={self._origin})より前")
            if end > self._total:
                raise ValueError(f"end={end} は total_samples={self._total} を超える")
            relative_start = start - self._origin
            relative_end = end - self._origin
            return self._data[relative_start:relative_end].copy()

    def trim_before(self, sample: int) -> None:
        """sample 未満のデータを破棄する。sample が origin 以下なら何もしない。"""
        if sample < 0:
            raise ValueError("sample は 0 以上であること")
        with self._lock:
            if sample <= self._origin:
                return
            if sample > self._total:
                raise ValueError(f"sample={sample} は total_samples={self._total} を超える")
            relative = sample - self._origin
            self._data = self._data[relative:].copy()
            self._origin = sample

    def wait_for(self, n_samples: int, timeout: float | None = None) -> bool:
        """total_samples が n_samples 以上になるまで待つ。

        到達したら True。close されても到達していなければ False。
        timeout 秒以内にどちらでもなければ False。
        """
        if n_samples < 0:
            raise ValueError("n_samples は 0 以上であること")
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._total >= n_samples or self._closed,
                timeout=timeout,
            )
            if not ok:
                return False
            return self._total >= n_samples

    def close(self) -> None:
        """これ以上の追記を止め、待機中の wait_for を起こす。"""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
