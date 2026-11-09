"""WAV ファイル入力(`transcribe` とテストで使う)。"""

from __future__ import annotations

import threading
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class WavFileSource:
    """PCM WAV をブロック単位で読む AudioSource。

    16bit PCM を想定する。stdlib の wave のみを使い、追加依存を増やさない。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        block_frames: int = 1024,
    ) -> None:
        if block_frames <= 0:
            raise ValueError("block_frames は 1 以上であること")
        self._path = Path(path)
        self._block_frames = block_frames
        # インスタンス寿命まで開いたままにする(イテレータと close で管理)
        self._wf = wave.open(str(self._path), "rb")  # noqa: SIM115
        try:
            if self._wf.getsampwidth() != 2:
                raise ValueError(f"16bit PCM WAV のみ対応(sampwidth={self._wf.getsampwidth()})")
            if self._wf.getcomptype() != "NONE":
                raise ValueError(f"非圧縮 PCM のみ対応(comptype={self._wf.getcomptype()})")
            self._sample_rate = int(self._wf.getframerate())
            self._channels = int(self._wf.getnchannels())
            self._nframes = int(self._wf.getnframes())
            if self._sample_rate <= 0 or self._channels <= 0:
                raise ValueError("不正な WAV ヘッダです")
        except Exception:
            self._wf.close()
            raise
        self._closed = False
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def duration_seconds(self) -> float:
        """ヘッダから求めた音声長(秒)。"""
        return self._nframes / self._sample_rate

    def __iter__(self) -> Iterator[NDArray[np.float32]]:
        if self._closed:
            raise ValueError("close 済みの WavFileSource は読めない")
        while True:
            with self._lock:
                if self._closed:
                    break
                raw = self._wf.readframes(self._block_frames)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
            samples /= 32768.0
            frames = samples.size // self._channels
            if frames * self._channels != samples.size:
                raise ValueError("WAV のフレーム境界が不正です")
            yield samples.reshape(frames, self._channels)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._wf.close()
