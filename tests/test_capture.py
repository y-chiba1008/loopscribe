"""capture の単体テスト。"""

from __future__ import annotations

import threading

import numpy as np
from numpy.typing import NDArray

from loopscribe.capture import AudioCapture


class _ListSource:
    def __init__(self, blocks: list[np.ndarray], *, sample_rate: int = 16_000, fail: bool = False):
        self.sample_rate = sample_rate
        self.channels = blocks[0].shape[1] if blocks else 1
        self._blocks = blocks
        self._fail = fail
        self.closed = 0

    def __iter__(self):
        yield from self._blocks
        if self._fail:
            raise OSError("device lost")

    def close(self) -> None:
        self.closed += 1


def _run(capture: AudioCapture) -> None:
    capture.start()
    capture.join(timeout=5.0)
    assert not capture.is_alive()


def test_capture_converts_to_mono_16k_and_finishes() -> None:
    received: list[NDArray[np.int16]] = []
    finished: list[bool] = []
    source = _ListSource(
        [np.zeros((4_800, 2), dtype=np.float32) for _ in range(10)], sample_rate=48_000
    )
    capture = AudioCapture(
        source,
        received.append,
        stop_event=threading.Event(),
        on_warning=lambda _m: None,
        on_finish=lambda: finished.append(True),
    )
    _run(capture)
    total = sum(s.size for s in received)
    assert abs(total - 16_000) <= 10
    assert all(s.dtype == np.int16 and s.ndim == 1 for s in received)
    assert finished == [True]
    assert source.closed >= 1
    assert capture.failed is False


def test_capture_failure_warns_and_sets_stop() -> None:
    warnings: list[str] = []
    stop = threading.Event()
    finished: list[bool] = []
    source = _ListSource([np.zeros((1_600, 1), dtype=np.float32)], fail=True)
    capture = AudioCapture(
        source,
        lambda _s: None,
        stop_event=stop,
        on_warning=warnings.append,
        on_finish=lambda: finished.append(True),
    )
    _run(capture)
    assert capture.failed is True
    assert stop.is_set()
    assert any("device lost" in w for w in warnings)
    assert finished == [True]


def test_close_source_if_stopped_closes_once() -> None:
    stop = threading.Event()
    source = _ListSource([])
    capture = AudioCapture(source, lambda _s: None, stop_event=stop, on_warning=lambda _m: None)
    capture.close_source_if_stopped()
    assert source.closed == 0
    stop.set()
    capture.close_source_if_stopped()
    capture.close_source_if_stopped()
    assert source.closed == 1
