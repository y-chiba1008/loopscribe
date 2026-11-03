"""AudioBuffer の単体テスト。"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from loopscribe.buffer import AudioBuffer


def _i16(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int16)


def test_append_and_read() -> None:
    buf = AudioBuffer()
    buf.append(_i16(1, 2, 3, 4))
    assert buf.total_samples == 4
    np.testing.assert_array_equal(buf.read(1, 3), _i16(2, 3))


def test_append_rejects_wrong_dtype_and_shape() -> None:
    buf = AudioBuffer()
    with pytest.raises(TypeError):
        buf.append(np.array([1.0], dtype=np.float32))
    with pytest.raises(ValueError):
        buf.append(np.array([[1, 2]], dtype=np.int16))


def test_read_rejects_trimmed_or_future() -> None:
    buf = AudioBuffer()
    buf.append(_i16(10, 20, 30, 40))
    buf.trim_before(2)
    with pytest.raises(ValueError):
        buf.read(0, 2)
    with pytest.raises(ValueError):
        buf.read(2, 5)


def test_trim_before_keeps_suffix() -> None:
    buf = AudioBuffer()
    buf.append(_i16(1, 2, 3, 4, 5))
    buf.trim_before(2)
    np.testing.assert_array_equal(buf.read(2, 5), _i16(3, 4, 5))
    assert buf.total_samples == 5


def test_wait_for_returns_true_when_enough() -> None:
    buf = AudioBuffer()
    buf.append(_i16(1, 2, 3))
    assert buf.wait_for(3, timeout=0.1) is True


def test_wait_for_false_on_timeout() -> None:
    buf = AudioBuffer()
    assert buf.wait_for(10, timeout=0.05) is False


def test_wait_for_woken_by_append() -> None:
    buf = AudioBuffer()
    result: list[bool] = []

    def waiter() -> None:
        result.append(buf.wait_for(4, timeout=1.0))

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    buf.append(_i16(1, 2, 3, 4))
    t.join(timeout=1.0)
    assert result == [True]


def test_wait_for_false_when_closed_early() -> None:
    buf = AudioBuffer()
    result: list[bool] = []

    def waiter() -> None:
        result.append(buf.wait_for(100, timeout=1.0))

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    buf.close()
    t.join(timeout=1.0)
    assert result == [False]
    assert buf.closed is True


def test_append_after_close_raises() -> None:
    buf = AudioBuffer()
    buf.close()
    with pytest.raises(ValueError):
        buf.append(_i16(1))
