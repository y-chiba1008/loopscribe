"""audio.convert の単体テスト。"""

from __future__ import annotations

import numpy as np

from loopscribe.audio.convert import AudioConverter


def test_converter_stereo_48k_to_mono_16k_length() -> None:
    in_rate = 48_000
    seconds = 1.0
    n = int(in_rate * seconds)
    t = np.arange(n, dtype=np.float32) / in_rate
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([left, right], axis=1)

    conv = AudioConverter(in_rate, 2, out_rate=16_000)
    out = conv.convert(stereo, last=True)
    assert out.dtype == np.int16
    assert out.ndim == 1
    # ストリーム終端の遅延で多少前後するが、おおよそ 1 秒分
    assert 15_000 < out.size < 17_000


def test_converter_blockwise_matches_oneshot_roughly() -> None:
    in_rate = 32_000
    out_rate = 16_000
    n = in_rate * 2
    mono = (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / in_rate)).astype(np.float32)

    oneshot = AudioConverter(in_rate, 1, out_rate=out_rate)
    a = oneshot.convert(mono, last=True)

    streamed = AudioConverter(in_rate, 1, out_rate=out_rate)
    parts: list[np.ndarray] = []
    block = 1024
    for i in range(0, n, block):
        last = i + block >= n
        chunk = mono[i : i + block]
        parts.append(streamed.convert(chunk, last=last))
    b = np.concatenate(parts) if parts else np.empty(0, dtype=np.int16)

    assert a.size == b.size
    # ストリーム分割でも一致する(境界で不連続にならない)
    np.testing.assert_array_equal(a, b)
