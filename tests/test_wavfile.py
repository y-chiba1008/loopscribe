"""WavFileSource の単体テスト。"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from loopscribe.audio.wavfile import WavFileSource


def _write_pcm16_wav(
    path: Path,
    samples: np.ndarray,
    *,
    sample_rate: int,
    channels: int,
) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.reshape(-1).tobytes())


def test_wavfile_source_reads_blocks(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    rate = 16_000
    n = rate * 2
    mono = (0.25 * np.sin(2 * np.pi * 440 * np.arange(n) / rate)).astype(np.float32)
    _write_pcm16_wav(path, mono, sample_rate=rate, channels=1)

    source = WavFileSource(path, block_frames=1000)
    assert source.sample_rate == rate
    assert source.channels == 1
    blocks = list(source)
    source.close()
    assert len(blocks) >= 2
    joined = np.concatenate([b[:, 0] for b in blocks])
    assert joined.shape[0] == n
    # int16 量子化誤差(書き込み *32767 / 読み込み /32768)
    assert np.max(np.abs(joined - mono)) < 2 / 32768


def test_wavfile_source_close_during_iteration_stops_cleanly(tmp_path: Path) -> None:
    """反復中に close しても例外にならず、以降のブロックは返さない。"""
    path = tmp_path / "long.wav"
    rate = 16_000
    _write_pcm16_wav(path, np.zeros(rate * 2, dtype=np.float32), sample_rate=rate, channels=1)

    source = WavFileSource(path, block_frames=1000)
    it = iter(source)
    next(it)
    source.close()
    assert list(it) == []


def test_wavfile_source_duration_from_header(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    rate = 48_000
    stereo = np.zeros(rate * 3 * 2, dtype=np.float32)
    _write_pcm16_wav(path, stereo, sample_rate=rate, channels=2)

    source = WavFileSource(path)
    assert source.duration_seconds == pytest.approx(3.0)
    source.close()


def test_wavfile_source_rejects_non_pcm16(tmp_path: Path) -> None:
    path = tmp_path / "bad.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(b"\x00" * 800)

    with pytest.raises(ValueError, match="16bit"):
        WavFileSource(path)
