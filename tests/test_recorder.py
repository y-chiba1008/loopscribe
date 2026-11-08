"""recorder の単体テスト。"""

from __future__ import annotations

import threading
import time
import wave
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from loopscribe.audio.wavfile import WavFileSource
from loopscribe.recorder import WAV_HEADER_BYTES, Recorder, WavWriter


def _console() -> Console:
    return Console(file=StringIO(), force_terminal=False, width=120, highlight=False)


def _read_wav(path: Path) -> tuple[wave._wave_params, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        params = wf.getparams()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    return params, pcm


def _write_wav(path: Path, *, seconds: float, sample_rate: int, channels: int) -> None:
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    pcm = np.repeat(tone[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def test_wav_writer_writes_mono_16bit(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    writer = WavWriter(path, sample_rate=16_000, flush_seconds=1.0)
    samples = np.arange(-500, 500, dtype=np.int16)
    writer.write(samples)
    writer.write(np.empty(0, dtype=np.int16))
    writer.close()

    params, pcm = _read_wav(path)
    assert (params.nchannels, params.sampwidth, params.framerate) == (1, 2, 16_000)
    np.testing.assert_array_equal(pcm, samples)
    assert path.stat().st_size == WAV_HEADER_BYTES + samples.size * 2


def test_wav_writer_flushes_readable_wav_every_flush_seconds(tmp_path: Path) -> None:
    """閉じる前でも、flush 済みの分までは読める wav になっている(落ちた場合に相当)。"""
    path = tmp_path / "a.wav"
    writer = WavWriter(path, sample_rate=1_000, flush_seconds=1.0)
    writer.write(np.ones(600, dtype=np.int16))
    assert writer.written_bytes == WAV_HEADER_BYTES
    writer.write(np.ones(600, dtype=np.int16))
    assert writer.written_bytes == WAV_HEADER_BYTES + 1_200 * 2

    params, pcm = _read_wav(path)
    assert params.nframes == 1_200
    assert pcm.size == 1_200

    writer.write(np.ones(300, dtype=np.int16))
    writer.close()
    assert _read_wav(path)[0].nframes == 1_500


def test_wav_writer_empty_recording_is_valid_wav(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    WavWriter(path).close()
    params, pcm = _read_wav(path)
    assert params.nframes == 0
    assert pcm.size == 0


def test_wav_writer_does_not_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        WavWriter(path)
    assert path.read_bytes() == b"existing"


def test_recorder_converts_source_to_16k_mono(tmp_path: Path) -> None:
    src = tmp_path / "src.wav"
    _write_wav(src, seconds=2.0, sample_rate=48_000, channels=2)
    out = tmp_path / "out.wav"
    recorder = Recorder(
        WavFileSource(src),
        WavWriter(out),
        stop_event=threading.Event(),
        console=_console(),
        ui_refresh_seconds=0.05,
    )
    assert recorder.run() == 0

    params, pcm = _read_wav(out)
    assert (params.nchannels, params.sampwidth, params.framerate) == (1, 2, 16_000)
    assert abs(pcm.size - 32_000) <= 10
    assert np.abs(pcm).max() > 5_000
    assert recorder.progress.recorded_seconds == pytest.approx(2.0, abs=0.01)
    assert recorder.progress.written_bytes == out.stat().st_size


def test_recorder_capture_failure_keeps_file_and_exits_1(tmp_path: Path) -> None:
    class FailingSource:
        sample_rate = 16_000
        channels = 1

        def __iter__(self):
            for _ in range(5):
                yield np.full((1_600, 1), 0.1, dtype=np.float32)
            raise OSError("device lost")

        def close(self) -> None:
            return None

    warnings: list[str] = []
    out = tmp_path / "out.wav"
    recorder = Recorder(
        FailingSource(),
        WavWriter(out),
        stop_event=threading.Event(),
        console=_console(),
        ui_refresh_seconds=0.05,
        on_warning=warnings.append,
    )
    assert recorder.run() == 1
    assert any("device lost" in w for w in warnings)
    assert abs(_read_wav(out)[1].size - 8_000) <= 10


def test_recorder_write_failure_stops_and_exits_1(tmp_path: Path) -> None:
    class EndlessSource:
        sample_rate = 16_000
        channels = 1

        def __init__(self) -> None:
            self.blocks = 0

        def __iter__(self):
            while True:
                self.blocks += 1
                yield np.zeros((1_600, 1), dtype=np.float32)

        def close(self) -> None:
            return None

    class FailingWriter(WavWriter):
        def write(self, samples: np.ndarray) -> None:
            raise OSError("disk full")

    warnings: list[str] = []
    source = EndlessSource()
    recorder = Recorder(
        source,
        FailingWriter(tmp_path / "out.wav"),
        stop_event=threading.Event(),
        console=_console(),
        ui_refresh_seconds=0.05,
        on_warning=warnings.append,
    )
    assert recorder.run() == 1
    assert sum("disk full" in w for w in warnings) == 1
    assert source.blocks < 100


def test_recorder_stop_while_source_silent_finalizes(tmp_path: Path) -> None:
    """ブロックが来ない(無音の WASAPI 相当)ときに stop しても、ファイルを確定して終わる。"""

    class SilentAfterAudioSource:
        sample_rate = 16_000
        channels = 1

        def __init__(self) -> None:
            self._closed = threading.Event()

        def __iter__(self):
            for _ in range(10):
                yield np.zeros((1_600, 1), dtype=np.float32)
            self._closed.wait()

        def close(self) -> None:
            self._closed.set()

    stop = threading.Event()
    out = tmp_path / "out.wav"
    recorder = Recorder(
        SilentAfterAudioSource(),
        WavWriter(out),
        stop_event=stop,
        console=_console(),
        ui_refresh_seconds=0.05,
    )
    result: list[int] = []
    runner = threading.Thread(target=lambda: result.append(recorder.run()), daemon=True)
    runner.start()
    deadline = time.monotonic() + 5.0
    while recorder.progress.recorded_seconds < 1.0 and time.monotonic() < deadline:
        time.sleep(0.01)

    stop.set()
    runner.join(timeout=5.0)
    assert not runner.is_alive(), "stop 後に録音が終了しない"
    assert result == [0]
    assert abs(_read_wav(out)[1].size - 16_000) <= 10
