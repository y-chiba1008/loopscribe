"""pipeline の統合テスト(偽 Transcriber)。"""

from __future__ import annotations

import threading
import time
import wave
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from loopscribe.audio.wavfile import WavFileSource
from loopscribe.output import AbsoluteSegment, ConsoleSink, MultiSink, TxtSink
from loopscribe.pipeline import PipelineProgress, TranscriptionPipeline
from loopscribe.transcriber import FakeTranscriber, Segment


def _write_silence_wav(path: Path, *, seconds: float, sample_rate: int = 16_000) -> None:
    n = int(seconds * sample_rate)
    pcm = np.zeros(n, dtype="<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _run_pipeline(
    wav_path: Path,
    transcriber: FakeTranscriber,
    *,
    chunk_seconds: float = 4.0,
    overlap_seconds: float = 1.0,
    min_tail_seconds: float = 1.0,
    sink=None,
    console: Console | None = None,
) -> tuple[int, list[str], PipelineProgress]:
    stream = StringIO()
    console = console or Console(file=stream, force_terminal=False, width=120, highlight=False)
    lines: list[str] = []

    class CollectSink:
        def write(self, segment: AbsoluteSegment) -> None:
            lines.append(segment.text)

        def close(self) -> None:
            return

    stop = threading.Event()
    source = WavFileSource(wav_path, block_frames=512)
    progress = PipelineProgress()
    pipeline = TranscriptionPipeline(
        source,
        transcriber,
        sink if sink is not None else CollectSink(),
        stop_event=stop,
        console=console,
        sample_rate=16_000,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        min_tail_seconds=min_tail_seconds,
        lag_warn_seconds=10_000.0,
        buffer_warn_mb=10_000.0,
        wait_timeout=0.05,
        progress=progress,
    )
    code = pipeline.run()
    return code, lines, progress


def test_pipeline_emits_owned_segments_without_dupes(tmp_path: Path) -> None:
    """重なり境界のセグメントは後続チャンクだけが採用する。"""
    wav = tmp_path / "a.wav"
    _write_silence_wav(wav, seconds=7.0)

    # chunk=4, overlap=1, step=3
    # chunk0 担当 [0, 3.5), chunk1 最終 [3.5, inf)
    # abs 3.6 は chunk0 からは除外、chunk1 から採用
    fake = FakeTranscriber(
        [
            [Segment(start=3.6, end=3.8, text="境界A"), Segment(start=1.0, end=1.2, text="前半")],
            [Segment(start=0.6, end=0.8, text="境界A"), Segment(start=2.0, end=2.2, text="後半")],
        ]
    )
    code, lines, progress = _run_pipeline(wav, fake)
    assert code == 0
    assert lines == ["前半", "境界A", "後半"]
    assert progress.processed_seconds == pytest.approx(7.0, abs=0.05)
    assert len(fake.calls) >= 2


def test_pipeline_processes_tail_chunk(tmp_path: Path) -> None:
    wav = tmp_path / "b.wav"
    # 5秒: chunk0 [0,4] のあと、開始3から残り2秒 ≥ min_tail → 端数
    _write_silence_wav(wav, seconds=5.0)
    fake = FakeTranscriber(
        [
            [Segment(start=0.5, end=0.7, text="一")],
            # 端数チャンク開始が 3 秒、担当下限は 3.5 なので start=0.6 → abs 3.6
            [Segment(start=0.6, end=0.8, text="尾")],
        ]
    )
    code, lines, _progress = _run_pipeline(wav, fake, min_tail_seconds=1.0)
    assert code == 0
    assert lines == ["一", "尾"]
    assert len(fake.calls) == 2


def test_pipeline_skips_chunk_on_transcriber_error(tmp_path: Path) -> None:
    wav = tmp_path / "c.wav"
    _write_silence_wav(wav, seconds=4.0)
    warnings: list[str] = []
    fake = FakeTranscriber(
        [[Segment(start=0.1, end=0.2, text="ok")]],
        error_on_call=0,
    )
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    stop = threading.Event()
    source = WavFileSource(wav)
    pipeline = TranscriptionPipeline(
        source,
        fake,
        ConsoleSink(console),
        stop_event=stop,
        console=console,
        chunk_seconds=4.0,
        overlap_seconds=1.0,
        min_tail_seconds=1.0,
        wait_timeout=0.05,
        on_warning=warnings.append,
    )
    assert pipeline.run() == 0
    assert any("スキップ" in w for w in warnings)
    assert console.file.getvalue() == "" or "[00:" not in stream.getvalue()


def test_pipeline_stop_event_ends_capture_and_flushes(tmp_path: Path) -> None:
    """stop_event で取得を止め、それまでに溜まった分は処理する。"""
    stop = threading.Event()

    class CountingSource:
        sample_rate = 16_000
        channels = 1

        def __init__(self) -> None:
            self.blocks_yielded = 0
            self.closed = False

        def __iter__(self):
            # 0.1 秒ブロック。数個読んだら stop を立て、pipeline 側で打ち切る
            for i in range(200):
                self.blocks_yielded += 1
                if i >= 5:
                    stop.set()
                yield np.zeros((1_600, 1), dtype=np.float32)

        def close(self) -> None:
            self.closed = True

    source = CountingSource()
    fake = FakeTranscriber([])
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    progress = PipelineProgress()

    class CollectSink:
        def write(self, segment: AbsoluteSegment) -> None:
            return

        def close(self) -> None:
            return

    pipeline = TranscriptionPipeline(
        source,
        fake,
        CollectSink(),
        stop_event=stop,
        console=console,
        chunk_seconds=4.0,
        overlap_seconds=1.0,
        min_tail_seconds=0.5,
        wait_timeout=0.05,
        progress=progress,
    )
    assert pipeline.run() == 0
    assert source.closed is True
    # stop 後も「現在のブロック」は処理されるので、全量よりは少ない
    assert source.blocks_yielded < 200
    assert progress.recorded_seconds > 0


def test_pipeline_stop_while_source_silent_flushes_and_exits() -> None:
    """ブロックが来ない(無音の WASAPI 相当)ときに stop しても、残りを出力して終わる。"""
    stop = threading.Event()

    class SilentAfterAudioSource:
        """4秒分を返したあと、close されるまでブロックを返さずに待つ。"""

        sample_rate = 16_000
        channels = 1

        def __init__(self) -> None:
            self._closed = threading.Event()

        def __iter__(self):
            for _ in range(40):
                yield np.zeros((1_600, 1), dtype=np.float32)
            self._closed.wait()

        def close(self) -> None:
            self._closed.set()

    lines: list[str] = []

    class CollectSink:
        def write(self, segment: AbsoluteSegment) -> None:
            lines.append(segment.text)

        def close(self) -> None:
            return

    fake = FakeTranscriber([[Segment(start=0.5, end=0.7, text="未出力")]])
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    pipeline = TranscriptionPipeline(
        SilentAfterAudioSource(),
        fake,
        CollectSink(),
        stop_event=stop,
        console=console,
        chunk_seconds=4.0,
        overlap_seconds=1.0,
        min_tail_seconds=1.0,
        wait_timeout=0.05,
        ui_refresh_seconds=0.05,
    )
    result: list[int] = []
    runner = threading.Thread(target=lambda: result.append(pipeline.run()), daemon=True)
    runner.start()

    deadline = time.monotonic() + 5.0
    while not fake.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake.calls, "最初のチャンクが処理されていない"
    # チャンク0の結果は次のチャンク待ちで保留されている
    assert lines == []

    stop.set()
    runner.join(timeout=5.0)
    assert not runner.is_alive(), "stop 後にパイプラインが終了しない"
    assert result == [0]
    assert lines == ["未出力"]


def test_pipeline_writes_txt(tmp_path: Path) -> None:
    wav = tmp_path / "d.wav"
    out = tmp_path / "out.txt"
    _write_silence_wav(wav, seconds=4.0)
    fake = FakeTranscriber([[Segment(start=0.5, end=1.0, text="こんにちは")]])
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)
    sink = MultiSink(
        [
            ConsoleSink(console),
            TxtSink(out, started_at=datetime(2026, 11, 3, 12, 0, 0), model_name="fake"),
        ]
    )
    code, _lines, _progress = _run_pipeline(wav, fake, sink=sink, console=console)
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "# loopscribe 開始:" in text
    assert "[00:00:00] こんにちは" in text
