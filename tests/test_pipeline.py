"""pipeline の統合テスト(偽 Transcriber)。"""

from __future__ import annotations

import math
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
from loopscribe.chunker import OwnershipBounds
from loopscribe.output import AbsoluteSegment, ConsoleSink, MultiSink, TxtSink
from loopscribe.pipeline import PipelineProgress, TranscriptionPipeline, owned_part
from loopscribe.transcriber import FakeTranscriber, Segment, Word


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


def _words(chunk_start: float, *items: tuple[float, str]) -> tuple[Word, ...]:
    """(絶対開始秒, 文字列) の列から、チャンク内の相対時刻の語を作る(長さは0.2秒)。"""
    return tuple(Word(start=t - chunk_start, end=t - chunk_start + 0.2, text=s) for t, s in items)


def test_owned_part_keeps_only_words_in_bounds() -> None:
    seg = Segment(
        start=2.8,
        end=4.1,
        text="あいうえ",
        words=_words(0.0, (2.8, "あ"), (3.2, "い"), (3.6, "う"), (3.9, "え")),
    )
    owned = owned_part(seg, chunk_start=0.0, bounds=OwnershipBounds(lo=0.0, hi=3.5))
    assert owned is not None
    assert owned.start == pytest.approx(2.8)
    assert owned.end == pytest.approx(3.4)
    assert owned.text == "あい"


def test_owned_part_uses_first_owned_word_start_and_strips() -> None:
    seg = Segment(
        start=0.1,
        end=1.1,
        text="いうえ",
        words=_words(3.0, (3.2, " い"), (3.6, " う"), (3.9, "え ")),
    )
    owned = owned_part(seg, chunk_start=3.0, bounds=OwnershipBounds(lo=3.5, hi=math.inf))
    assert owned is not None
    assert owned.start == pytest.approx(3.6)
    assert owned.end == pytest.approx(4.1)
    assert owned.text == "うえ"


def test_owned_part_returns_none_when_no_word_in_bounds() -> None:
    seg = Segment(start=0.1, end=0.5, text="い", words=_words(3.0, (3.2, "い")))
    assert owned_part(seg, chunk_start=3.0, bounds=OwnershipBounds(lo=3.5, hi=math.inf)) is None


def test_owned_part_returns_none_when_owned_words_are_blank() -> None:
    seg = Segment(start=0.6, end=1.0, text="x", words=_words(3.0, (3.6, " "), (3.9, "")))
    assert owned_part(seg, chunk_start=3.0, bounds=OwnershipBounds(lo=3.5, hi=math.inf)) is None


def test_owned_part_falls_back_to_segment_start_without_words() -> None:
    seg = Segment(start=0.6, end=0.8, text="境界")
    bounds = OwnershipBounds(lo=3.5, hi=math.inf)
    assert owned_part(seg, chunk_start=3.0, bounds=bounds) == AbsoluteSegment(
        start=3.6, end=3.8, text="境界"
    )
    assert owned_part(seg, chunk_start=2.0, bounds=bounds) is None


def test_pipeline_splits_boundary_segment_by_words(tmp_path: Path) -> None:
    """境目をまたぐセグメントは語単位で分かれ、重複も欠落も出ない。"""
    wav = tmp_path / "w.wav"
    _write_silence_wav(wav, seconds=7.0)

    # chunk=4, overlap=1, step=3。chunk0 [0, 4] の担当 [0, 3.5)、chunk1 [3, 7] の担当 [3.5, inf)
    # 両チャンクとも、境目をまたぐ発話を1つのセグメントとして返す。
    # chunk1 のセグメントの開始(abs 3.1)は chunk0 の担当範囲に寄っている(Issue #12 の状況)
    fake = FakeTranscriber(
        [
            [
                Segment(start=1.0, end=1.4, text="前半", words=_words(0.0, (1.0, "前半"))),
                Segment(
                    start=2.8,
                    end=4.0,
                    text="あいうえ",
                    words=_words(0.0, (2.8, "あ"), (3.2, "い"), (3.6, "う"), (3.8, "え")),
                ),
            ],
            [
                Segment(
                    start=0.1,
                    end=1.1,
                    text="いうえお",
                    words=_words(3.0, (3.2, "い"), (3.6, "う"), (3.8, "え"), (4.0, "お")),
                ),
                Segment(start=2.0, end=2.4, text="後半", words=_words(3.0, (5.0, "後半"))),
            ],
        ]
    )
    segments: list[AbsoluteSegment] = []

    class CollectSink:
        def write(self, segment: AbsoluteSegment) -> None:
            segments.append(segment)

        def close(self) -> None:
            return

    code, _lines, _progress = _run_pipeline(wav, fake, sink=CollectSink())
    assert code == 0
    assert [s.text for s in segments] == ["前半", "あい", "うえお", "後半"]
    assert [s.start for s in segments] == pytest.approx([1.0, 2.8, 3.6, 5.0])
    assert "".join(s.text for s in segments) == "前半あいうえお後半"


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


class _CountingSource:
    """0.1秒のブロックを `n_blocks` 個返す。返した数を数える。"""

    sample_rate = 16_000
    channels = 1

    def __init__(self, n_blocks: int) -> None:
        self._n_blocks = n_blocks
        self.blocks_yielded = 0

    def __iter__(self):
        for _ in range(self._n_blocks):
            self.blocks_yielded += 1
            yield np.zeros((1_600, 1), dtype=np.float32)

    def close(self) -> None:
        return None


class _GatedTranscriber(FakeTranscriber):
    """呼ばれたことを知らせ、`release` がセットされるまで推論を終えない。"""

    def __init__(self, responses: list[list[Segment]] | None = None) -> None:
        super().__init__(responses)
        self.called = threading.Event()
        self.release = threading.Event()

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        result = super().transcribe(samples)
        self.called.set()
        assert self.release.wait(timeout=5.0)
        return result


def _start_pipeline(pipeline: TranscriptionPipeline) -> tuple[threading.Thread, list[int]]:
    result: list[int] = []
    runner = threading.Thread(target=lambda: result.append(pipeline.run()), daemon=True)
    runner.start()
    return runner, result


def _silent_console() -> Console:
    return Console(file=StringIO(), force_terminal=False, width=120, highlight=False)


def test_pipeline_read_ahead_limits_capture() -> None:
    """推論に渡していない音声が上限に達したら、取得を待たせる。"""
    source = _CountingSource(300)
    fake = _GatedTranscriber()
    pipeline = TranscriptionPipeline(
        source,
        fake,
        MultiSink([]),
        stop_event=threading.Event(),
        console=_silent_console(),
        chunk_seconds=4.0,
        overlap_seconds=1.0,
        wait_timeout=0.05,
        ui_refresh_seconds=0.05,
        read_ahead_seconds=8.0,
    )
    runner, result = _start_pipeline(pipeline)

    assert fake.called.wait(timeout=5.0)
    time.sleep(0.3)
    # チャンク0(0〜4秒)を推論中。読み込みは 4 + 8 秒で止まる
    assert source.blocks_yielded * 0.1 == pytest.approx(12.0, abs=0.2)

    fake.release.set()
    runner.join(timeout=5.0)
    assert not runner.is_alive()
    assert result == [0]
    assert source.blocks_yielded == 300
    assert pipeline.progress.processed_seconds == pytest.approx(30.0, abs=0.05)


def test_pipeline_read_ahead_must_cover_chunk() -> None:
    with pytest.raises(ValueError, match="read_ahead_seconds"):
        TranscriptionPipeline(
            _CountingSource(1),
            FakeTranscriber(),
            MultiSink([]),
            stop_event=threading.Event(),
            console=_silent_console(),
            chunk_seconds=4.0,
            overlap_seconds=1.0,
            read_ahead_seconds=3.0,
        )


def test_pipeline_stop_without_backlog_outputs_current_chunk_only() -> None:
    """停止の指示で、処理中のチャンクを終えたら、そこまでを出力して残りは処理しない。"""
    stop = threading.Event()
    source = _CountingSource(300)
    # チャンク0の 3.8秒の語は、通常ならチャンク1の担当(3.5秒〜)。中断時は最終チャンクとして出す
    fake = _GatedTranscriber(
        [
            [
                Segment(start=1.0, end=1.2, text="前半", words=_words(0.0, (1.0, "前半"))),
                Segment(start=3.8, end=4.0, text="境目", words=_words(0.0, (3.8, "境目"))),
            ],
            [Segment(start=2.0, end=2.2, text="未処理")],
        ]
    )
    segments: list[AbsoluteSegment] = []

    class CollectSink:
        def write(self, segment: AbsoluteSegment) -> None:
            segments.append(segment)

        def close(self) -> None:
            return

    pipeline = TranscriptionPipeline(
        source,
        fake,
        CollectSink(),
        stop_event=stop,
        console=_silent_console(),
        chunk_seconds=4.0,
        overlap_seconds=1.0,
        wait_timeout=0.05,
        ui_refresh_seconds=0.05,
        read_ahead_seconds=8.0,
        process_backlog_on_stop=False,
    )
    runner, result = _start_pipeline(pipeline)

    assert fake.called.wait(timeout=5.0)
    stop.set()
    fake.release.set()
    runner.join(timeout=5.0)
    assert not runner.is_alive()
    assert result == [0]
    assert len(fake.calls) == 1
    assert [s.text for s in segments] == ["前半", "境目"]
    assert source.blocks_yielded < 300
    assert pipeline.progress.processed_seconds == pytest.approx(4.0)


def test_pipeline_capture_failure_still_processes_backlog_without_backlog_mode() -> None:
    """取得の失敗による停止では、ファイル入力でも残りを処理する(終了コード1)。"""

    class FailingSource(_CountingSource):
        def __iter__(self):
            yield from super().__iter__()
            raise OSError("read error")

    lines: list[str] = []

    class CollectSink:
        def write(self, segment: AbsoluteSegment) -> None:
            lines.append(segment.text)

        def close(self) -> None:
            return

    fake = FakeTranscriber(
        [[Segment(start=0.5, end=0.7, text="一")], [Segment(start=1.0, end=1.2, text="二")]]
    )
    pipeline = TranscriptionPipeline(
        FailingSource(70),
        fake,
        CollectSink(),
        stop_event=threading.Event(),
        console=_silent_console(),
        chunk_seconds=4.0,
        overlap_seconds=1.0,
        wait_timeout=0.05,
        read_ahead_seconds=8.0,
        process_backlog_on_stop=False,
        on_warning=lambda _message: None,
    )
    assert pipeline.run() == 1
    assert lines == ["一", "二"]


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
