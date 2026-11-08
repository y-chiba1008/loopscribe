"""取得・変換・チャンク文字起こし・出力のパイプライン。取得は `capture.AudioCapture` に任せる。"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rich.console import Console

from loopscribe.audio.source import AudioSource
from loopscribe.buffer import AudioBuffer
from loopscribe.capture import AudioCapture
from loopscribe.chunker import (
    ChunkSpan,
    OwnershipBounds,
    chunk_ready_at,
    chunk_span,
    is_chunk_ready,
    ownership_bounds,
    step_seconds,
    tail_span_after_stop,
)
from loopscribe.constants import (
    BUFFER_WARN_MB,
    CHUNK_SECONDS,
    LAG_WARN_SECONDS,
    MIN_TAIL_SECONDS,
    OVERLAP_SECONDS,
    SAMPLE_RATE,
    SIGNAL_POLL_SECONDS,
    UI_REFRESH_SECONDS,
)
from loopscribe.output import AbsoluteSegment, Sink
from loopscribe.progress import ProgressReporter
from loopscribe.transcriber import Segment, Transcriber


def owned_part(
    segment: Segment,
    *,
    chunk_start: float,
    bounds: OwnershipBounds,
) -> AbsoluteSegment | None:
    """セグメントのうち、チャンクの担当範囲に入る部分を絶対時刻で返す(無ければ None)。

    語があれば、絶対開始時刻が担当範囲に入る語だけを採用する。語が無ければ、
    セグメントの開始時刻で判定する。
    """
    if not segment.words:
        start = chunk_start + segment.start
        if not bounds.contains(start):
            return None
        return AbsoluteSegment(start=start, end=chunk_start + segment.end, text=segment.text)

    words = [w for w in segment.words if bounds.contains(chunk_start + w.start)]
    text = "".join(w.text for w in words).strip()
    if not text:
        return None
    return AbsoluteSegment(
        start=chunk_start + words[0].start,
        end=chunk_start + words[-1].end,
        text=text,
    )


@dataclass
class PipelineProgress:
    """共有の進捗カウンタ(メインスレッドが読む)。"""

    recorded_seconds: float = 0.0
    processed_seconds: float = 0.0
    done: bool = False
    exit_code: int = 0


class TranscriptionPipeline:
    """capture / transcribe スレッドで AudioSource から Sink まで流す。"""

    def __init__(
        self,
        source: AudioSource,
        transcriber: Transcriber,
        sink: Sink,
        *,
        stop_event: threading.Event,
        console: Console,
        sample_rate: int = SAMPLE_RATE,
        chunk_seconds: float = CHUNK_SECONDS,
        overlap_seconds: float = OVERLAP_SECONDS,
        min_tail_seconds: float = MIN_TAIL_SECONDS,
        lag_warn_seconds: float = LAG_WARN_SECONDS,
        buffer_warn_mb: float = BUFFER_WARN_MB,
        wait_timeout: float = SIGNAL_POLL_SECONDS,
        ui_refresh_seconds: float = UI_REFRESH_SECONDS,
        progress: PipelineProgress | None = None,
        reporter: ProgressReporter | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._sink = sink
        self._stop_event = stop_event
        self._console = console
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._overlap_seconds = overlap_seconds
        self._min_tail_seconds = min_tail_seconds
        self._lag_warn_seconds = lag_warn_seconds
        self._buffer_warn_mb = buffer_warn_mb
        self._wait_timeout = wait_timeout
        self._ui_refresh_seconds = ui_refresh_seconds
        self._progress = progress if progress is not None else PipelineProgress()
        self._reporter = reporter
        self._on_warning = on_warning
        self._buffer = AudioBuffer()
        self._lock = threading.Lock()
        self._lag_warn_level = 0
        self._buffer_warned = False
        self._capture = AudioCapture(
            source,
            self._append_converted,
            stop_event=stop_event,
            on_warning=self._warn,
            on_finish=self._on_capture_finished,
            sample_rate=sample_rate,
        )

    @property
    def progress(self) -> PipelineProgress:
        return self._progress

    def run(self) -> int:
        """パイプラインを実行し、終了コードを返す(0=正常、1=取得失敗後の処理完了)。

        メインスレッドは短いタイムアウトで join を繰り返し、プログレスを更新する。
        (Windows でシグナル処理が遅れないようにするため。)
        """
        capture = self._capture
        transcribe = threading.Thread(target=self._transcribe_loop, name="transcribe", daemon=True)
        capture.start()
        transcribe.start()
        try:
            while capture.is_alive() or transcribe.is_alive():
                self._refresh_reporter()
                capture.close_source_if_stopped()
                if capture.is_alive():
                    capture.join(timeout=self._ui_refresh_seconds)
                else:
                    transcribe.join(timeout=self._ui_refresh_seconds)
            self._refresh_reporter()
        finally:
            with contextlib.suppress(Exception):
                self._sink.close()
            with self._lock:
                self._progress.done = True
                self._progress.exit_code = 1 if capture.failed else 0
        return self._progress.exit_code

    def _refresh_reporter(self) -> None:
        if self._reporter is None:
            return
        with self._lock:
            processed = self._progress.processed_seconds
            recorded = self._progress.recorded_seconds
        self._reporter.update(processed, recorded)

    def _warn(self, message: str) -> None:
        if self._on_warning is not None:
            self._on_warning(message)
        else:
            self._console.print(f"[yellow]警告:[/yellow] {message}")

    def _append_converted(self, samples: NDArray[np.int16]) -> None:
        self._buffer.append(samples)
        self._refresh_recorded()
        self._maybe_warn_buffer()

    def _on_capture_finished(self) -> None:
        self._buffer.close()
        self._refresh_recorded()

    def _refresh_recorded(self) -> None:
        recorded = self._buffer.total_samples / self._sample_rate
        with self._lock:
            self._progress.recorded_seconds = recorded
            if self._progress.processed_seconds > recorded:
                self._progress.processed_seconds = recorded

    def _maybe_warn_buffer(self) -> None:
        bytes_used = int(self._buffer.total_samples) * 2
        mb = bytes_used / (1024 * 1024)
        if mb > self._buffer_warn_mb and not self._buffer_warned:
            self._buffer_warned = True
            self._warn(
                f"音声バッファが {mb:.0f}MB を超えました"
                f"(閾値 {self._buffer_warn_mb:.0f}MB)。破棄はしません。"
            )

    def _transcribe_loop(self) -> None:
        index = 0
        pending: tuple[ChunkSpan, list[Segment]] | None = None

        while True:
            need = int(
                round(
                    chunk_ready_at(
                        index,
                        chunk_seconds=self._chunk_seconds,
                        overlap_seconds=self._overlap_seconds,
                    )
                    * self._sample_rate
                )
            )
            while not self._buffer.wait_for(need, timeout=self._wait_timeout):
                if self._buffer.closed:
                    break
            recorded = self._buffer.total_samples / self._sample_rate
            if is_chunk_ready(
                index,
                recorded,
                chunk_seconds=self._chunk_seconds,
                overlap_seconds=self._overlap_seconds,
            ):
                span = chunk_span(
                    index,
                    chunk_seconds=self._chunk_seconds,
                    overlap_seconds=self._overlap_seconds,
                )
                if pending is not None:
                    self._finalize_chunk(pending[0], pending[1], is_last=False)
                segments = self._transcribe_span(span)
                pending = (span, segments)
                index += 1
                self._maybe_warn_lag()
                continue
            if self._buffer.closed:
                break

        recorded = self._buffer.total_samples / self._sample_rate
        while is_chunk_ready(
            index,
            recorded,
            chunk_seconds=self._chunk_seconds,
            overlap_seconds=self._overlap_seconds,
        ):
            span = chunk_span(
                index,
                chunk_seconds=self._chunk_seconds,
                overlap_seconds=self._overlap_seconds,
            )
            if pending is not None:
                self._finalize_chunk(pending[0], pending[1], is_last=False)
            segments = self._transcribe_span(span)
            pending = (span, segments)
            index += 1

        tail = tail_span_after_stop(
            recorded,
            index,
            chunk_seconds=self._chunk_seconds,
            overlap_seconds=self._overlap_seconds,
            min_tail_seconds=self._min_tail_seconds,
        )
        if pending is not None:
            self._finalize_chunk(pending[0], pending[1], is_last=tail is None)
            pending = None
        if tail is not None:
            segments = self._transcribe_span(tail)
            self._finalize_chunk(tail, segments, is_last=True)

        with self._lock:
            self._progress.processed_seconds = min(
                self._progress.processed_seconds,
                recorded,
            )
            # 端数が MIN_TAIL 未満で捨てた場合も、表示上は追いついたことにする
            if recorded > 0 and self._progress.processed_seconds < recorded and tail is None:
                self._progress.processed_seconds = recorded

    def _transcribe_span(self, span: ChunkSpan) -> list[Segment]:
        start = int(round(span.start * self._sample_rate))
        end = int(round(span.end * self._sample_rate))
        end = min(end, self._buffer.total_samples)
        if end <= start:
            return []
        try:
            pcm = self._buffer.read(start, end)
            samples = pcm.astype(np.float32) / 32768.0
            return self._transcriber.transcribe(samples)
        except Exception as exc:
            self._warn(
                f"チャンク {span.index} ({span.start:.1f}-{span.end:.1f}秒) の"
                f"推論に失敗したためスキップします: {exc}"
            )
            return []

    def _finalize_chunk(
        self,
        span: ChunkSpan,
        segments: list[Segment],
        *,
        is_last: bool,
    ) -> None:
        bounds = ownership_bounds(
            span.index,
            is_last=is_last,
            chunk_seconds=self._chunk_seconds,
            overlap_seconds=self._overlap_seconds,
        )
        for seg in segments:
            owned = owned_part(seg, chunk_start=span.start, bounds=bounds)
            if owned is not None:
                self._sink.write(owned)
        with self._lock:
            self._progress.processed_seconds = min(
                span.end,
                self._buffer.total_samples / self._sample_rate,
            )
        self._trim_after_chunk(span.index)
        self._maybe_warn_lag()

    def _trim_after_chunk(self, index: int) -> None:
        next_start = (index + 1) * step_seconds(self._chunk_seconds, self._overlap_seconds)
        trim_at = int(round(next_start * self._sample_rate))
        total = self._buffer.total_samples
        if trim_at <= 0:
            return
        if trim_at > total:
            # 最終チャンク後など、保持分より先は trim しない
            return
        with contextlib.suppress(ValueError):
            self._buffer.trim_before(trim_at)

    def _maybe_warn_lag(self) -> None:
        with self._lock:
            lag = self._progress.recorded_seconds - self._progress.processed_seconds
        if self._lag_warn_seconds <= 0:
            return
        level = int(lag // self._lag_warn_seconds)
        if level > self._lag_warn_level and level > 0:
            self._lag_warn_level = level
            self._warn(
                f"文字起こしの遅れが約 {lag:.0f} 秒です"
                f"(閾値の {level} 倍: {self._lag_warn_seconds:.0f}秒)。"
                "音声は破棄しません。"
            )
