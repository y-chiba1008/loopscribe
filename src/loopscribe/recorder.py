"""システム音声の録音(wav への追記)。文字起こしはしない。"""

from __future__ import annotations

import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray
from rich.console import Console

from loopscribe.audio.source import AudioSource
from loopscribe.capture import AudioCapture
from loopscribe.constants import RECORD_FLUSH_SECONDS, SAMPLE_RATE, UI_REFRESH_SECONDS
from loopscribe.progress import RecordingReporter

# 標準ライブラリの wave が書く PCM のヘッダ(RIFF + fmt + data)
WAV_HEADER_BYTES = 44
_SAMPLE_WIDTH = 2


class WavWriter:
    """モノラル・16bit PCM のサンプルを、新規の wav に追記する。

    `flush_seconds` ぶん溜まるごとに書いて flush する。`wave` は書くたびにヘッダの長さを
    更新するので、途中で落ちても、そこまでは読める wav が残る。
    既存のファイルは上書きしない(あれば `FileExistsError`)。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        sample_rate: int = SAMPLE_RATE,
        flush_seconds: float = RECORD_FLUSH_SECONDS,
    ) -> None:
        self._path = Path(path)
        self._file: BinaryIO = self._path.open("xb")
        try:
            self._wav = wave.open(self._file, "wb")  # noqa: SIM115
            self._wav.setnchannels(1)
            self._wav.setsampwidth(_SAMPLE_WIDTH)
            self._wav.setframerate(sample_rate)
        except Exception:
            self._file.close()
            raise
        self._flush_frames = max(1, round(flush_seconds * sample_rate))
        self._pending: list[NDArray[np.int16]] = []
        self._pending_frames = 0
        self._written_frames = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def written_bytes(self) -> int:
        """ファイルに書いた量(ヘッダを含む)。"""
        return WAV_HEADER_BYTES + self._written_frames * _SAMPLE_WIDTH

    def write(self, samples: NDArray[np.int16]) -> None:
        if samples.size == 0:
            return
        self._pending.append(samples)
        self._pending_frames += samples.size
        if self._pending_frames >= self._flush_frames:
            self.flush()

    def flush(self) -> None:
        """溜めたサンプルを書き、ファイルを flush する。"""
        if self._pending:
            data = np.concatenate(self._pending).astype("<i2", copy=False)
            self._pending = []
            self._pending_frames = 0
            self._wav.writeframes(data.tobytes())
            self._written_frames += data.size
        self._file.flush()

    def close(self) -> None:
        """残りを書いて、ヘッダを確定して閉じる。"""
        try:
            self.flush()
        finally:
            try:
                self._wav.close()
            finally:
                self._file.close()


@dataclass
class RecordingProgress:
    """共有の進捗カウンタ(メインスレッドが読む)。"""

    recorded_seconds: float = 0.0
    written_bytes: int = WAV_HEADER_BYTES


class Recorder:
    """capture スレッドで AudioSource から読み、WavWriter に追記する。"""

    def __init__(
        self,
        source: AudioSource,
        writer: WavWriter,
        *,
        stop_event: threading.Event,
        console: Console,
        sample_rate: int = SAMPLE_RATE,
        ui_refresh_seconds: float = UI_REFRESH_SECONDS,
        reporter: RecordingReporter | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self._writer = writer
        self._stop_event = stop_event
        self._console = console
        self._sample_rate = sample_rate
        self._ui_refresh_seconds = ui_refresh_seconds
        self._reporter = reporter
        self._on_warning = on_warning
        self._progress = RecordingProgress()
        self._lock = threading.Lock()
        self._recorded_samples = 0
        self._write_failed = False
        self._capture = AudioCapture(
            source,
            self._write,
            stop_event=stop_event,
            on_warning=self._warn,
            sample_rate=sample_rate,
        )

    @property
    def progress(self) -> RecordingProgress:
        return self._progress

    def run(self) -> int:
        """録音して wav を確定し、終了コードを返す(0=正常、1=取得または書き込みの失敗)。"""
        capture = self._capture
        capture.start()
        try:
            while capture.is_alive():
                self._refresh_reporter()
                capture.close_source_if_stopped()
                capture.join(timeout=self._ui_refresh_seconds)
        finally:
            try:
                self._writer.close()
            except Exception as exc:
                if not self._write_failed:
                    self._write_failed = True
                    self._warn(f"wav の確定に失敗しました: {exc}")
            self._update_progress()
            self._refresh_reporter()
        return 1 if capture.failed or self._write_failed else 0

    def _write(self, samples: NDArray[np.int16]) -> None:
        if self._write_failed:
            return
        try:
            self._writer.write(samples)
        except Exception as exc:
            self._write_failed = True
            self._warn(f"wav の書き込みに失敗したため録音を停止します: {exc}")
            self._stop_event.set()
            return
        self._recorded_samples += samples.size
        self._update_progress()

    def _update_progress(self) -> None:
        with self._lock:
            self._progress.recorded_seconds = self._recorded_samples / self._sample_rate
            self._progress.written_bytes = self._writer.written_bytes

    def _refresh_reporter(self) -> None:
        if self._reporter is None:
            return
        with self._lock:
            recorded = self._progress.recorded_seconds
            written = self._progress.written_bytes
        self._reporter.update(recorded, written)

    def _warn(self, message: str) -> None:
        if self._on_warning is not None:
            self._on_warning(message)
        else:
            self._console.print(f"[yellow]警告:[/yellow] {message}")
