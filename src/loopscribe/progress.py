"""処理済み / 録音のプログレスバー表示と、`record` の録音状況の表示。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.status import Status

from loopscribe.constants import UI_REFRESH_SECONDS
from loopscribe.output import format_timestamp


def format_duration(seconds: float) -> str:
    """経過秒を HH:MM:SS にする(プログレス用。括弧なし)。"""
    return format_timestamp(max(0.0, seconds))


def lag_seconds(processed_seconds: float, recorded_seconds: float) -> float:
    """録音に対する処理の遅れ(秒)。負にはしない。"""
    return max(0.0, recorded_seconds - processed_seconds)


def progress_ratio(processed_seconds: float, recorded_seconds: float) -> float:
    """処理済み / 録音。録音が 0 なら 0。1 を超えない。"""
    if recorded_seconds <= 0:
        return 0.0
    return min(1.0, max(0.0, processed_seconds / recorded_seconds))


class ProgressReporter:
    """処理済み時間 / 録音時間のプログレスバー。

    `total_seconds` を渡すと(ファイル入力)、合計をその値に固定し、
    「処理済み / 音声」を表示する(遅れは出さない)。
    標準出力が TTY でない場合は無効化し、本文出力だけを残す。
    """

    def __init__(
        self,
        console: Console,
        *,
        enabled: bool | None = None,
        refresh_seconds: float = UI_REFRESH_SECONDS,
        total_seconds: float | None = None,
    ) -> None:
        if enabled is None:
            enabled = bool(console.is_terminal)
        self.console = console
        self.refresh_seconds = refresh_seconds
        self._enabled = enabled
        self._total_seconds = total_seconds
        if total_seconds is None:
            columns = (
                TextColumn("処理済み {task.fields[processed]} / 録音 {task.fields[recorded]}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TextColumn("遅れ {task.fields[lag]}"),
            )
        else:
            columns = (
                TextColumn("処理済み {task.fields[processed]} / 音声 {task.fields[recorded]}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
            )
        self._progress = Progress(
            *columns,
            console=console,
            expand=True,
            transient=False,
            disable=not enabled,
            refresh_per_second=max(1.0, 1.0 / refresh_seconds) if refresh_seconds > 0 else 2.0,
        )
        self._task_id: int | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __enter__(self) -> ProgressReporter:
        self._progress.start()
        self._task_id = self._progress.add_task(
            "transcribe",
            total=1.0,
            completed=0.0,
            processed=format_duration(0.0),
            recorded=format_duration(self._total_seconds or 0.0),
            lag=format_duration(0.0),
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._progress.stop()
        self._task_id = None

    def update(self, processed_seconds: float, recorded_seconds: float) -> None:
        """共有カウンタの値でバーを更新する。"""
        if self._task_id is None:
            return
        processed = max(0.0, processed_seconds)
        if self._total_seconds is not None:
            recorded = self._total_seconds
            total = max(recorded, 1e-6)
        else:
            recorded = max(0.0, recorded_seconds)
            # recorded が伸び続ける間は total も追従させる(録音中は 100% にならない)
            total = max(recorded, processed, 1e-6)
        self._progress.update(
            self._task_id,
            completed=min(processed, total),
            total=total,
            processed=format_duration(processed),
            recorded=format_duration(recorded),
            lag=format_duration(lag_seconds(processed, recorded)),
        )


def format_size(num_bytes: int) -> str:
    """バイト数を MB で表す(小数1桁)。"""
    return f"{max(0, num_bytes) / (1024 * 1024):.1f}MB"


class RecordingReporter:
    """`record` の録音時間とファイルサイズの表示。

    標準出力が TTY でない場合は無効化する。
    """

    def __init__(
        self,
        console: Console,
        *,
        enabled: bool | None = None,
        refresh_seconds: float = UI_REFRESH_SECONDS,
    ) -> None:
        if enabled is None:
            enabled = bool(console.is_terminal)
        self.console = console
        self._enabled = enabled
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("録音 {task.fields[recorded]}  ファイル {task.fields[size]}"),
            console=console,
            transient=False,
            disable=not enabled,
            refresh_per_second=max(1.0, 1.0 / refresh_seconds) if refresh_seconds > 0 else 2.0,
        )
        self._task_id: int | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __enter__(self) -> RecordingReporter:
        self._progress.start()
        self._task_id = self._progress.add_task(
            "record",
            total=None,
            recorded=format_duration(0.0),
            size=format_size(0),
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._progress.stop()
        self._task_id = None

    def update(self, recorded_seconds: float, written_bytes: int) -> None:
        if self._task_id is None:
            return
        self._progress.update(
            self._task_id,
            recorded=format_duration(recorded_seconds),
            size=format_size(written_bytes),
        )


@contextmanager
def model_loading_status(
    console: Console,
    *,
    model_name: str,
) -> Iterator[None]:
    """モデル読み込み中の表示。TTY ならスピナー、非TTY なら1行メッセージ。"""
    message = f"モデルを読み込み中: {model_name}"
    if console.is_terminal:
        status: Status = console.status(f"{message} ...")
        status.start()
        try:
            yield
        finally:
            status.stop()
    else:
        console.print(f"{message} ...")
        yield
    console.print("モデルの読み込みが完了しました")
