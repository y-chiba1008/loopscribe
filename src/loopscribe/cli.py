"""Typer による CLI。サブコマンド: start / record / download-model / devices。"""

from __future__ import annotations

import contextlib
import importlib.metadata
import os
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from loopscribe.audio.source import AudioSource
from loopscribe.constants import OUTPUT_NAME_FORMAT, RECORDING_SUFFIX, TRANSCRIPT_SUFFIX
from loopscribe.errors import FatalStartupError, InvalidOutputNameError, ModelUnavailableError
from loopscribe.models import DEFAULT_MODEL, MODELS, get_model
from loopscribe.output import (
    ConsoleSink,
    MultiSink,
    TxtSink,
    normalize_output_name,
    unique_output_path,
)
from loopscribe.pipeline import PipelineProgress, TranscriptionPipeline
from loopscribe.progress import ProgressReporter, RecordingReporter, model_loading_status
from loopscribe.recorder import Recorder, WavWriter

app = typer.Typer(
    name="loopscribe",
    help="Windowsのシステム音声をローカルで日本語文字起こしする",
    no_args_is_help=True,
    add_completion=False,
)

# `--model` の選択肢。typer は Enum から選択肢とヘルプを作る
ModelName = StrEnum("ModelName", {name: name for name in MODELS})
_DEFAULT_MODEL_NAME = ModelName(DEFAULT_MODEL)

ModelOption = Annotated[
    ModelName,
    typer.Option(
        "--model",
        help="文字起こしのモデル。kotoba は small より高精度だが、遅くRAMも多く使う",
    ),
]

DeviceOption = Annotated[
    str | None,
    typer.Option("--device", help="出力デバイス(番号、または名前の一部)。既定は既定デバイス"),
]

OutputDirOption = Annotated[
    Path,
    typer.Option(
        "-o",
        "--output-dir",
        help="出力ファイルの保存先ディレクトリ(既定: カレントディレクトリ)",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=False,
    ),
]

OutputNameOption = Annotated[
    str | None,
    typer.Option(
        "-n",
        "--output-name",
        help=(
            "出力ファイルの名前(拡張子なし。末尾の .txt / .wav は取り除く)。"
            "同名のファイルがあれば _2, _3 … を付ける。既定: loopscribe_YYYYMMDD_HHMMSS"
        ),
    ),
]


def _version_callback(value: bool) -> None:
    if value:
        version = importlib.metadata.version("loopscribe")
        typer.echo(f"loopscribe {version}")
        raise typer.Exit(0)


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="バージョンを表示して終了する",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """loopscribe CLI。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _make_console() -> Console:
    return Console(highlight=False)


def _require_windows(console: Console) -> None:
    if sys.platform != "win32":
        console.print("[red]エラー:[/red] 音声取得は Windows でのみ利用できます。")
        raise typer.Exit(1)


def _load_wasapi():
    """wasapi を遅延 import する。未実装・非Windows向け依存欠落時は FatalStartupError。"""
    try:
        from loopscribe.audio import wasapi
    except ImportError as exc:
        raise FatalStartupError(
            "WASAPI ループバック取得が利用できません"
            f"(pyaudiowpatch の導入、または wasapi 実装を確認してください): {exc}"
        ) from exc
    return wasapi


def resolve_output_path(console: Console, output_dir: Path, stem: str, suffix: str) -> Path:
    """出力先のパスを決める。同名のファイルがあれば連番を付けて警告する。"""
    path = unique_output_path(output_dir, stem, suffix)
    if path.name != f"{stem}{suffix}":
        console.print(
            f"[yellow]警告:[/yellow] {stem}{suffix} は既にあるため、{path.name} に保存します。"
        )
    return path


def _resolve_named_output(
    console: Console, output_dir: Path, output_name: str | None, suffix: str
) -> Path | None:
    """`-n` の名前を検査し、出力先を決める(未指定なら None)。不正な名前は終了コード1。"""
    if output_name is None:
        return None
    try:
        stem = normalize_output_name(output_name)
    except InvalidOutputNameError as exc:
        console.print(f"[red]エラー:[/red] {exc}")
        raise typer.Exit(1) from exc
    return resolve_output_path(console, output_dir, stem, suffix)


def _open_loopback_source(console: Console, device: str | None) -> AudioSource:
    """WASAPI ループバックの取得を開く。失敗したら終了コード1。"""
    try:
        wasapi = _load_wasapi()
        return wasapi.open_loopback_source(device)
    except FatalStartupError as exc:
        console.print(f"[red]エラー:[/red] {exc}")
        if "デバイス" in str(exc):
            console.print("`loopscribe devices` で一覧を確認してください。")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]エラー:[/red] 音声デバイスの開始に失敗しました: {exc}")
        console.print("`loopscribe devices` で一覧を確認してください。")
        raise typer.Exit(1) from exc


@contextlib.contextmanager
def _handle_sigint(
    console: Console,
    stop: threading.Event,
    *,
    first_message: str,
    on_force_exit: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Ctrl+C 1回目で `stop` をセットする。

    2回目は `on_force_exit` を呼んで、終了コード130で即時終了する。
    """
    interrupt_count = 0

    def _on_sigint(_signum: int, _frame: object) -> None:
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count == 1:
            console.print(f"\n停止信号を受け取りました。{first_message}")
            stop.set()
            return
        console.print("\n2回目の停止信号です。直ちに終了します。")
        if on_force_exit is not None:
            with contextlib.suppress(Exception):
                on_force_exit()
        os._exit(130)

    previous = signal.signal(signal.SIGINT, _on_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def run_transcription(
    source: AudioSource,
    *,
    console: Console,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    output_path: Path | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """AudioSource から文字起こしを実行する(CLI と開発用スクリプトの共通経路)。

    `output_path` は `resolve_output_path` で決めた txt のパス。
    未指定なら、開始時刻から `OUTPUT_NAME_FORMAT` で作り、`output_dir` に置く。
    """
    stop = stop_event if stop_event is not None else threading.Event()

    try:
        spec = get_model(model_name)
        model_path = spec.ensure()
    except (ValueError, ModelUnavailableError) as exc:
        console.print(f"[red]エラー:[/red] {exc}")
        return 1

    with model_loading_status(console, model_name=spec.repo_id):
        transcriber = spec.load(model_path)

    started_at = datetime.now()
    txt_path = output_path or resolve_output_path(
        console, output_dir, started_at.strftime(OUTPUT_NAME_FORMAT), TRANSCRIPT_SUFFIX
    )
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            txt_sink = TxtSink(txt_path, started_at=started_at, model_name=spec.repo_id)
            break
        except FileExistsError:
            # モデルの読み込み中に、同名のファイルが作られた場合
            txt_path = resolve_output_path(console, txt_path.parent, txt_path.stem, txt_path.suffix)

    sink = MultiSink([ConsoleSink(console), txt_sink])
    with _handle_sigint(
        console, stop, first_message="録音を止め、残りを処理します...", on_force_exit=sink.close
    ):
        console.print(f"録音を開始します ({started_at.strftime('%Y-%m-%d %H:%M:%S')})")
        console.print(f"出力: {txt_path}")

        progress = PipelineProgress()
        with ProgressReporter(console) as reporter:
            pipeline = TranscriptionPipeline(
                source,
                transcriber,
                sink,
                stop_event=stop,
                console=console,
                progress=progress,
                reporter=reporter,
            )
            return pipeline.run()


def run_recording(
    source: AudioSource,
    *,
    console: Console,
    output_dir: Path,
    output_path: Path | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """AudioSource を wav に録音する。

    `output_path` は `resolve_output_path` で決めた wav のパス。
    未指定なら、開始時刻から `OUTPUT_NAME_FORMAT` で作り、`output_dir` に置く。
    """
    stop = stop_event if stop_event is not None else threading.Event()

    started_at = datetime.now()
    wav_path = output_path or resolve_output_path(
        console, output_dir, started_at.strftime(OUTPUT_NAME_FORMAT), RECORDING_SUFFIX
    )
    try:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                writer = WavWriter(wav_path)
                break
            except FileExistsError:
                wav_path = resolve_output_path(
                    console, wav_path.parent, wav_path.stem, wav_path.suffix
                )
    except OSError as exc:
        with contextlib.suppress(Exception):
            source.close()
        console.print(f"[red]エラー:[/red] wav を作成できません: {exc}")
        return 1

    with _handle_sigint(console, stop, first_message="録音を止め、ファイルを確定します..."):
        console.print(f"録音を開始します ({started_at.strftime('%Y-%m-%d %H:%M:%S')})")
        console.print(f"出力: {wav_path}")

        with RecordingReporter(console) as reporter:
            recorder = Recorder(source, writer, stop_event=stop, console=console, reporter=reporter)
            code = recorder.run()
        console.print(f"録音を終了しました: {wav_path}")
        return code


@app.command("download-model")
def download_model_cmd(model: ModelOption = _DEFAULT_MODEL_NAME) -> None:
    """モデルを事前に取得する。"""
    console = _make_console()
    spec = get_model(model.value)
    console.print(f"モデルを取得します: {spec.repo_id}")
    try:
        path = spec.download()
    except ModelUnavailableError as exc:
        console.print(f"[red]エラー:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"取得完了: {path}")


@app.command("devices")
def devices_cmd() -> None:
    """WASAPI の出力デバイス(ループバック対象)を一覧表示する。"""
    console = _make_console()
    _require_windows(console)
    try:
        wasapi = _load_wasapi()
        devices = wasapi.list_output_devices()
        wasapi.print_device_list(devices, console=console)
    except FatalStartupError as exc:
        console.print(f"[red]エラー:[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]エラー:[/red] デバイス一覧の取得に失敗しました: {exc}")
        raise typer.Exit(1) from exc


@app.command("start")
def start_cmd(
    device: DeviceOption = None,
    output_dir: OutputDirOption = Path("."),
    output_name: OutputNameOption = None,
    model: ModelOption = _DEFAULT_MODEL_NAME,
) -> None:
    """システム音声の取得と文字起こしを開始する。"""
    console = _make_console()
    _require_windows(console)
    txt_path = _resolve_named_output(console, output_dir, output_name, TRANSCRIPT_SUFFIX)
    source = _open_loopback_source(console, device)
    code = run_transcription(
        source,
        console=console,
        output_dir=output_dir,
        model_name=model.value,
        output_path=txt_path,
    )
    raise typer.Exit(code)


@app.command("record")
def record_cmd(
    device: DeviceOption = None,
    output_dir: OutputDirOption = Path("."),
    output_name: OutputNameOption = None,
) -> None:
    """システム音声を wav(16kHz・モノラル・16bit PCM)に録音する。文字起こしはしない。"""
    console = _make_console()
    _require_windows(console)
    wav_path = _resolve_named_output(console, output_dir, output_name, RECORDING_SUFFIX)
    source = _open_loopback_source(console, device)
    code = run_recording(source, console=console, output_dir=output_dir, output_path=wav_path)
    raise typer.Exit(code)
