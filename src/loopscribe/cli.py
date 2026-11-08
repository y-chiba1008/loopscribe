"""Typer による CLI。サブコマンド: start / download-model / devices。"""

from __future__ import annotations

import contextlib
import importlib.metadata
import os
import signal
import sys
import threading
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from loopscribe.audio.source import AudioSource
from loopscribe.constants import OUTPUT_NAME_FORMAT, TRANSCRIPT_SUFFIX
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
from loopscribe.progress import ProgressReporter, model_loading_status

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


def run_transcription(
    source: AudioSource,
    *,
    console: Console,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    output_name: str | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """AudioSource から文字起こしを実行する(CLI と開発用スクリプトの共通経路)。

    `output_name` は `normalize_output_name` で検査済みの、拡張子なしの名前。
    未指定なら、開始時刻から `OUTPUT_NAME_FORMAT` で作る。
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

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    stem = output_name if output_name is not None else started_at.strftime(OUTPUT_NAME_FORMAT)
    txt_path = unique_output_path(output_dir, stem, TRANSCRIPT_SUFFIX)
    if txt_path.name != f"{stem}{TRANSCRIPT_SUFFIX}":
        console.print(
            f"[yellow]警告:[/yellow] {stem}{TRANSCRIPT_SUFFIX} は既にあるため、"
            f"{txt_path.name} に保存します。"
        )

    sink_holder: dict[str, MultiSink | None] = {"sink": None}
    interrupt_count = 0

    def _on_sigint(_signum: int, _frame: object) -> None:
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count == 1:
            console.print("\n停止信号を受け取りました。録音を止め、残りを処理します...")
            stop.set()
            return
        console.print("\n2回目の停止信号です。直ちに終了します。")
        sink = sink_holder["sink"]
        if sink is not None:
            with contextlib.suppress(Exception):
                sink.close()
        os._exit(130)

    previous = signal.signal(signal.SIGINT, _on_sigint)
    try:
        console_sink = ConsoleSink(console)
        txt_sink = TxtSink(txt_path, started_at=started_at, model_name=spec.repo_id)
        sink = MultiSink([console_sink, txt_sink])
        sink_holder["sink"] = sink

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
    finally:
        signal.signal(signal.SIGINT, previous)


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
    device: Annotated[
        str | None,
        typer.Option("--device", help="出力デバイス(番号、または名前の一部)。既定は既定デバイス"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-dir",
            help="txt の保存先ディレクトリ(既定: カレントディレクトリ)",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=False,
        ),
    ] = Path("."),
    output_name: OutputNameOption = None,
    model: ModelOption = _DEFAULT_MODEL_NAME,
) -> None:
    """システム音声の取得と文字起こしを開始する。"""
    console = _make_console()
    _require_windows(console)
    try:
        stem = normalize_output_name(output_name) if output_name is not None else None
    except InvalidOutputNameError as exc:
        console.print(f"[red]エラー:[/red] {exc}")
        raise typer.Exit(1) from exc
    try:
        wasapi = _load_wasapi()
        source = wasapi.open_loopback_source(device)
    except FatalStartupError as exc:
        console.print(f"[red]エラー:[/red] {exc}")
        if "デバイス" in str(exc):
            console.print("`loopscribe devices` で一覧を確認してください。")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]エラー:[/red] 音声デバイスの開始に失敗しました: {exc}")
        console.print("`loopscribe devices` で一覧を確認してください。")
        raise typer.Exit(1) from exc

    code = run_transcription(
        source,
        console=console,
        output_dir=output_dir,
        model_name=model.value,
        output_name=stem,
    )
    raise typer.Exit(code)
