#!/usr/bin/env python3
"""開発用: WAV を本番と同じパイプラインで文字起こしする。

CLI オプションにはしない。WSL での手動確認用。

使い方:
  uv run scripts/run_wav.py path/to/audio.wav
  uv run scripts/run_wav.py path/to/audio.wav -o /tmp/out
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path

from rich.console import Console

from loopscribe.audio.wavfile import WavFileSource
from loopscribe.constants import MODEL_NAME, OUTPUT_FILENAME_FORMAT
from loopscribe.errors import ModelUnavailableError
from loopscribe.model import ensure_model
from loopscribe.output import ConsoleSink, MultiSink, TxtSink
from loopscribe.pipeline import PipelineProgress, TranscriptionPipeline
from loopscribe.transcriber import FasterWhisperTranscriber


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="loopscribe 開発用: WAV を文字起こしする")
    parser.add_argument("wav", type=Path, help="入力 WAV(16bit PCM)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="txt の出力先ディレクトリ(既定: カレント)",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    console = Console(highlight=False)
    wav_path: Path = args.wav
    if not wav_path.is_file():
        console.print(f"[red]エラー:[/red] ファイルがありません: {wav_path}")
        return 1

    try:
        model_path = ensure_model(MODEL_NAME)
    except ModelUnavailableError as exc:
        console.print(f"[red]エラー:[/red] {exc}")
        return 1

    console.print(f"モデルを読み込み中: {MODEL_NAME} ...")
    transcriber = FasterWhisperTranscriber(str(model_path))
    console.print("モデルの読み込みが完了しました")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    txt_path = output_dir / started_at.strftime(OUTPUT_FILENAME_FORMAT)

    stop_event = threading.Event()
    sink_holder: dict[str, MultiSink | None] = {"sink": None}
    interrupt_count = 0

    def _on_sigint(_signum: int, _frame: object) -> None:
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count == 1:
            console.print("\n停止信号を受け取りました。録音を止め、残りを処理します...")
            stop_event.set()
            return
        console.print("\n2回目の停止信号です。直ちに終了します。")
        sink = sink_holder["sink"]
        if sink is not None:
            with contextlib.suppress(Exception):
                sink.close()
        os._exit(130)

    signal.signal(signal.SIGINT, _on_sigint)

    source = WavFileSource(wav_path)
    console_sink = ConsoleSink(console)
    txt_sink = TxtSink(txt_path, started_at=started_at, model_name=MODEL_NAME)
    sink = MultiSink([console_sink, txt_sink])
    sink_holder["sink"] = sink

    console.print(f"文字起こしを開始します: {wav_path}")
    console.print(f"出力: {txt_path}")
    progress = PipelineProgress()
    pipeline = TranscriptionPipeline(
        source,
        transcriber,
        sink,
        stop_event=stop_event,
        console=console,
        progress=progress,
    )
    code = pipeline.run()
    console.print(
        f"完了: 処理済み {progress.processed_seconds:.1f}秒"
        f" / 録音 {progress.recorded_seconds:.1f}秒"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
