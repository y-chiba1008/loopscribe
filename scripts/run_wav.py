#!/usr/bin/env python3
"""開発用: WAV を本番と同じパイプラインで文字起こしする。

CLI オプションにはしない。WSL での手動確認用。

使い方:
  uv run scripts/run_wav.py path/to/audio.wav
  uv run scripts/run_wav.py path/to/audio.wav -o /tmp/out
  uv run scripts/run_wav.py path/to/audio.wav --model kotoba
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from loopscribe.audio.wavfile import WavFileSource
from loopscribe.cli import run_transcription
from loopscribe.models import DEFAULT_MODEL, MODELS


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
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default=DEFAULT_MODEL,
        help=f"文字起こしのモデル(既定: {DEFAULT_MODEL})",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    console = Console(highlight=False)
    wav_path: Path = args.wav
    if not wav_path.is_file():
        console.print(f"[red]エラー:[/red] ファイルがありません: {wav_path}")
        return 1

    source = WavFileSource(wav_path)
    console.print(f"入力: {wav_path}")
    return run_transcription(
        source, console=console, output_dir=args.output_dir, model_name=args.model
    )


if __name__ == "__main__":
    raise SystemExit(main())
