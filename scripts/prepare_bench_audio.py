#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "datasets>=3.2,<4",
#   "huggingface-hub>=0.24",
#   "numpy>=2.1",
#   "soundfile>=0.12.1",
#   "soxr>=0.5.0",
# ]
# ///
"""モデル検証用の wav と正解テキストを、ReazonSpeech からローカルに作る。

Hugging Face の reazon-research/reazonspeech（tiny）を先頭から読み、
発話を丸ごと足して指定秒数に達したところで止める。出力は 16bit PCM・
16kHz・モノラルの wav と、同じ順の書き起こし。

生成物は情報解析（著作権法30条の4）のためのローカルファイルであり、
リポジトリには含めない。scripts/data/ は .gitignore 済み。

使い方:

    # 事前にブラウザで利用条件へ同意する（スクリプトからはできない）
    # https://huggingface.co/datasets/reazon-research/reazonspeech
    uv run scripts/prepare_bench_audio.py
    uv run scripts/prepare_bench_audio.py --seconds 180

未ログインなら、対話端末ではブラウザログインを始める。トークンを渡す場合:

    HF_TOKEN=hf_... uv run scripts/prepare_bench_audio.py

datasets 4 以降はデータセットのローディングスクリプトを実行できないため、
依存は 4 未満に固定している。
"""

from __future__ import annotations

import argparse
import io
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from datasets import Audio, load_dataset
from huggingface_hub import get_token, login

DATASET_ID = "reazon-research/reazonspeech"
DATASET_URL = "https://huggingface.co/datasets/reazon-research/reazonspeech"
CONFIG = "tiny"
SAMPLE_RATE = 16_000
DEFAULT_SECONDS = 120.0
RECOMMENDED_MIN_SECONDS = 60.0
RECOMMENDED_MAX_SECONDS = 180.0
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
WAV_NAME = "sample.wav"
TEXT_NAME = "sample.txt"

AGREE_HINT = (
    "ReazonSpeech を取得できません。ブラウザでログインし、利用条件に同意してください。\n"
    f"  {DATASET_URL}\n"
    "同意はこのスクリプトからはできません。同意済みのアカウントで再実行してください。\n"
    "  uvx hf auth login\n"
    "  HF_TOKEN=... uv run scripts/prepare_bench_audio.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ReazonSpeech からモデル検証用の wav と正解テキストを作る。",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help=(
            "出力する音声の最短の長さ（秒）。発話の途中では切らない。"
            f"既定は {DEFAULT_SECONDS:g}。検証の目安は "
            f"{RECOMMENDED_MIN_SECONDS:g}〜{RECOMMENDED_MAX_SECONDS:g}"
        ),
    )
    return parser.parse_args()


def ensure_logged_in() -> None:
    """保存済みトークンが無ければ、対話端末でログインを始める。"""
    if get_token() is not None:
        return
    print(
        "Hugging Face に未ログインです。\n"
        "データセットの利用条件への同意は、ブラウザで一度だけ行ってください。\n"
        f"  {DATASET_URL}",
        file=sys.stderr,
    )
    if not sys.stdin.isatty():
        print(
            "対話端末ではないため、ログインを開始できません。次のいずれかで再実行してください。\n"
            "  uvx hf auth login\n"
            "  HF_TOKEN=... uv run scripts/prepare_bench_audio.py",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("ブラウザでログインします。表示されたコードを承認してください。", file=sys.stderr)
    login()
    if get_token() is None:
        print("ログインできませんでした。", file=sys.stderr)
        raise SystemExit(1)


def is_access_error(exc: BaseException) -> bool:
    """ゲート付きデータセットの未同意・未認証かを、例外の連鎖から判定する。"""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in {"GatedRepoError", "GatedRepoAccessError"}:
            return True
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status in {401, 403}:
            return True
        current = current.__cause__ or current.__context__
    return False


def to_mono(audio: np.ndarray) -> np.ndarray:
    """float の波形をモノラル 1 次元にする。"""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1).astype(np.float32)
    if samples.ndim != 1:
        raise ValueError(f"音声の次元が想定外です: {samples.ndim}")
    return samples


def resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == SAMPLE_RATE:
        return audio
    converted = soxr.resample(audio, sample_rate, SAMPLE_RATE)
    return np.asarray(converted, dtype=np.float32)


def read_clip(raw: dict) -> tuple[np.ndarray, int]:
    """デコード前の音声（bytes またはパス）を float32 の波形にする。

    datasets の自動デコードは librosa を要求するため、soundfile で読む。
    """
    payload = raw.get("bytes")
    if payload is not None:
        source = io.BytesIO(payload)
    else:
        path = raw.get("path")
        if not path:
            raise ValueError("音声データがありません")
        source = path
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def collect(seconds: float) -> tuple[np.ndarray, list[str], list[str]]:
    """先頭から発話を足し、累計が seconds 以上になるまで読む。

    戻り値は (波形, 書き起こし, 発話ID)。発話の途中では切らない。
    """
    dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    target = int(seconds * SAMPLE_RATE)
    pieces: list[np.ndarray] = []
    lines: list[str] = []
    names: list[str] = []
    total = 0
    print(
        f"{DATASET_ID} の {CONFIG} を先頭から読みます。累計 {seconds:g} 秒に達したら停止します。",
        file=sys.stderr,
    )
    for example in dataset:
        text = str(example["transcription"]).strip()
        audio, sample_rate = read_clip(example["audio"])
        audio = to_mono(audio)
        audio = resample(audio, sample_rate)
        if audio.size == 0 or not text:
            continue
        pieces.append(audio)
        lines.append(text)
        name = str(example.get("name", ""))
        names.append(name)
        total += int(audio.shape[0])
        print(
            f"{name or '(無名)'}  {audio.shape[0] / SAMPLE_RATE:.1f}秒"
            f"  累計 {total / SAMPLE_RATE:.1f}秒",
            file=sys.stderr,
        )
        if total >= target:
            break
    if total < target:
        raise RuntimeError(
            f"データセットを最後まで読みましたが {seconds:g} 秒に足りません"
            f"（{total / SAMPLE_RATE:.1f}秒）。"
        )
    return np.concatenate(pieces), lines, names


def write_wav(path: Path, audio: np.ndarray) -> None:
    """16bit PCM・モノラルの wav を書く。"""
    clipped = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".wav.partial")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(clipped.tobytes())
    temporary.replace(path)


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".txt.partial")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.seconds <= 0:
        print("--seconds は正の数にしてください。", file=sys.stderr)
        raise SystemExit(2)
    if not (RECOMMENDED_MIN_SECONDS <= args.seconds <= RECOMMENDED_MAX_SECONDS):
        print(
            "検証の目安は "
            f"{RECOMMENDED_MIN_SECONDS:g}〜{RECOMMENDED_MAX_SECONDS:g} 秒です"
            f"（指定は {args.seconds:g} 秒）。",
            file=sys.stderr,
        )
    ensure_logged_in()
    try:
        audio, lines, names = collect(args.seconds)
    except Exception as exc:
        if is_access_error(exc):
            print(AGREE_HINT, file=sys.stderr)
            raise SystemExit(1) from exc
        raise
    wav_path = OUTPUT_DIR / WAV_NAME
    text_path = OUTPUT_DIR / TEXT_NAME
    write_wav(wav_path, audio)
    write_text(text_path, lines)
    duration = audio.shape[0] / SAMPLE_RATE
    print(
        "書き出しました。このファイルは Git に含めません。\n"
        f"  {wav_path}  ({duration:.1f}秒, {len(lines)}発話)\n"
        f"  {text_path}\n"
        "モデル検証:\n"
        f"  uv run scripts/bench_models.py {wav_path} --reference {text_path}"
    )
    if names:
        print("使用した発話:", file=sys.stderr)
        for name in names:
            print(f"  {name}", file=sys.stderr)


if __name__ == "__main__":
    main()
