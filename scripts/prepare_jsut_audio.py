#!/usr/bin/env python3
"""検証用の wav と正解テキストを、ローカルの JSUT コーパスから作る。

scripts/data/jsut_ver1.1/ に展開済みの JSUT を読み、指定したサブセットから発話を選んで
指定秒数に達するまでつなぐ。出力は prepare_bench_audio.py と同じ形式
(16bit PCM・16kHz・モノラルの <name>.wav、1行1発話の <name>.txt、発話ごとの位置の <name>.json)。
bench_models.py / eval_boundary.py / compare_boundary_rules.py / `loopscribe transcribe` に
そのまま渡せる。

発話の選び方は、サブセットの全発話を --seed で並べ替えて先頭から取る(同じ引数なら同じ組になる)。
JSUT の各ファイルには、前後に合わせて0.2秒ほどの無音がある。--gap-seconds はその外側に足す無音。

JSUT の音声は再配布できない(README_ja.txt)。生成物は scripts/data/ に置き、Git に含めない。

使い方:
    uv run scripts/prepare_jsut_audio.py
    uv run scripts/prepare_jsut_audio.py --seconds 600 --name jsut600
    uv run scripts/prepare_jsut_audio.py --seconds 600 --gap-seconds 1 --name jsut600_gap1
    uv run scripts/prepare_jsut_audio.py --subsets basic5000 travel1000 --seed 1 --name jsut_mix
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr

SAMPLE_RATE = 16_000
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data"
JSUT_DIR = OUTPUT_DIR / "jsut_ver1.1"
DEFAULT_SUBSETS = ["basic5000"]
DEFAULT_SECONDS = 600.0
DEFAULT_NAME = "jsut600"


@dataclass(frozen=True)
class Utterance:
    id: str
    path: Path
    text: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSUT から検証用の wav と正解テキストを作る。")
    p.add_argument("--jsut-dir", type=Path, default=JSUT_DIR, help=f"既定は {JSUT_DIR}")
    p.add_argument(
        "--subsets",
        nargs="+",
        default=DEFAULT_SUBSETS,
        help=f"使うサブセット(ディレクトリ名)。既定は {' '.join(DEFAULT_SUBSETS)}",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help=f"出力する音声の最短の長さ(秒、無音を含まない)。既定は {DEFAULT_SECONDS:g}",
    )
    p.add_argument("--seed", type=int, default=0, help="発話を選ぶ乱数の種。既定は 0")
    p.add_argument(
        "--gap-seconds",
        type=float,
        default=0.0,
        help="発話の間に足す無音の秒数。既定は 0(ファイルの前後の無音だけ)",
    )
    p.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help=f"出力ファイル名(拡張子なし)。scripts/data/ に書く。既定は {DEFAULT_NAME}",
    )
    args = p.parse_args(argv)
    if args.seconds <= 0:
        p.error("--seconds は正の数にしてください")
    if args.gap_seconds < 0:
        p.error("--gap-seconds は 0 以上にしてください")
    if not args.name or Path(args.name).name != args.name:
        p.error("--name にはディレクトリを含まないファイル名を指定してください")
    return args


def load_subset(jsut_dir: Path, subset: str) -> list[Utterance]:
    """サブセットの transcript_utf8.txt(`ID:本文`)を読み、wav があるものを返す。"""
    directory = jsut_dir / subset
    transcript = directory / "transcript_utf8.txt"
    if not transcript.is_file():
        raise FileNotFoundError(f"{transcript} がありません")
    utterances: list[Utterance] = []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        uid, sep, text = line.partition(":")
        if not sep or not text.strip():
            continue
        path = directory / "wav" / f"{uid.strip()}.wav"
        if path.is_file():
            utterances.append(Utterance(id=uid.strip(), path=path, text=text.strip()))
    return utterances


def read_wav(path: Path) -> np.ndarray:
    """16bit PCM の wav を float32・16kHz・モノラルで読む。"""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"16bit PCM ではありません: {path}")
        channels = wf.getnchannels()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = np.asarray(soxr.resample(audio, rate, SAMPLE_RATE), dtype=np.float32)
    return audio


def collect(
    utterances: list[Utterance], seconds: float
) -> tuple[list[np.ndarray], list[Utterance]]:
    """先頭から発話を足し、累計が seconds 以上になるまで読む。発話の途中では切らない。"""
    target = int(seconds * SAMPLE_RATE)
    pieces: list[np.ndarray] = []
    used: list[Utterance] = []
    total = 0
    for utterance in utterances:
        audio = read_wav(utterance.path)
        pieces.append(audio)
        used.append(utterance)
        total += audio.shape[0]
        if total >= target:
            return pieces, used
    raise RuntimeError(
        f"発話を使い切りましたが {seconds:g} 秒に足りません({total / SAMPLE_RATE:.1f}秒)"
    )


def join_with_gaps(
    pieces: list[np.ndarray], gap_seconds: float
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """発話の間に無音を入れてつなぐ。戻り値は (波形, 発話ごとの [開始, 終了) サンプル位置)。"""
    gap = np.zeros(round(gap_seconds * SAMPLE_RATE), dtype=np.float32)
    parts: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    position = 0
    for index, piece in enumerate(pieces):
        if index > 0 and gap.size:
            parts.append(gap)
            position += gap.size
        parts.append(piece)
        spans.append((position, position + piece.shape[0]))
        position += piece.shape[0]
    return np.concatenate(parts), spans


def write_wav(path: Path, audio: np.ndarray) -> None:
    clipped = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    temporary = path.with_suffix(".wav.partial")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(clipped.tobytes())
    temporary.replace(path)


def write_segments(
    path: Path,
    used: list[Utterance],
    spans: list[tuple[int, int]],
    args: argparse.Namespace,
    total_samples: int,
) -> None:
    """発話ごとの位置と長さを、prepare_bench_audio.py と同じ形式の JSON で書く。"""
    data = {
        "sample_rate": SAMPLE_RATE,
        "gap_seconds": args.gap_seconds,
        "duration_sec": round(total_samples / SAMPLE_RATE, 3),
        "source": {"corpus": "jsut_ver1.1", "subsets": args.subsets, "seed": args.seed},
        "utterances": [
            {
                "index": index,
                "id": utterance.id,
                "start_sec": round(start / SAMPLE_RATE, 3),
                "end_sec": round(end / SAMPLE_RATE, 3),
                "duration_sec": round((end - start) / SAMPLE_RATE, 3),
            }
            for index, (utterance, (start, end)) in enumerate(
                zip(used, spans, strict=True), start=1
            )
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.jsut_dir.is_dir():
        print(f"JSUT が見つかりません: {args.jsut_dir}", file=sys.stderr)
        return 1
    utterances: list[Utterance] = []
    for subset in args.subsets:
        utterances.extend(load_subset(args.jsut_dir, subset))
    random.Random(args.seed).shuffle(utterances)
    pieces, used = collect(utterances, args.seconds)
    audio, spans = join_with_gaps(pieces, args.gap_seconds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = OUTPUT_DIR / f"{args.name}.wav"
    text_path = OUTPUT_DIR / f"{args.name}.txt"
    segments_path = OUTPUT_DIR / f"{args.name}.json"
    write_wav(wav_path, audio)
    text_path.write_text("".join(f"{u.text}\n" for u in used), encoding="utf-8")
    write_segments(segments_path, used, spans, args, audio.shape[0])

    gap_note = f", 発話間の無音 {args.gap_seconds:g}秒" if args.gap_seconds else ""
    print(
        "書き出しました。このファイルは Git に含めません。\n"
        f"  {wav_path}  ({audio.shape[0] / SAMPLE_RATE:.1f}秒, {len(used)}発話{gap_note})\n"
        f"  {text_path}\n"
        f"  {segments_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
