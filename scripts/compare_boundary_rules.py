#!/usr/bin/env python3
"""開発用: チャンクの境目での取りこぼしへの対処の候補を比べる(Issue #12)。

重なりの長さごとに、本番と同じ推論条件(constants の値)で各チャンクを1回だけ推論し、
セグメントを --out-dir に保存する。
担当を決める規則は、保存したセグメントに後から当てはめて比べる。
本番のコードは変えない。

規則:
- start: セグメントの開始時刻で担当を決める(現行。設計資料§6.6)
- mid:   セグメントの中央の時刻で担当を決める
- fill:  start に加えて、前のチャンクの担当範囲で始まるために捨てたセグメントのうち、
         前のチャンクが採用したセグメントと時間が重ならないものを補う
- word:  語ごとの開始時刻で担当を決める(--word-timestamps が必要)

使い方:
    uv run scripts/compare_boundary_rules.py scripts/data/long600.wav \\
        --overlaps 3 5 10 --out-dir scripts/data/issue12/rules
    uv run scripts/compare_boundary_rules.py scripts/data/long600.wav \\
        --overlaps 3 --word-timestamps --rules start mid word --out-dir scripts/data/issue12/rules
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_models import align, normalize_for_cer, normalize_reference  # noqa: E402
from eval_boundary import production_spans, score_lines  # noqa: E402
from loopscribe.chunker import ChunkSpan, ownership_bounds  # noqa: E402
from loopscribe.constants import (  # noqa: E402
    BEAM_SIZE,
    CHUNK_SECONDS,
    COMPUTE_TYPE,
    CONDITION_ON_PREVIOUS_TEXT,
    CPU_THREADS,
    LANGUAGE,
    MIN_TAIL_SECONDS,
    MODEL_NAME,
    SAMPLE_RATE,
    VAD_FILTER,
)
from loopscribe.output import AbsoluteSegment, format_line  # noqa: E402

RULES = ("start", "mid", "fill", "word")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    word: str


@dataclass(frozen=True)
class RawSegment:
    """チャンク内の相対時刻を持つセグメント。words は --word-timestamps のときだけ入る。"""

    start: float
    end: float
    text: str
    words: list[Word] | None = field(default=None)


@dataclass
class ChunkResult:
    span: ChunkSpan
    elapsed: float
    segments: list[RawSegment]


def read_wav(path: Path) -> np.ndarray:
    """16bit PCM・16kHz・モノラルの wav を float32 で読む(pipeline と同じ /32768)。"""
    with wave.open(str(path), "rb") as wf:
        if (wf.getsampwidth(), wf.getnchannels(), wf.getframerate()) != (2, 1, SAMPLE_RATE):
            raise ValueError("16bit PCM・モノラル・16kHz の wav を指定してください")
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def make_transcribe(word_timestamps: bool) -> Callable[[np.ndarray], list[RawSegment]]:
    """FasterWhisperTranscriber と同じ条件で推論する関数を作る。"""
    from faster_whisper import WhisperModel

    from loopscribe.model import ensure_model

    model = WhisperModel(
        str(ensure_model(MODEL_NAME)),
        device="cpu",
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
    )

    def transcribe(samples: np.ndarray) -> list[RawSegment]:
        raw_segments, _info = model.transcribe(
            samples,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            word_timestamps=word_timestamps,
        )
        result: list[RawSegment] = []
        for seg in raw_segments:
            text = seg.text.strip()
            if not text:
                continue
            words = (
                [Word(float(w.start), float(w.end), w.word) for w in seg.words]
                if seg.words is not None
                else None
            )
            result.append(RawSegment(float(seg.start), float(seg.end), text, words))
        return result

    return transcribe


def transcribe_chunks(
    audio: np.ndarray,
    spans: list[ChunkSpan],
    transcribe: Callable[[np.ndarray], list[RawSegment]],
) -> list[ChunkResult]:
    results: list[ChunkResult] = []
    for span in spans:
        a = int(round(span.start * SAMPLE_RATE))
        b = min(int(round(span.end * SAMPLE_RATE)), audio.shape[0])
        t0 = time.perf_counter()
        segments = transcribe(audio[a:b])
        elapsed = time.perf_counter() - t0
        results.append(ChunkResult(span=span, elapsed=elapsed, segments=segments))
        print(f"  chunk {span.index:>3} {span.start:>7.1f}-{span.end:<7.1f} {elapsed:5.1f}秒")
    return results


def save_cache(path: Path, results: list[ChunkResult]) -> None:
    data = [
        {
            "span": asdict(r.span),
            "elapsed": r.elapsed,
            "segments": [asdict(s) for s in r.segments],
        }
        for r in results
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cache(path: Path) -> list[ChunkResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results: list[ChunkResult] = []
    for item in data:
        segments = []
        for s in item["segments"]:
            words = s.get("words")
            segments.append(
                RawSegment(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    words=None if words is None else [Word(**w) for w in words],
                )
            )
        results.append(
            ChunkResult(span=ChunkSpan(**item["span"]), elapsed=item["elapsed"], segments=segments)
        )
    return results


def assign(
    results: list[ChunkResult],
    rule: str,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[AbsoluteSegment]:
    """規則に従って、各チャンクのセグメントから採用するものを選ぶ。"""
    last = len(results) - 1
    kept_per_chunk: list[list[AbsoluteSegment]] = []
    for r in results:
        bounds = ownership_bounds(
            r.span.index,
            is_last=r.span.index == last,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
        kept: list[AbsoluteSegment] = []
        rejected_early: list[AbsoluteSegment] = []
        for seg in r.segments:
            if rule == "word":
                if seg.words is None:
                    raise ValueError("word の規則には --word-timestamps が必要です")
                words = [w for w in seg.words if bounds.lo <= r.span.start + w.start < bounds.hi]
                text = "".join(w.word for w in words).strip()
                if text:
                    kept.append(
                        AbsoluteSegment(
                            start=r.span.start + words[0].start,
                            end=r.span.start + words[-1].end,
                            text=text,
                        )
                    )
                continue
            absolute = AbsoluteSegment(
                start=r.span.start + seg.start, end=r.span.start + seg.end, text=seg.text
            )
            key = (absolute.start + absolute.end) / 2 if rule == "mid" else absolute.start
            if bounds.lo <= key < bounds.hi:
                kept.append(absolute)
            elif key < bounds.lo:
                rejected_early.append(absolute)
        if rule == "fill" and kept_per_chunk:
            previous = kept_per_chunk[-1]
            fills = [
                seg
                for seg in rejected_early
                if not any(p.start < seg.end and seg.start < p.end for p in previous)
            ]
            kept = fills + kept
        kept_per_chunk.append(kept)
    return [seg for kept in kept_per_chunk for seg in kept]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="境目での取りこぼしへの対処の候補を比べる")
    p.add_argument("wav", type=Path, help="prepare_bench_audio.py が書く wav")
    p.add_argument("--reference", type=Path, help="正解テキスト。既定は wav と同名の .txt")
    p.add_argument("--overlaps", nargs="+", type=float, default=[3.0, 5.0])
    p.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS)
    p.add_argument("--min-tail-seconds", type=float, default=MIN_TAIL_SECONDS)
    p.add_argument("--rules", nargs="+", choices=RULES, default=["start", "mid", "fill"])
    p.add_argument(
        "--word-timestamps",
        action="store_true",
        help="transcribe(word_timestamps=True) で推論する。word の規則に必要",
    )
    p.add_argument("--drop-threshold", type=float, default=0.5)
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="推論結果のキャッシュと、規則ごとの認識結果(txt)を書く。Git管理外に置くこと",
    )
    p.add_argument("--force", action="store_true", help="キャッシュがあっても推論し直す")
    args = p.parse_args(argv)
    if "word" in args.rules and not args.word_timestamps:
        p.error("word の規則には --word-timestamps が必要です")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audio = read_wav(args.wav)
    duration = audio.shape[0] / SAMPLE_RATE
    reference_path = args.reference or args.wav.with_suffix(".txt")
    ref_lines = normalize_reference(reference_path.read_text(encoding="utf-8"))
    ref_text = "".join(ref_lines)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_words" if args.word_timestamps else ""

    transcribe = None
    rows: list[tuple[str, float, float, float, float, float, int]] = []
    for overlap in args.overlaps:
        spans = production_spans(
            duration,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=overlap,
            min_tail_seconds=args.min_tail_seconds,
        )
        cache = args.out_dir / f"chunks_L{args.chunk_seconds:g}_O{overlap:g}{suffix}.json"
        if cache.is_file() and not args.force:
            results = load_cache(cache)
            print(f"[cache] {cache}")
        else:
            if transcribe is None:
                transcribe = make_transcribe(args.word_timestamps)
                print("[warmup] 先頭チャンクを1回処理(結果は捨てる)", flush=True)
                transcribe(audio[: int(args.chunk_seconds * SAMPLE_RATE)])
            print(f"[run] 重なり {overlap:g}秒 / チャンク {len(spans)}個", flush=True)
            results = transcribe_chunks(audio, spans, transcribe)
            save_cache(cache, results)
        rtf = sum(r.elapsed for r in results) / duration

        for rule in args.rules:
            kept = assign(results, rule, chunk_seconds=args.chunk_seconds, overlap_seconds=overlap)
            label = f"O{overlap:g}{suffix}_{rule}"
            (args.out_dir / f"{label}.txt").write_text(
                "".join(format_line(s.start, s.text) + "\n" for s in kept), encoding="utf-8"
            )
            ops = align(ref_text, normalize_for_cer("".join(s.text for s in kept)))
            line_scores = score_lines(ops, ref_lines)
            ref_chars = sum(s.ref_chars for s in line_scores)
            subs = sum(s.substitutions for s in line_scores)
            dels = sum(s.deletions for s in line_scores)
            ins = sum(s.insertions for s in line_scores)
            dropped = sum(s.deletion_rate >= args.drop_threshold for s in line_scores)
            rows.append(
                (
                    label,
                    rtf,
                    (subs + dels + ins) / ref_chars,
                    subs / ref_chars,
                    dels / ref_chars,
                    ins / ref_chars,
                    dropped,
                )
            )

    print(f"\n== 結果(音声 {duration:.1f}秒 / 発話 {len(ref_lines)}) ==")
    print(
        f"{'条件':<16} {'RTF':>5} {'CER':>6} {'置換':>6} {'削除':>6} {'挿入':>6} {'落ちた発話':>6}"
    )
    for label, rtf, cer, s, d, i, dropped in rows:
        print(f"{label:<16} {rtf:>5.2f} {cer:>6.3f} {s:>6.3f} {d:>6.3f} {i:>6.3f} {dropped:>6}")
    print(
        f"\n注: 落ちた発話 = 削除率 {args.drop_threshold:.0%} 以上の発話。"
        "RTF は推論時間の合計 / 音声の長さ(モデルの読み込みを除く)。"
        f"\n規則ごとの認識結果: {args.out_dir}/O*_*.txt(eval_boundary.py の --hyp に渡せる)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
