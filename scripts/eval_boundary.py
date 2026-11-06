#!/usr/bin/env python3
"""開発用: チャンクの境目での発話の取りこぼしを、正解の発話単位で集計する(Issue #12)。

prepare_bench_audio.py が書く <name>.json(発話ごとの開始・終了秒)と <name>.txt(正解)を使い、
認識結果ごとに、発話ごとの削除率を求める。各発話について、本番と同じ規則で担当チャンクを決め、
担当チャンクの末尾で発話が切れているか(取りこぼしの起きやすい位置か)で分けて集計する。

認識結果は次のどちらでもよい。
- loopscribe の txt(先頭のヘッダ行と、行頭の [HH:MM:SS] は除いて読む)
- bench_models.py --save-text の *.hyp.txt(1行1セグメント)

使い方:
    uv run scripts/eval_boundary.py scripts/data/long600.json \\
        --hyp prod=scripts/data/issue12/prod/loopscribe_....txt \\
        --hyp single=scripts/data/issue12/single/small_thr4_beam1_vadon_wclnone.hyp.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_models import Op, align, normalize_for_cer, normalize_reference  # noqa: E402
from loopscribe.chunker import ChunkSpan, chunk_span, ownership_bounds  # noqa: E402
from loopscribe.constants import CHUNK_SECONDS, MIN_TAIL_SECONDS, OVERLAP_SECONDS  # noqa: E402

TIMESTAMP = re.compile(r"^\[\d+:\d{2}:\d{2}\]\s*")
HEADER_PREFIX = "# loopscribe"


@dataclass(frozen=True)
class Utterance:
    index: int
    start: float
    end: float
    text: str  # 正規化後


@dataclass(frozen=True)
class Placement:
    """発話と担当チャンクの位置関係。"""

    owner: int
    owner_end: float
    in_overlap: bool  # 開始時刻が、いずれかの重なり区間に入る
    truncated: bool  # 担当チャンクの末尾より後まで続く


@dataclass
class LineScore:
    ref_chars: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def deletion_rate(self) -> float:
        return self.deletions / max(self.ref_chars, 1)


def production_spans(
    duration: float,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
    min_tail_seconds: float,
) -> list[ChunkSpan]:
    """pipeline と同じ規則で、録音全体のチャンク区間を並べる(停止時の端数を含む)。"""
    spans: list[ChunkSpan] = []
    index = 0
    while True:
        span = chunk_span(index, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds)
        if span.end > duration:
            break
        spans.append(span)
        index += 1
    tail_start = chunk_span(
        index, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds
    ).start
    if duration - tail_start >= min_tail_seconds:
        spans.append(ChunkSpan(index=index, start=tail_start, end=duration))
    return spans


def place(
    utterance: Utterance,
    spans: list[ChunkSpan],
    *,
    chunk_seconds: float,
    overlap_seconds: float,
) -> Placement:
    last = len(spans) - 1
    owner = last
    for span in spans:
        bounds = ownership_bounds(
            span.index,
            is_last=span.index == last,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
        if bounds.lo <= utterance.start < bounds.hi:
            owner = span.index
            break
    in_overlap = any(
        spans[k + 1].start <= utterance.start < spans[k].end for k in range(len(spans) - 1)
    )
    owner_end = spans[owner].end
    return Placement(
        owner=owner,
        owner_end=owner_end,
        in_overlap=in_overlap,
        truncated=utterance.end > owner_end,
    )


def read_hypothesis(path: Path) -> str:
    """認識結果の本文をつなげる。loopscribe のヘッダ行とタイムスタンプは除く。"""
    parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(HEADER_PREFIX):
            continue
        parts.append(TIMESTAMP.sub("", line))
    return "".join(parts)


def score_lines(ops: list[Op], ref_lines: list[str]) -> list[LineScore]:
    """編集操作を正解の発話ごとに数える。発話の境目にある挿入は、前の発話に含める
    (bench_models.render_diff と同じ扱い)。"""
    ends: list[int] = []
    pos = 0
    for line in ref_lines:
        pos += len(line)
        ends.append(pos)
    scores = [LineScore() for _ in ref_lines]
    line_index = 0
    ref_pos = 0
    for op, _r, _h in ops:
        if op != "I":
            while line_index < len(ends) - 1 and ref_pos >= ends[line_index]:
                line_index += 1
            ref_pos += 1
            scores[line_index].ref_chars += 1
        if op == "S":
            scores[line_index].substitutions += 1
        elif op == "D":
            scores[line_index].deletions += 1
        elif op == "I":
            scores[line_index].insertions += 1
    return scores


def parse_hyp(value: str) -> tuple[str, Path]:
    label, sep, path = value.partition("=")
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError("LABEL=PATH の形で指定してください")
    return label, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="チャンクの境目での発話の取りこぼしを集計する")
    p.add_argument("segments", type=Path, help="prepare_bench_audio.py が書く <name>.json")
    p.add_argument("--reference", type=Path, help="正解テキスト。既定は segments と同名の .txt")
    p.add_argument(
        "--hyp",
        type=parse_hyp,
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="認識結果。複数指定で比べる",
    )
    p.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS)
    p.add_argument("--overlap-seconds", type=float, default=OVERLAP_SECONDS)
    p.add_argument("--min-tail-seconds", type=float, default=MIN_TAIL_SECONDS)
    p.add_argument(
        "--drop-threshold",
        type=float,
        default=0.5,
        help="削除率がこの値以上の発話を「落ちた」とみなす。既定は 0.5",
    )
    p.add_argument("--json", type=Path, help="発話ごとの結果をJSONで保存するパス")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    meta = json.loads(args.segments.read_text(encoding="utf-8"))
    reference_path = args.reference or args.segments.with_suffix(".txt")
    ref_lines = normalize_reference(reference_path.read_text(encoding="utf-8"))
    raw_utterances = meta["utterances"]
    if len(ref_lines) != len(raw_utterances):
        print(
            f"正解の行数({len(ref_lines)})と発話数({len(raw_utterances)})が一致しません。",
            file=sys.stderr,
        )
        return 2
    utterances = [
        Utterance(index=u["index"], start=u["start_sec"], end=u["end_sec"], text=text)
        for u, text in zip(raw_utterances, ref_lines, strict=True)
    ]
    duration = float(meta["duration_sec"])
    spans = production_spans(
        duration,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        min_tail_seconds=args.min_tail_seconds,
    )
    placements = [
        place(u, spans, chunk_seconds=args.chunk_seconds, overlap_seconds=args.overlap_seconds)
        for u in utterances
    ]

    labels = [label for label, _ in args.hyp]
    scores: dict[str, list[LineScore]] = {}
    for label, path in args.hyp:
        ops = align("".join(ref_lines), normalize_for_cer(read_hypothesis(path)))
        scores[label] = score_lines(ops, ref_lines)

    print(
        f"音声: {duration:.1f}秒 / 発話: {len(utterances)} / チャンク: {len(spans)}個"
        f"({args.chunk_seconds:g}秒、重なり{args.overlap_seconds:g}秒)"
    )
    print(
        f"「落ちた」= 削除率 {args.drop_threshold:.0%} 以上。"
        "「末尾で切れる」= 担当チャンクの終端より後まで続く発話\n"
    )

    print("== 全体 ==")
    for label in labels:
        line_scores = scores[label]
        ref_chars = sum(s.ref_chars for s in line_scores)
        errors = sum(s.substitutions + s.deletions + s.insertions for s in line_scores)
        deletions = sum(s.deletions for s in line_scores)
        dropped = sum(s.deletion_rate >= args.drop_threshold for s in line_scores)
        print(
            f"{label:<10} CER {errors / max(ref_chars, 1):.3f}"
            f"  削除率 {deletions / max(ref_chars, 1):.3f}  落ちた発話 {dropped}/{len(line_scores)}"
        )

    groups = {
        "末尾で切れる": [p.truncated for p in placements],
        "重なり区間で開始": [p.in_overlap for p in placements],
        "それ以外": [not (p.truncated or p.in_overlap) for p in placements],
    }
    print("\n== 位置ごと(落ちた発話 / 発話数、削除文字 / 正解文字) ==")
    for name, mask in groups.items():
        members = [i for i, flag in enumerate(mask) if flag]
        cells = []
        for label in labels:
            line_scores = [scores[label][i] for i in members]
            dropped = sum(s.deletion_rate >= args.drop_threshold for s in line_scores)
            deletions = sum(s.deletions for s in line_scores)
            ref_chars = sum(s.ref_chars for s in line_scores)
            cells.append(f"{label}: {dropped}/{len(members)}, {deletions}/{ref_chars}")
        print(f"{name:<12} " + "  |  ".join(cells))

    print("\n== 落ちた発話(いずれかの認識結果で) ==")
    print(
        f"{'#':>3} {'開始':>7} {'終了':>7} {'担当':>4} {'担当終端':>8} {'位置':<14} "
        + " ".join(f"{label + '削除':>12}" for label in labels)
    )
    rows = []
    for i, (u, p) in enumerate(zip(utterances, placements, strict=True)):
        rates = {label: scores[label][i] for label in labels}
        flagged = any(s.deletion_rate >= args.drop_threshold for s in rates.values())
        position = "末尾で切れる" if p.truncated else ("重なりで開始" if p.in_overlap else "-")
        if flagged:
            print(
                f"{u.index:>3} {u.start:>7.1f} {u.end:>7.1f} {p.owner:>4} {p.owner_end:>8.1f} "
                f"{position:<14} "
                + " ".join(
                    f"{s.deletions:>4}/{s.ref_chars:<3}({s.deletion_rate:>4.0%})"
                    for s in rates.values()
                )
            )
        rows.append(
            {
                "index": u.index,
                "start_sec": u.start,
                "end_sec": u.end,
                "owner_chunk": p.owner,
                "owner_end_sec": p.owner_end,
                "in_overlap": p.in_overlap,
                "truncated": p.truncated,
                "scores": {
                    label: {
                        "ref_chars": s.ref_chars,
                        "substitutions": s.substitutions,
                        "deletions": s.deletions,
                        "insertions": s.insertions,
                    }
                    for label, s in rates.items()
                },
            }
        )

    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSONを保存しました: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
