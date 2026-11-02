#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "faster-whisper>=1.1.0",
#   # 1.2.1 は av.open(metadata_errors=...) を渡す。PyAV 19 で引数が削除され、
#   # 修正は未リリース(master の PR #1495)なので 19 未満に固定する。
#   "av>=11,<19",
#   "psutil>=6.0",
# ]
# ///
"""モデル検証用スクリプト。

faster-whisper(CPU / int8)で、モデルごとの処理速度(RTF)・ピークメモリ・精度(CER)を測る。
アプリ本体と同じ条件(30秒チャンク + 3秒の重なり + 重複除去)で処理するので、
結果はそのまま本番の見積もりに使える。

使い方 (WSL / Windows どちらでも動く。依存は PEP 723 のメタデータで自動解決される):

    # 最小: 既定の2モデル(kotoba-whisper と small)を比較
    uv run scripts/bench_models.py sample.wav

    # 正解テキストがあれば CER も出す。条件を振る例
    uv run scripts/bench_models.py sample.wav --reference sample.txt \\
        --beam-sizes 1 5 --threads 4 8 --vad on off --runs 3 --json result.json

サンプル音声の目安: 1〜3分程度、16bit PCM の wav、普段文字起こししたい音声に近いもの。
RTF = 処理にかかった時間 / 音声の長さ。1.0を超えると実時間に追いつかない(遅延が累積する)。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

SAMPLE_RATE = 16_000
LANGUAGE = "ja"
DEFAULT_MODELS = ["kotoba-tech/kotoba-whisper-v2.0-faster", "small"]


# ---------------------------------------------------------------- 計算ユーティリティ


def make_chunks(total: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """サンプル数 total を、長さ chunk・重なり overlap のチャンク区間[(start, end)]に分割する。"""
    if chunk <= overlap:
        raise ValueError("chunk は overlap より大きくすること")
    step = chunk - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + chunk, total)
        spans.append((start, end))
        if end >= total:
            break
        start += step
    return spans


def normalize_for_cer(text: str) -> str:
    """CER用の正規化: NFKC・小文字化し、空白と句読点・記号を除く。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if not (ch.isspace() or unicodedata.category(ch)[0] in "PSZ"))


def edit_distance(a: str, b: str) -> int:
    """レーベンシュタイン距離(2行DP)。数千文字程度までを想定。"""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    return edit_distance(ref, hyp) / max(len(ref), 1)


def verdict(rtf: float) -> str:
    """RTFの目安。遅延は無制限に許容する方針なので、RTF>1でも即不採用にはしない。"""
    if rtf <= 0.7:
        return "余裕あり"
    if rtf <= 1.0:
        return "可(遅延は増えない)"
    return f"遅延が累積(録音1時間あたり約{(rtf - 1) * 60:.0f}分遅れ)"


class PeakMemory:
    """with ブロック中の、このプロセスのRSSのピークを一定間隔でサンプリングして記録する。"""

    def __init__(self, interval: float = 0.05) -> None:
        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._proc.memory_info().rss)
            self._stop.wait(self._interval)

    def __enter__(self) -> PeakMemory:
        self._peak = self._proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, self._proc.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self._peak / 1024**2


# ---------------------------------------------------------------- 文字起こし


def transcribe_all(
    model,  # faster_whisper.WhisperModel (テストではスタブ)
    audio,  # float32 / 16kHz / モノラル の配列
    spans: list[tuple[int, int]],
    *,
    beam_size: int,
    vad: bool,
    overlap_sec: float,
) -> tuple[float, str]:
    """全チャンクを順に文字起こしし、(推論にかかった合計秒, 重複除去後の本文) を返す。

    重複除去: 各セグメントを「開始時刻(録音開始からの絶対秒)が属する担当チャンク」だけが採用する。
    チャンクkの担当範囲は [s_k + O/2, s_{k+1} + O/2)(先頭チャンクの下限は0、最終チャンクの上限は無限大)。
    """
    elapsed = 0.0
    kept: list[str] = []
    last = len(spans) - 1
    for k, (a, b) in enumerate(spans):
        lo = 0.0 if k == 0 else spans[k][0] / SAMPLE_RATE + overlap_sec / 2
        hi = float("inf") if k == last else spans[k + 1][0] / SAMPLE_RATE + overlap_sec / 2
        t0 = time.perf_counter()
        segments, _info = model.transcribe(
            audio[a:b],
            language=LANGUAGE,
            beam_size=beam_size,
            vad_filter=vad,
            condition_on_previous_text=False,
        )
        segs = list(segments)  # ジェネレータを消費するまでが推論時間
        elapsed += time.perf_counter() - t0
        offset = a / SAMPLE_RATE
        for seg in segs:
            if lo <= offset + seg.start < hi:
                kept.append(seg.text.strip())
    return elapsed, "".join(kept)


@dataclass
class Result:
    model: str
    cpu_threads: int
    beam_size: int
    vad: bool
    load_sec: float
    audio_sec: float
    run_secs: list[float]
    rtf_mean: float
    rtf_min: float
    rtf_max: float
    peak_rss_mb: float
    cer: float | None
    verdict: str
    text_head: str


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_threads = psutil.cpu_count(logical=False) or 4
    p = argparse.ArgumentParser(description="faster-whisper のCPU性能(RTF/メモリ/CER)を測る")
    p.add_argument("audio", help="サンプル音声(wav など)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="モデル名またはHFのrepo id")
    p.add_argument("--threads", nargs="+", type=int, default=[default_threads], help="cpu_threads")
    p.add_argument("--beam-sizes", nargs="+", type=int, default=[1], help="beam_size")
    p.add_argument("--vad", nargs="+", choices=["on", "off"], default=["on"], help="VADの有無")
    p.add_argument("--compute-type", default="int8", help="既定: int8")
    p.add_argument("--chunk-seconds", type=int, default=30)
    p.add_argument("--overlap-seconds", type=int, default=3)
    p.add_argument("--runs", type=int, default=3, help="条件ごとの計測回数(ウォームアップ除く)")
    p.add_argument("--seconds", type=float, help="音声の先頭N秒だけを使う")
    p.add_argument("--reference", help="正解テキスト(UTF-8)。指定するとCERを計算する")
    p.add_argument("--json", help="結果をJSONで保存するパス")
    p.add_argument("--show-text", action="store_true", help="認識結果の先頭を表示する")
    return p.parse_args(argv)


def print_environment() -> None:
    import ctranslate2
    import faster_whisper

    print("== 環境 ==")
    print(f"python        : {sys.version.split()[0]} / {platform.platform()}")
    print(
        f"cpu           : 物理{psutil.cpu_count(logical=False)}コア / 論理{os.cpu_count()}スレッド"
    )
    print(f"ram           : {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"faster-whisper: {getattr(faster_whisper, '__version__', '?')}")
    print(f"ctranslate2   : {getattr(ctranslate2, '__version__', '?')}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # 重い import は引数チェックの後に行う(--help を速くするため)
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    print_environment()

    audio = decode_audio(args.audio, sampling_rate=SAMPLE_RATE)
    if args.seconds:
        audio = audio[: int(args.seconds * SAMPLE_RATE)]
    audio_sec = len(audio) / SAMPLE_RATE
    spans = make_chunks(
        len(audio), args.chunk_seconds * SAMPLE_RATE, args.overlap_seconds * SAMPLE_RATE
    )
    reference = Path(args.reference).read_text(encoding="utf-8") if args.reference else None
    print(
        f"音声: {audio_sec:.1f}秒 / チャンク: {len(spans)}個"
        f"({args.chunk_seconds}秒、重なり{args.overlap_seconds}秒)\n"
    )

    results: list[Result] = []
    for model_name in args.models:
        for threads in args.threads:
            print(
                f"[load] {model_name} (cpu_threads={threads}, {args.compute_type}) ...", flush=True
            )
            t0 = time.perf_counter()
            model = WhisperModel(
                model_name, device="cpu", compute_type=args.compute_type, cpu_threads=threads
            )
            load_sec = time.perf_counter() - t0
            print(f"[load] 完了 {load_sec:.1f}秒", flush=True)

            warmed = False
            for beam in args.beam_sizes:
                for vad_flag in args.vad:
                    vad = vad_flag == "on"
                    if not warmed:
                        print("[warmup] 先頭チャンクを1回処理(結果は捨てる)", flush=True)
                        a, b = spans[0]
                        list(
                            model.transcribe(
                                audio[a:b],
                                language=LANGUAGE,
                                beam_size=beam,
                                vad_filter=vad,
                                condition_on_previous_text=False,
                            )[0]
                        )
                        warmed = True

                    label = f"beam={beam} vad={vad_flag}"
                    print(f"[run] {label} x{args.runs}", flush=True)
                    run_secs: list[float] = []
                    text = ""
                    with PeakMemory() as mem:
                        for i in range(args.runs):
                            elapsed, text = transcribe_all(
                                model,
                                audio,
                                spans,
                                beam_size=beam,
                                vad=vad,
                                overlap_sec=args.overlap_seconds,
                            )
                            run_secs.append(elapsed)
                            print(
                                f"       #{i + 1}: {elapsed:.1f}秒 (RTF {elapsed / audio_sec:.2f})",
                                flush=True,
                            )

                    rtfs = [s / audio_sec for s in run_secs]
                    rtf_mean = sum(rtfs) / len(rtfs)
                    results.append(
                        Result(
                            model=model_name,
                            cpu_threads=threads,
                            beam_size=beam,
                            vad=vad,
                            load_sec=load_sec,
                            audio_sec=audio_sec,
                            run_secs=run_secs,
                            rtf_mean=rtf_mean,
                            rtf_min=min(rtfs),
                            rtf_max=max(rtfs),
                            peak_rss_mb=mem.peak_mb,
                            cer=cer(reference, text) if reference else None,
                            verdict=verdict(rtf_mean),
                            text_head=text[:120],
                        )
                    )
            del model
            gc.collect()
            print()

    print("== 結果 ==")
    header = f"{'model':<42} {'thr':>3} {'beam':>4} {'vad':>3} {'load':>6} {'RTF(平均)':>9} {'min-max':>11} {'RSS(MB)':>8} {'CER':>6}  判定"
    print(header)
    for r in results:
        cer_s = f"{r.cer:.3f}" if r.cer is not None else "-"
        print(
            f"{r.model:<42} {r.cpu_threads:>3} {r.beam_size:>4} {'on' if r.vad else 'off':>3} "
            f"{r.load_sec:>5.1f}s {r.rtf_mean:>9.2f} {r.rtf_min:>5.2f}-{r.rtf_max:<5.2f} "
            f"{r.peak_rss_mb:>8.0f} {cer_s:>6}  {r.verdict}"
        )
        if args.show_text:
            print(f"    本文(先頭): {r.text_head}")
    print(
        "\n注: RSSはモデル常駐分を含むプロセス全体の概算。正確に比べるにはモデルごとに別プロセスで実行すること。"
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSONを保存しました: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
