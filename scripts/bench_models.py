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

    # 誤りの内訳を調べる例: 差分を表示し、全文を保存する。
    # transcribe(chunk_length=...) は未指定(none)と15秒を比べる
    uv run scripts/bench_models.py sample.wav --reference sample.txt \\
        --whisper-chunk-length none 15 --show-diff --save-text scripts/data/results

サンプル音声の目安: 1〜3分程度、16bit PCM の wav、普段文字起こししたい音声に近いもの。
RTF = 処理にかかった時間 / 音声の長さ。1.0を超えると実時間に追いつかない(遅延が累積する)。
正解テキストは1行1発話。CERの内訳(置換・削除・挿入)と差分は、この行単位で表示する。
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


def normalize_reference(reference: str) -> list[str]:
    """正解テキストを、1行1発話として行ごとに正規化する(空行は除く)。"""
    lines = (normalize_for_cer(line) for line in reference.splitlines())
    return [line for line in lines if line]


# 編集操作: "=" 一致、"S" 置換、"D" 削除(正解にあるが認識にない)、"I" 挿入(認識にだけある)
Op = tuple[str, str, str]  # (操作, 正解の文字, 認識の文字)


def align(ref: str, hyp: str) -> list[Op]:
    """レーベンシュタイン距離のDP表をバックトレースし、編集操作の列を返す。

    DP表を丸ごと持つので、メモリも O(len(ref) × len(hyp))。数千文字程度までを想定。
    """
    n, m = len(ref), len(hyp)
    dp = [list(range(m + 1))]
    for i in range(1, n + 1):
        prev = dp[-1]
        cur = [i] + [0] * m
        r = ref[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != hyp[j - 1]))
        dp.append(cur)

    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


@dataclass
class Score:
    """CERの内訳。S + D + I がレーベンシュタイン距離に等しい。"""

    ref_chars: int
    hyp_chars: int
    substitutions: int
    deletions: int
    insertions: int

    @classmethod
    def from_ops(cls, ops: list[Op]) -> Score:
        count = {"=": 0, "S": 0, "D": 0, "I": 0}
        for op, _r, _h in ops:
            count[op] += 1
        return cls(
            ref_chars=count["="] + count["S"] + count["D"],
            hyp_chars=count["="] + count["S"] + count["I"],
            substitutions=count["S"],
            deletions=count["D"],
            insertions=count["I"],
        )

    def rate(self, n: int) -> float:
        return n / max(self.ref_chars, 1)

    @property
    def cer(self) -> float:
        return self.rate(self.substitutions + self.deletions + self.insertions)

    @property
    def length_ratio(self) -> float:
        """認識 / 正解 の文字数比。0.8未満なら削除が多い、1.2超なら挿入が多い目安。"""
        return self.hyp_chars / max(self.ref_chars, 1)


def render_diff(ops: list[Op], ref_lines: list[str]) -> list[str]:
    """編集操作の列を、正解の発話ごとに1行の差分にする。

    表記: 一致はそのまま、[正解→認識] 置換、[-正解] 削除、[+認識] 挿入。
    連続する誤りは1つの括弧にまとめる。各行の先頭に、発話番号とその発話のS/D/Iを付ける。
    発話の境目にある挿入は、前の発話に含める。
    """
    ends: list[int] = []
    pos = 0
    for line in ref_lines:
        pos += len(line)
        ends.append(pos)

    rendered: list[str] = []
    body: list[str] = []
    pending_ref: list[str] = []
    pending_hyp: list[str] = []
    counts = {"S": 0, "D": 0, "I": 0}
    line_index = 0
    ref_pos = 0

    def flush_pending() -> None:
        r, h = "".join(pending_ref), "".join(pending_hyp)
        if r and h:
            body.append(f"[{r}→{h}]")
        elif r:
            body.append(f"[-{r}]")
        elif h:
            body.append(f"[+{h}]")
        pending_ref.clear()
        pending_hyp.clear()

    def close_line() -> None:
        flush_pending()
        rendered.append(
            f"{line_index + 1:>3} [S{counts['S']} D{counts['D']} I{counts['I']}] {''.join(body)}"
        )
        body.clear()
        for key in counts:
            counts[key] = 0

    for op, r, h in ops:
        if op != "I":
            while line_index < len(ends) - 1 and ref_pos >= ends[line_index]:
                close_line()
                line_index += 1
            ref_pos += 1
        if op == "=":
            flush_pending()
            body.append(r)
            continue
        counts[op] += 1
        pending_ref.append(r)
        pending_hyp.append(h)
    close_line()
    return rendered


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
    whisper_chunk_length: int | None = None,
) -> tuple[float, list[str]]:
    """全チャンクを順に文字起こしし、(推論にかかった合計秒, 重複除去後のセグメント本文) を返す。

    whisper_chunk_length は transcribe(chunk_length=...) に渡す。None なら faster-whisper の既定。

    重複除去: 各セグメントを「開始時刻(録音開始からの絶対秒)が属する担当チャンク」だけが採用する。
    チャンクkの担当範囲は [s_k + O/2, s_{k+1} + O/2)
    (先頭チャンクの下限は0、最終チャンクの上限は無限大)。
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
            chunk_length=whisper_chunk_length,
        )
        segs = list(segments)  # ジェネレータを消費するまでが推論時間
        elapsed += time.perf_counter() - t0
        offset = a / SAMPLE_RATE
        for seg in segs:
            if lo <= offset + seg.start < hi:
                kept.append(seg.text.strip())
    return elapsed, kept


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
    whisper_chunk_length: int | None = None
    ref_chars: int | None = None
    hyp_chars: int | None = None
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None
    substitution_rate: float | None = None
    deletion_rate: float | None = None
    insertion_rate: float | None = None
    length_ratio: float | None = None

    @property
    def label(self) -> str:
        wcl = "none" if self.whisper_chunk_length is None else self.whisper_chunk_length
        return (
            f"{self.model} thr={self.cpu_threads} beam={self.beam_size} "
            f"vad={'on' if self.vad else 'off'} wcl={wcl}"
        )

    @property
    def file_stem(self) -> str:
        wcl = "none" if self.whisper_chunk_length is None else self.whisper_chunk_length
        model = self.model.replace("/", "__")
        return (
            f"{model}_thr{self.cpu_threads}_beam{self.beam_size}"
            f"_vad{'on' if self.vad else 'off'}_wcl{wcl}"
        )


def save_texts(
    directory: Path,
    result: Result,
    segments: list[str],
    hyp_norm: str,
    diff_lines: list[str] | None,
) -> None:
    """条件ごとに、認識結果の全文(1行1セグメント)・正規化後の認識・差分を保存する。"""
    stem = result.file_stem
    (directory / f"{stem}.hyp.txt").write_text("\n".join(segments) + "\n", encoding="utf-8")
    (directory / f"{stem}.hyp.norm.txt").write_text(hyp_norm + "\n", encoding="utf-8")
    if diff_lines is not None:
        (directory / f"{stem}.diff.txt").write_text("\n".join(diff_lines) + "\n", encoding="utf-8")


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
    p.add_argument(
        "--whisper-chunk-length",
        nargs="+",
        type=parse_whisper_chunk_length,
        default=[None],
        metavar="N",
        help="transcribe(chunk_length=N) に渡す秒数。none は未指定(既定)。複数指定で比較する",
    )
    p.add_argument("--json", help="結果をJSONで保存するパス")
    p.add_argument("--show-text", action="store_true", help="認識結果の先頭を表示する")
    p.add_argument(
        "--show-diff",
        action="store_true",
        help="正規化後の正解と認識の差分を、発話ごとに表示する(--reference が必要)",
    )
    p.add_argument(
        "--save-text",
        metavar="DIR",
        help="条件ごとに認識の全文・正規化後の正解と認識・差分を保存する"
        "(Git管理外の scripts/data/ の下を推奨)",
    )
    args = p.parse_args(argv)
    if args.show_diff and not args.reference:
        p.error("--show-diff には --reference が必要です")
    return args


def parse_whisper_chunk_length(value: str) -> int | None:
    if value.lower() == "none":
        return None
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("正の整数か none を指定してください")
    return n


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
    ref_lines = (
        normalize_reference(Path(args.reference).read_text(encoding="utf-8"))
        if args.reference
        else None
    )
    save_dir = Path(args.save_text) if args.save_text else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        if ref_lines is not None:
            (save_dir / "reference.norm.txt").write_text(
                "\n".join(ref_lines) + "\n", encoding="utf-8"
            )
    print(
        f"音声: {audio_sec:.1f}秒 / チャンク: {len(spans)}個"
        f"({args.chunk_seconds}秒、重なり{args.overlap_seconds}秒)\n"
    )

    results: list[Result] = []
    diffs: list[list[str] | None] = []
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
                    for wcl in args.whisper_chunk_length:
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
                                    chunk_length=wcl,
                                )[0]
                            )
                            warmed = True

                        label = f"beam={beam} vad={vad_flag} wcl={'none' if wcl is None else wcl}"
                        print(f"[run] {label} x{args.runs}", flush=True)
                        run_secs: list[float] = []
                        segments: list[str] = []
                        with PeakMemory() as mem:
                            for i in range(args.runs):
                                elapsed, segments = transcribe_all(
                                    model,
                                    audio,
                                    spans,
                                    beam_size=beam,
                                    vad=vad,
                                    overlap_sec=args.overlap_seconds,
                                    whisper_chunk_length=wcl,
                                )
                                run_secs.append(elapsed)
                                print(
                                    f"       #{i + 1}: {elapsed:.1f}秒"
                                    f" (RTF {elapsed / audio_sec:.2f})",
                                    flush=True,
                                )

                        text = "".join(segments)
                        hyp_norm = normalize_for_cer(text)
                        rtfs = [s / audio_sec for s in run_secs]
                        rtf_mean = sum(rtfs) / len(rtfs)
                        result = Result(
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
                            cer=None,
                            verdict=verdict(rtf_mean),
                            text_head=text[:120],
                            whisper_chunk_length=wcl,
                        )
                        diff_lines: list[str] | None = None
                        if ref_lines is not None:
                            ops = align("".join(ref_lines), hyp_norm)
                            score = Score.from_ops(ops)
                            result.cer = score.cer
                            result.ref_chars = score.ref_chars
                            result.hyp_chars = score.hyp_chars
                            result.substitutions = score.substitutions
                            result.deletions = score.deletions
                            result.insertions = score.insertions
                            result.substitution_rate = score.rate(score.substitutions)
                            result.deletion_rate = score.rate(score.deletions)
                            result.insertion_rate = score.rate(score.insertions)
                            result.length_ratio = score.length_ratio
                            diff_lines = render_diff(ops, ref_lines)
                        if save_dir is not None:
                            save_texts(save_dir, result, segments, hyp_norm, diff_lines)
                        results.append(result)
                        diffs.append(diff_lines)
            del model
            gc.collect()
            print()

    print("== 結果 ==")
    header = (
        f"{'model':<42} {'thr':>3} {'beam':>4} {'vad':>3} {'wcl':>4} {'load':>6} "
        f"{'RTF(平均)':>9} {'min-max':>11} {'RSS(MB)':>8} {'CER':>6} {'長さ比':>5}  判定"
    )
    print(header)
    for r in results:
        cer_s = f"{r.cer:.3f}" if r.cer is not None else "-"
        ratio_s = f"{r.length_ratio:.2f}" if r.length_ratio is not None else "-"
        wcl_s = "none" if r.whisper_chunk_length is None else str(r.whisper_chunk_length)
        print(
            f"{r.model:<42} {r.cpu_threads:>3} {r.beam_size:>4} {'on' if r.vad else 'off':>3} "
            f"{wcl_s:>4} {r.load_sec:>5.1f}s {r.rtf_mean:>9.2f} "
            f"{r.rtf_min:>5.2f}-{r.rtf_max:<5.2f} {r.peak_rss_mb:>8.0f} {cer_s:>6} "
            f"{ratio_s:>8}  {r.verdict}"
        )
        if r.cer is not None:
            print(
                f"    誤り: 置換 {r.substitutions} ({r.substitution_rate:.1%})"
                f" / 削除 {r.deletions} ({r.deletion_rate:.1%})"
                f" / 挿入 {r.insertions} ({r.insertion_rate:.1%})"
                f"  文字数: 正解 {r.ref_chars} / 認識 {r.hyp_chars}"
            )
        if args.show_text:
            print(f"    本文(先頭): {r.text_head}")
    print(
        "\n注: RSSはモデル常駐分を含むプロセス全体の概算。"
        "正確に比べるにはモデルごとに別プロセスで実行すること。"
    )
    if ref_lines is not None:
        print(
            "注: 削除は正解にあって認識にない文字(取りこぼし)、挿入は認識にだけある文字(幻覚など)。"
            "長さ比 = 認識 / 正解 の文字数。"
        )

    if args.show_diff:
        print("\n== 差分 ==")
        print("表記: [正解→認識] 置換、[-正解] 削除、[+認識] 挿入。行頭は発話番号とその発話のS/D/I")
        for r, diff_lines in zip(results, diffs, strict=True):
            print(f"\n-- {r.label} --")
            for line in diff_lines or []:
                print(line)

    if save_dir is not None:
        print(f"\n認識結果を保存しました: {save_dir}(Gitに含めないこと)")
    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSONを保存しました: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
