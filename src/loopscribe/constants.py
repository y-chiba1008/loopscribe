"""モデルに依存しない設定値の定数。CLIオプションにはせず、ここに集約する。

推論パラメータはモデルごとに `loopscribe.models` の各ファイルで定義する。
"""

import os

# 物理コア数の目安(論理コア数の半分)。論理コア数ぶん使うより速い
CPU_THREADS = max(1, (os.cpu_count() or 2) // 2)
LANGUAGE = "ja"

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30.0
OVERLAP_SECONDS = 3.0
MIN_TAIL_SECONDS = 1.0

# WASAPI ループバックのコールバックバッファ長(フレーム数)
WASAPI_FRAMES_PER_BUFFER = 512
# キュー待ちのタイムアウト(ストリーム停止検知用)
WASAPI_QUEUE_POLL_SECONDS = 0.2

LAG_WARN_SECONDS = 600.0
BUFFER_WARN_MB = 2048.0
UI_REFRESH_SECONDS = 0.5
SIGNAL_POLL_SECONDS = 0.2

# `-n/--output-name` 未指定時の出力名(拡張子なし)。拡張子はコマンドが付ける
OUTPUT_NAME_FORMAT = "loopscribe_%Y%m%d_%H%M%S"
TRANSCRIPT_SUFFIX = ".txt"
# `-n/--output-name` の末尾にあれば取り除く拡張子
OUTPUT_NAME_STRIP_SUFFIXES = (".txt", ".wav")
OUTPUT_ENCODING = "utf-8"
