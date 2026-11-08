"""kotoba-whisper: チャンク内をVADで区切り、15秒以下のクリップごとに推論する(設計資料§6.8)。

固定長のウィンドウで発話の途中を切ると、kotoba-whisper はその発話を飛ばしやすい。
BatchedInferencePipeline は、VADで無音の位置を探して chunk_length 秒以下のクリップにまとめるので、
発話の途中で切らない(Issue #29)。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel

from loopscribe.constants import CPU_THREADS, LANGUAGE
from loopscribe.errors import ModelUnavailableError
from loopscribe.models.base import ModelSpec
from loopscribe.transcriber import Segment, as_mono_float32, to_segments

REPO_ID = "kotoba-tech/kotoba-whisper-v2.0-faster"
COMPUTE_TYPE = "int8"
# beam=5 はわずかに悪化する(Issue #29)
BEAM_SIZE = 1
# クリップの上限秒数。15秒を超えると精度が急に落ちる
CLIP_SECONDS = 15
# CPUでは大きくしても速くならない
BATCH_SIZE = 1
# 時刻トークンなしでは、クリップの最後の発話を出さずに終わることがある
WITHOUT_TIMESTAMPS = False
CONDITION_ON_PREVIOUS_TEXT = False

# 配布の config.json には large-v3 の値(デコーダーの7〜25層)が入っているが、
# デコーダーは2層しかなく、word_timestamps=True で ctranslate2 が異常終了する。
# 最後の層(1層)の全20ヘッドを使う
ALIGNMENT_HEADS: list[list[int]] = [[1, head] for head in range(20)]


def fix_alignment_heads(
    model_dir: Path,
    *,
    alignment_heads: list[list[int]] = ALIGNMENT_HEADS,
) -> bool:
    """model_dir/config.json の alignment_heads を書き換える。書き換えたら True。

    HFキャッシュの config.json は blobs/ へのシンボリックリンクのことがある。
    一時ファイルに書いて os.replace で置き換え、リンク先の実体には触らない。
    """
    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("alignment_heads") == alignment_heads:
            return False
        config["alignment_heads"] = alignment_heads
        mode = config_path.stat().st_mode & 0o777
        fd, tmp_name = tempfile.mkstemp(dir=model_dir, prefix=".config.", suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            # mkstemp は 0600 で作るので、元のファイルの権限に戻す
            os.chmod(tmp_name, mode)
            os.replace(tmp_name, config_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    except (OSError, ValueError) as exc:
        raise ModelUnavailableError(
            f"モデルの設定({config_path})を修正できませんでした: {exc}"
        ) from exc
    return True


class KotobaTranscriber:
    """kotoba-whisper の Transcriber 実装。"""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        cpu_threads: int = CPU_THREADS,
        compute_type: str = COMPUTE_TYPE,
        language: str = LANGUAGE,
        beam_size: int = BEAM_SIZE,
        clip_seconds: int = CLIP_SECONDS,
        batch_size: int = BATCH_SIZE,
        without_timestamps: bool = WITHOUT_TIMESTAMPS,
        condition_on_previous_text: bool = CONDITION_ON_PREVIOUS_TEXT,
    ) -> None:
        self._language = language
        self._beam_size = beam_size
        self._clip_seconds = clip_seconds
        self._batch_size = batch_size
        self._without_timestamps = without_timestamps
        self._condition_on_previous_text = condition_on_previous_text
        model = WhisperModel(
            str(model_name_or_path),
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        self._pipeline = BatchedInferencePipeline(model)

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        raw_segments, _info = self._pipeline.transcribe(
            as_mono_float32(samples),
            language=self._language,
            beam_size=self._beam_size,
            # VADでクリップに区切るので、オフにしない
            vad_filter=True,
            condition_on_previous_text=self._condition_on_previous_text,
            chunk_length=self._clip_seconds,
            batch_size=self._batch_size,
            without_timestamps=self._without_timestamps,
            word_timestamps=True,
        )
        return to_segments(raw_segments)


class KotobaModel(ModelSpec):
    name = "kotoba"
    repo_id = REPO_ID

    def fix_files(self, model_dir: Path) -> None:
        fix_alignment_heads(model_dir)

    def load(self, model_dir: Path, *, cpu_threads: int = CPU_THREADS) -> KotobaTranscriber:
        return KotobaTranscriber(model_dir, cpu_threads=cpu_threads)
