"""モデルごとの取得・修正・読み込みの共通インターフェース。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from loopscribe.constants import CPU_THREADS
from loopscribe.model import download_model, ensure_model, is_model_available
from loopscribe.transcriber import Transcriber


class ModelSpec(ABC):
    """`--model` で選べるモデル。推論パラメータは実装ごとに固定値で持つ。"""

    # `--model` に渡す名前
    name: ClassVar[str]
    # faster-whisper のサイズ名、または Hugging Face の repo id
    repo_id: ClassVar[str]

    def is_available(self, *, cache_dir: str | Path | None = None) -> bool:
        """ローカルキャッシュにモデルがあるか。ネットワークにはアクセスしない。"""
        return is_model_available(self.repo_id, cache_dir=cache_dir)

    def download(self, *, cache_dir: str | Path | None = None) -> Path:
        """モデルを取得し、設定を修正してローカルパスを返す。"""
        path = download_model(self.repo_id, cache_dir=cache_dir)
        self.fix_files(path)
        return path

    def ensure(self, *, cache_dir: str | Path | None = None) -> Path:
        """未取得なら取得し、設定を修正してローカルパスを返す。"""
        path = ensure_model(self.repo_id, cache_dir=cache_dir)
        self.fix_files(path)
        return path

    def fix_files(self, model_dir: Path) -> None:
        """取得したモデルのファイルを、必要なら修正する。既定では何もしない。"""
        return None

    @abstractmethod
    def load(self, model_dir: Path, *, cpu_threads: int = CPU_THREADS) -> Transcriber:
        """ローカルのモデルを読み込み、このモデル用の Transcriber を返す。"""
