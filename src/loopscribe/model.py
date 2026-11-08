"""Whisper モデルの取得と未取得判定(モデルの種類に依存しない)。

model_name には faster-whisper のサイズ名(`small` など)か、Hugging Face の repo id を渡す。
"""

from __future__ import annotations

from pathlib import Path

from faster_whisper.utils import download_model as _fw_download_model
from huggingface_hub.errors import LocalEntryNotFoundError

from loopscribe.errors import ModelUnavailableError


def is_model_available(
    model_name: str,
    *,
    cache_dir: str | Path | None = None,
) -> bool:
    """ローカルキャッシュにモデルがあるか。ネットワークにはアクセスしない。"""
    try:
        _fw_download_model(
            model_name,
            local_files_only=True,
            cache_dir=_as_optional_str(cache_dir),
        )
    except LocalEntryNotFoundError:
        return False
    except ValueError:
        # 未知のモデル名など
        return False
    return True


def download_model(
    model_name: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """モデルを取得してローカルパスを返す。失敗時は ModelUnavailableError。"""
    try:
        path = _fw_download_model(
            model_name,
            local_files_only=False,
            cache_dir=_as_optional_str(cache_dir),
        )
    except Exception as exc:
        raise ModelUnavailableError(f"モデル '{model_name}' の取得に失敗しました: {exc}") from exc
    return Path(path)


def ensure_model(
    model_name: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """モデルのローカルパスを返す。未取得なら取得する。"""
    cache = _as_optional_str(cache_dir)
    try:
        path = _fw_download_model(model_name, local_files_only=True, cache_dir=cache)
        return Path(path)
    except LocalEntryNotFoundError:
        return download_model(model_name, cache_dir=cache_dir)
    except ValueError as exc:
        raise ModelUnavailableError(f"モデル '{model_name}' は無効です: {exc}") from exc


def _as_optional_str(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(path)
