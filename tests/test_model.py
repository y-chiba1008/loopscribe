"""model の単体テスト(ネットワークなし)。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from loopscribe.errors import ModelUnavailableError
from loopscribe.model import download_model, ensure_model, is_model_available


def test_is_model_available_true() -> None:
    with patch(
        "loopscribe.model._fw_download_model",
        return_value="/cache/model",
    ) as mock_dl:
        assert is_model_available("small") is True
        mock_dl.assert_called_once_with("small", local_files_only=True, cache_dir=None)


def test_is_model_available_false_when_missing() -> None:
    with patch(
        "loopscribe.model._fw_download_model",
        side_effect=LocalEntryNotFoundError("missing"),
    ):
        assert is_model_available("tiny") is False


def test_is_model_available_false_on_invalid_name() -> None:
    with patch(
        "loopscribe.model._fw_download_model",
        side_effect=ValueError("Invalid model size"),
    ):
        assert is_model_available("no-such-model") is False


def test_download_model_returns_path() -> None:
    with patch(
        "loopscribe.model._fw_download_model",
        return_value="/cache/downloaded",
    ) as mock_dl:
        path = download_model("small", cache_dir=Path("/tmp/cache"))
        assert path == Path("/cache/downloaded")
        mock_dl.assert_called_once_with(
            "small",
            local_files_only=False,
            cache_dir="/tmp/cache",
        )


def test_download_model_wraps_failure() -> None:
    with (
        patch(
            "loopscribe.model._fw_download_model",
            side_effect=RuntimeError("network down"),
        ),
        pytest.raises(ModelUnavailableError, match="取得に失敗"),
    ):
        download_model("small")


def test_ensure_model_uses_cache_when_present() -> None:
    with patch(
        "loopscribe.model._fw_download_model",
        return_value="/cache/hit",
    ) as mock_dl:
        path = ensure_model("small")
        assert path == Path("/cache/hit")
        mock_dl.assert_called_once_with("small", local_files_only=True, cache_dir=None)


def test_ensure_model_downloads_when_missing() -> None:
    def side_effect(name: str, *, local_files_only: bool, cache_dir: str | None):
        if local_files_only:
            raise LocalEntryNotFoundError("missing")
        return "/cache/new"

    with patch("loopscribe.model._fw_download_model", side_effect=side_effect):
        path = ensure_model("small")
        assert path == Path("/cache/new")
