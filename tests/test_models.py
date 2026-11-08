"""models の単体テスト(実モデル・ネットワークなし)。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from loopscribe.errors import ModelUnavailableError
from loopscribe.models import DEFAULT_MODEL, MODELS, get_model
from loopscribe.models.kotoba import (
    ALIGNMENT_HEADS,
    KotobaModel,
    KotobaTranscriber,
    fix_alignment_heads,
)
from loopscribe.models.small import SmallModel, SmallTranscriber
from loopscribe.transcriber import Segment

_LARGE_V3_HEADS = [[7, 0], [10, 17], [12, 18], [25, 6]]


def _raw_result() -> tuple[object, object]:
    raw = [SimpleNamespace(start=0.0, end=1.0, text=" こんにちは", words=None)]
    return iter(raw), SimpleNamespace()


# --- 登録 ---


def test_registry_has_small_and_kotoba() -> None:
    assert list(MODELS) == ["small", "kotoba"]
    assert DEFAULT_MODEL == "small"
    assert isinstance(get_model("small"), SmallModel)
    assert isinstance(get_model("kotoba"), KotobaModel)
    assert get_model("kotoba").repo_id == "kotoba-tech/kotoba-whisper-v2.0-faster"


def test_get_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="small, kotoba"):
        get_model("large")


# --- ModelSpec の取得 ---


def test_ensure_fixes_files_after_getting_model(tmp_path: Path) -> None:
    spec = KotobaModel()
    with (
        patch("loopscribe.models.base.ensure_model", return_value=tmp_path) as mock_ensure,
        patch.object(KotobaModel, "fix_files") as mock_fix,
    ):
        assert spec.ensure() == tmp_path
    mock_ensure.assert_called_once_with("kotoba-tech/kotoba-whisper-v2.0-faster", cache_dir=None)
    mock_fix.assert_called_once_with(tmp_path)


def test_download_fixes_files_after_download(tmp_path: Path) -> None:
    spec = KotobaModel()
    with (
        patch("loopscribe.models.base.download_model", return_value=tmp_path) as mock_dl,
        patch.object(KotobaModel, "fix_files") as mock_fix,
    ):
        assert spec.download() == tmp_path
    mock_dl.assert_called_once_with("kotoba-tech/kotoba-whisper-v2.0-faster", cache_dir=None)
    mock_fix.assert_called_once_with(tmp_path)


def test_small_does_not_touch_files(tmp_path: Path) -> None:
    with patch("loopscribe.models.base.ensure_model", return_value=tmp_path):
        assert SmallModel().ensure() == tmp_path
    assert list(tmp_path.iterdir()) == []


# --- small ---


def test_small_transcriber_uses_fixed_parameters() -> None:
    mock_model = MagicMock()
    mock_model.transcribe.return_value = _raw_result()
    with patch("loopscribe.models.small.WhisperModel", return_value=mock_model) as mock_cls:
        tr = SmallModel().load(Path("/models/small"), cpu_threads=3)

    assert isinstance(tr, SmallTranscriber)
    mock_cls.assert_called_once_with(
        str(Path("/models/small")), device="cpu", compute_type="int8", cpu_threads=3
    )
    assert tr.transcribe(np.zeros(16_000, dtype=np.float32)) == [
        Segment(start=0.0, end=1.0, text="こんにちは")
    ]
    kwargs = mock_model.transcribe.call_args.kwargs
    assert kwargs == {
        "language": "ja",
        "beam_size": 1,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }


# --- kotoba ---


def test_kotoba_transcriber_uses_batched_pipeline() -> None:
    mock_pipeline = MagicMock()
    mock_pipeline.transcribe.return_value = _raw_result()
    with (
        patch("loopscribe.models.kotoba.WhisperModel") as mock_cls,
        patch(
            "loopscribe.models.kotoba.BatchedInferencePipeline", return_value=mock_pipeline
        ) as mock_batched,
    ):
        tr = KotobaModel().load(Path("/models/kotoba"), cpu_threads=6)

    assert isinstance(tr, KotobaTranscriber)
    mock_cls.assert_called_once_with(
        str(Path("/models/kotoba")), device="cpu", compute_type="int8", cpu_threads=6
    )
    mock_batched.assert_called_once_with(mock_cls.return_value)
    assert tr.transcribe(np.zeros(16_000, dtype=np.float64)) == [
        Segment(start=0.0, end=1.0, text="こんにちは")
    ]
    passed = mock_pipeline.transcribe.call_args.args[0]
    assert passed.dtype == np.float32
    kwargs = mock_pipeline.transcribe.call_args.kwargs
    assert kwargs == {
        "language": "ja",
        "beam_size": 1,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "chunk_length": 15,
        "batch_size": 1,
        "without_timestamps": False,
        "word_timestamps": True,
    }


def test_kotoba_alignment_heads_use_last_decoder_layer() -> None:
    heads = ALIGNMENT_HEADS
    assert heads == [[1, h] for h in range(20)]


def _write_config(path: Path, heads: list[list[int]]) -> dict[str, object]:
    config: dict[str, object] = {"alignment_heads": heads, "lang_ids": [50259, 50260]}
    path.write_text(json.dumps(config), encoding="utf-8")
    return config


def test_fix_alignment_heads_rewrites_regular_file(tmp_path: Path) -> None:
    _write_config(tmp_path / "config.json", _LARGE_V3_HEADS)

    assert fix_alignment_heads(tmp_path) is True

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config == {"alignment_heads": ALIGNMENT_HEADS, "lang_ids": [50259, 50260]}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX の権限で確認する")
def test_fix_alignment_heads_keeps_file_mode(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_config(path, _LARGE_V3_HEADS)
    path.chmod(0o644)

    assert fix_alignment_heads(tmp_path) is True
    assert path.stat().st_mode & 0o777 == 0o644


def test_fix_alignment_heads_is_idempotent(tmp_path: Path) -> None:
    _write_config(tmp_path / "config.json", _LARGE_V3_HEADS)
    assert fix_alignment_heads(tmp_path) is True
    before = (tmp_path / "config.json").stat().st_mtime_ns

    assert fix_alignment_heads(tmp_path) is False
    assert (tmp_path / "config.json").stat().st_mtime_ns == before


def test_fix_alignment_heads_replaces_symlink_without_touching_blob(tmp_path: Path) -> None:
    # HFキャッシュの構成: snapshots/<rev>/config.json -> ../../blobs/<hash>
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "rev"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    blob = blobs / "abc123"
    original = _write_config(blob, _LARGE_V3_HEADS)
    link = snapshot / "config.json"
    link.symlink_to(Path("..") / ".." / "blobs" / "abc123")

    assert fix_alignment_heads(snapshot) is True

    assert not link.is_symlink()
    fixed = json.loads(link.read_text(encoding="utf-8"))
    assert fixed["alignment_heads"] == ALIGNMENT_HEADS
    assert json.loads(blob.read_text(encoding="utf-8")) == original


def test_fix_alignment_heads_missing_config(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="修正できませんでした"):
        fix_alignment_heads(tmp_path)


def test_fix_alignment_heads_broken_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelUnavailableError, match="修正できませんでした"):
        fix_alignment_heads(tmp_path)
