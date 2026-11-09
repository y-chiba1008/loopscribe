"""テスト全体の共通設定。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def config_file(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """設定ファイルを一時ディレクトリに向け、実際のユーザーの設定ファイルを触らない。

    ファイルはまだ作らない(テストで必要なときに作る)。
    """
    path = tmp_path_factory.mktemp("config") / "config.toml"
    monkeypatch.setenv("LOOPSCRIBE_CONFIG", str(path))
    yield path
