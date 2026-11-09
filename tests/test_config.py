"""config モジュールの単体テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from loopscribe.config import (
    DEFAULT_MODEL_KEY,
    config_path,
    load_settings,
    read_raw,
    validate_value,
    write_item,
)
from loopscribe.errors import ConfigError
from loopscribe.models import DEFAULT_MODEL


def test_config_path_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "custom.toml"
    monkeypatch.setenv("LOOPSCRIBE_CONFIG", str(target))
    monkeypatch.setattr(sys, "platform", "win32")
    assert config_path() == target


def test_config_path_windows_uses_appdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LOOPSCRIBE_CONFIG")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert config_path() == tmp_path / "loopscribe" / "config.toml"


def test_config_path_linux_uses_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LOOPSCRIBE_CONFIG")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_path() == tmp_path / "loopscribe" / "config.toml"


def test_config_path_linux_ignores_relative_xdg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LOOPSCRIBE_CONFIG")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/dir")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_path() == tmp_path / ".config" / "loopscribe" / "config.toml"


def test_missing_file_uses_default(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "none.toml")
    assert settings.warnings == ()
    assert settings.values == {}
    assert settings.value_of(DEFAULT_MODEL_KEY) == DEFAULT_MODEL


def test_valid_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('default-model = "kotoba"\n', encoding="utf-8")
    settings = load_settings(path)
    assert settings.warnings == ()
    assert settings.value_of(DEFAULT_MODEL_KEY) == "kotoba"


def test_file_with_bom_is_read(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes('\ufeffdefault-model = "kotoba"\n'.encode("utf-8"))
    assert load_settings(path).value_of(DEFAULT_MODEL_KEY) == "kotoba"


def test_broken_toml_is_warned_and_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('default-model = "kotoba\n', encoding="utf-8")
    settings = load_settings(path)
    assert len(settings.warnings) == 1
    assert "書式が不正" in settings.warnings[0]
    assert settings.value_of(DEFAULT_MODEL_KEY) == DEFAULT_MODEL


def test_invalid_model_is_warned_and_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('default-model = "large"\n', encoding="utf-8")
    settings = load_settings(path)
    assert len(settings.warnings) == 1
    assert "large" in settings.warnings[0]
    assert settings.value_of(DEFAULT_MODEL_KEY) == DEFAULT_MODEL


def test_non_string_value_is_warned_and_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("default-model = 1\n", encoding="utf-8")
    settings = load_settings(path)
    assert len(settings.warnings) == 1
    assert settings.value_of(DEFAULT_MODEL_KEY) == DEFAULT_MODEL


def test_unknown_key_is_warned_and_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('unknown = "x"\n', encoding="utf-8")
    settings = load_settings(path)
    assert len(settings.warnings) == 1
    assert "未知の項目" in settings.warnings[0]


def test_validate_value_rejects_unknown_key_and_value() -> None:
    with pytest.raises(ConfigError):
        validate_value("nope", "x")
    with pytest.raises(ConfigError):
        validate_value(DEFAULT_MODEL_KEY, "large")
    assert validate_value(DEFAULT_MODEL_KEY, "small") == "small"


def test_write_item_creates_file_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "config.toml"
    write_item(path, DEFAULT_MODEL_KEY, "kotoba")
    assert path.is_file()
    assert read_raw(path) == {DEFAULT_MODEL_KEY: "kotoba"}
    assert load_settings(path).value_of(DEFAULT_MODEL_KEY) == "kotoba"


def test_write_item_keeps_other_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('other = "keep"\ndefault-model = "small"\n', encoding="utf-8")
    write_item(path, DEFAULT_MODEL_KEY, "kotoba")
    assert read_raw(path) == {"other": "keep", DEFAULT_MODEL_KEY: "kotoba"}


def test_write_item_none_removes_key(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    write_item(path, DEFAULT_MODEL_KEY, "kotoba")
    write_item(path, DEFAULT_MODEL_KEY, None)
    assert read_raw(path) == {}
    assert load_settings(path).value_of(DEFAULT_MODEL_KEY) == DEFAULT_MODEL


def test_write_item_rejects_invalid_input_without_touching_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    write_item(path, DEFAULT_MODEL_KEY, "kotoba")
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ConfigError):
        write_item(path, DEFAULT_MODEL_KEY, "large")
    with pytest.raises(ConfigError):
        write_item(path, "nope", "x")
    with pytest.raises(ConfigError):
        write_item(path, "nope", None)
    assert path.read_text(encoding="utf-8") == before


def test_write_item_does_not_overwrite_broken_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("this is = not [valid\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        write_item(path, DEFAULT_MODEL_KEY, "kotoba")
    assert path.read_text(encoding="utf-8") == "this is = not [valid\n"
    assert not path.with_name("config.toml.tmp").exists()
