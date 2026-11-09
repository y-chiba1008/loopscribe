"""設定ファイル(TOML)の読み書き。項目は `default-model` だけ。

- 場所: Windows は `%APPDATA%\\loopscribe\\config.toml`、
  それ以外は `$XDG_CONFIG_HOME/loopscribe/config.toml`(未設定なら `~/.config`)。
  環境変数 `LOOPSCRIBE_CONFIG` があれば、そのパスを使う。
- 優先順位は CLIオプション > 設定ファイル > 組み込みの既定。
- 起動時(`load_settings`)は、壊れている項目を警告して無視する(警告のみの方針)。
- `config` での更新(`write_item`)は、壊れたファイルを上書きしない(エラーで止める)。
  書き出しは `tomli-w` で行うため、手で書いたコメントは消える。
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from loopscribe.constants import OUTPUT_ENCODING
from loopscribe.errors import ConfigError
from loopscribe.models import DEFAULT_MODEL, MODELS

CONFIG_ENV_VAR = "LOOPSCRIBE_CONFIG"
CONFIG_DIR_NAME = "loopscribe"
CONFIG_FILE_NAME = "config.toml"

DEFAULT_MODEL_KEY = "default-model"


@dataclass(frozen=True)
class ConfigItem:
    """設定ファイルの項目。"""

    key: str
    default: str
    description: str


# 項目の一覧。項目を増やすときは、ここと `validate_value` を更新する
ITEMS: tuple[ConfigItem, ...] = (
    ConfigItem(
        key=DEFAULT_MODEL_KEY,
        default=DEFAULT_MODEL,
        description="--model を省略したときに使うモデル(" + " / ".join(MODELS) + ")",
    ),
)
_ITEMS_BY_KEY: dict[str, ConfigItem] = {item.key: item for item in ITEMS}


def config_path() -> Path:
    """設定ファイルの場所を返す。`LOOPSCRIBE_CONFIG` があれば優先する。"""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override)
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        # XDG の仕様どおり、相対パスは無視する
        base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".config"
    return base / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def check_key(key: str) -> ConfigItem:
    """項目名を検査し、項目を返す。未知の項目名なら ConfigError。"""
    item = _ITEMS_BY_KEY.get(key)
    if item is None:
        names = ", ".join(_ITEMS_BY_KEY)
        raise ConfigError(f"未知の項目です: {key}(使えるのは {names})")
    return item


def validate_value(key: str, value: str) -> str:
    """項目の値を検査して返す。不正なら ConfigError。"""
    check_key(key)
    if key == DEFAULT_MODEL_KEY and value not in MODELS:
        choices = " / ".join(MODELS)
        raise ConfigError(f"{key} に指定できるのは {choices} です: {value}")
    return value


def read_raw(path: Path) -> dict[str, object]:
    """設定ファイルを読んで、そのままの辞書で返す。

    ファイルが無ければ空の辞書。読めなければ ConfigError。
    """
    try:
        # Windows のメモ帳などが付ける BOM があっても読めるように utf-8-sig で読む
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"設定ファイルを読めません: {path}: {exc}") from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"設定ファイルの書式が不正です: {path}: {exc}") from exc


@dataclass(frozen=True)
class Settings:
    """起動時に読んだ設定。`values` は設定ファイルにある有効な項目だけ。"""

    path: Path
    values: Mapping[str, str]
    warnings: tuple[str, ...] = ()

    def value_of(self, key: str) -> str:
        """項目の値。設定ファイルに無ければ組み込みの既定。"""
        if key in self.values:
            return self.values[key]
        return check_key(key).default


def load_settings(path: Path | None = None) -> Settings:
    """起動時に設定を読む。壊れている項目は警告を付けて無視する。"""
    path = path if path is not None else config_path()
    try:
        raw = read_raw(path)
    except ConfigError as exc:
        return Settings(path, {}, (f"{exc}。設定を無視して続行します。",))

    values: dict[str, str] = {}
    warnings: list[str] = []
    for key, value in raw.items():
        try:
            if not isinstance(value, str):
                raise ConfigError(f"{key} は文字列で指定してください: {value!r}")
            values[key] = validate_value(key, value)
        except ConfigError as exc:
            warnings.append(f"{exc}。この項目を無視します。")
    return Settings(path, values, tuple(warnings))


def write_item(path: Path, key: str, value: str | None) -> None:
    """項目を設定する。`value` が None なら項目を消して既定に戻す。他の項目はそのまま残す。

    項目名や値が不正なとき、またはファイルが壊れているときは ConfigError を出し、
    ファイルには触らない。
    """
    check_key(key)
    raw = read_raw(path)
    if value is None:
        raw.pop(key, None)
    else:
        raw[key] = validate_value(key, value)

    path.parent.mkdir(parents=True, exist_ok=True)
    # 途中で失敗しても元のファイルが壊れないように、一時ファイルに書いてから置き換える
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(tomli_w.dumps(raw), encoding=OUTPUT_ENCODING)
        os.replace(tmp_path, path)
    except OSError as exc:
        raise ConfigError(f"設定ファイルを書けません: {path}: {exc}") from exc
