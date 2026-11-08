"""`--model` で選べるモデルの登録。"""

from __future__ import annotations

from loopscribe.models.base import ModelSpec
from loopscribe.models.kotoba import KotobaModel
from loopscribe.models.small import SmallModel

MODELS: dict[str, ModelSpec] = {spec.name: spec for spec in (SmallModel(), KotobaModel())}
DEFAULT_MODEL = SmallModel.name


def get_model(name: str) -> ModelSpec:
    """名前からモデルを返す。未知の名前なら ValueError。"""
    try:
        return MODELS[name]
    except KeyError:
        choices = ", ".join(MODELS)
        raise ValueError(f"未知のモデルです: {name}(選べるのは {choices})") from None


__all__ = ["DEFAULT_MODEL", "MODELS", "ModelSpec", "get_model"]
