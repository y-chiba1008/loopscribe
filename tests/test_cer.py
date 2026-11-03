"""CER ヘルパーの単体テスト。"""

from __future__ import annotations

import pytest

from tests.cer import character_error_rate, normalize_for_cer


def test_normalize_for_cer_strips_punctuation_and_space() -> None:
    assert normalize_for_cer("こんにちは、 世界！") == "こんにちは世界"


def test_character_error_rate_exact() -> None:
    assert character_error_rate("あいう", "あいう") == 0.0


def test_character_error_rate_one_substitution() -> None:
    assert character_error_rate("あいう", "あえう") == pytest.approx(1 / 3)
