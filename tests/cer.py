"""slow テスト用の CER 計算(bench_models と同じ正規化)。"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


def normalize_for_cer(text: str) -> str:
    """CER用の正規化: NFKC・小文字化し、空白と句読点・記号を除く。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if not (ch.isspace() or unicodedata.category(ch)[0] in "PSZ"))


def character_error_rate(reference: str, hypothesis: str) -> float:
    """正規化後の文字誤り率(レーベンシュタイン距離 / 正解文字数)。"""
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class CerScore:
    cer: float
    ref_chars: int
    hyp_chars: int
