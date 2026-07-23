"""Stage 5 前半:文字正規化。

- 顯示/翻譯用文字:NFKC、折疊連續空白、統一換行,但不移除中日文標點。
- 只供比對用的 match_text:額外轉小寫並移除空白,不破壞顯示原文。
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_INLINE_WS = re.compile(r"[^\S\n]+")  # 空白但不含換行
_ALL_WS = re.compile(r"\s+")


def has_meaningful_text(text: str) -> bool:
    """至少要有一個字母或數字(含中日文表意字);純空白、純標點、單一裝飾符號回 False。"""
    for ch in text:
        if unicodedata.category(ch)[0] in ("L", "N"):
            return True
    return False


def normalize_display(text: str) -> str:
    """顯示與翻譯用的正規化文字:NFKC + 折疊行內空白 + 統一換行,不移除標點。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def match_text(text: str) -> str:
    """只供比對:在顯示正規化之上再轉小寫並移除所有空白。"""
    t = normalize_display(text)
    t = unicodedata.normalize("NFKC", t).casefold()
    return _ALL_WS.sub("", t)


def similarity(a: str, b: str) -> float:
    """兩段文字的正規化相似度(0~1)。"""
    ma, mb = match_text(a), match_text(b)
    if not ma and not mb:
        return 1.0
    if not ma or not mb:
        return 0.0
    return SequenceMatcher(None, ma, mb).ratio()
