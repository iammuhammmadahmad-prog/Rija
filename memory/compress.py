"""
Lightweight context compression for RAG / long prompts.

Inspired by OmniRoute's lite compression (whitespace collapse, filler
stripping, hard budget). Keeps Rija educational — no external models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


_FILLER = re.compile(
    r"\b(actually|basically|literally|just|really|very|quite|perhaps|"
    r"maybe|sort of|kind of|you know|i mean|in order to|due to the fact that)\b",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")


@dataclass
class CompressionResult:
    text: str
    original_chars: int
    compressed_chars: int

    @property
    def ratio(self) -> float:
        if self.original_chars == 0:
            return 1.0
        return self.compressed_chars / self.original_chars


def collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NL.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def strip_filler(text: str) -> str:
    text = _FILLER.sub("", text)
    return collapse_whitespace(text)


def hard_budget(text: str, max_chars: int, keep_tail: bool = False) -> str:
    """Truncate to max_chars, preferring head (or tail for chat history)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if keep_tail:
        return "…" + text[-(max_chars - 1) :]
    return text[: max_chars - 1] + "…"


def compress_context(
    text: str,
    *,
    max_chars: Optional[int] = 1500,
    strip_fillers: bool = True,
    keep_tail: bool = False,
) -> CompressionResult:
    original = len(text)
    out = collapse_whitespace(text)
    if strip_fillers:
        out = strip_filler(out)
    if max_chars is not None:
        out = hard_budget(out, max_chars, keep_tail=keep_tail)
    return CompressionResult(text=out, original_chars=original, compressed_chars=len(out))


def compress_snippets(snippets: List[str], max_chars: int = 1500) -> str:
    """Join snippets under a shared budget (OmniRoute-style hard budget)."""
    parts: List[str] = []
    used = 0
    for snip in snippets:
        cleaned = compress_context(snip, max_chars=None).text
        if used + len(cleaned) + 2 > max_chars:
            remain = max_chars - used - 2
            if remain > 40:
                parts.append(hard_budget(cleaned, remain))
            break
        parts.append(cleaned)
        used += len(cleaned) + 2
    return "\n\n".join(parts)
