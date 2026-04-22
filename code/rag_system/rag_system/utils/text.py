from __future__ import annotations

import re

AZERBAIJANI_WHITESPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return HTML_RE.sub(" ", text or "")


def normalize_whitespace(text: str) -> str:
    return AZERBAIJANI_WHITESPACE_RE.sub(" ", text or "").strip()


def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value)
