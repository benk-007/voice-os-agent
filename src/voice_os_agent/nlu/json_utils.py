"""Turn the raw T5 output into a Python dict.

The model was trained to emit flat ``"key":"value"`` pairs, usually without the
surrounding braces (e.g. ``"app":"firefox","intent":"open_app"``). These helpers
are tolerant to the common deviations (missing braces, single quotes, curly
quotes, trailing commas). Parsing never evaluates code: ``ast.literal_eval``
only accepts Python literals.
"""

from __future__ import annotations

import ast
import json
import re

_KV_LIST = re.compile(
    r"""^\s*(?:["']?[\w\- ]+["']?\s*:\s*["'][^"']*["']\s*)(?:,\s*(?:["']?[\w\- ]+["']?\s*:\s*["'][^"']*["']\s*))*$"""
)
_CURLY_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def normalize_quotes_and_commas(s: str) -> str:
    """Replace typographic quotes with ASCII quotes and drop trailing commas."""
    s = s.translate(_CURLY_QUOTES)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


def looks_like_kv_list(s: str) -> bool:
    return bool(_KV_LIST.match(s.strip()))


def ensure_json_like(s: str) -> str:
    """Wrap a bare ``"k":"v", ...`` list in braces when needed."""
    s = normalize_quotes_and_commas(s)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        return m.group(0)
    if looks_like_kv_list(s) or re.match(r"""^\s*["'].*["']\s*:\s*["'].*""", s):
        return "{" + s + "}"
    return s


def parse_json_from_text(s: str) -> dict | None:
    """Best-effort conversion of a model output to a dict; ``None`` if impossible."""
    cand = ensure_json_like(s)
    for loader in (json.loads, lambda c: json.loads(c.replace("'", '"')), ast.literal_eval):
        try:
            value = loader(cand)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        return value if isinstance(value, dict) else None
    return None
