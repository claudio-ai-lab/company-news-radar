"""Safe rendering of the sparse inline Markdown the summariser is allowed to emit.

LLM output and feed text are untrusted. We HTML-escape everything first and only then
turn ``**bold**`` / ``*italic*`` markers into ``<strong>`` / ``<em>``. Because escaping
happens before any tag is inserted, the only tags that can appear are our own.
"""

from __future__ import annotations

import re

from markupsafe import Markup, escape

_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.DOTALL)
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", re.DOTALL)


def balance_markers(text: str) -> str:
    """Drop a dangling ``**`` or ``*`` (e.g. after truncation) so it never renders literally."""
    out = text
    if out.count("**") % 2:
        i = out.rfind("**")
        out = out[:i] + out[i + 2 :]
    if out.replace("**", "").count("*") % 2:
        i = out.rfind("*")
        out = out[:i] + out[i + 1 :]
    return out


def render_inline(text: str | None) -> Markup:
    """Escape ``text`` and convert bold/italic markers into HTML. Safe for ``|safe`` output."""
    if not text:
        return Markup("")
    escaped = str(escape(balance_markers(text)))
    escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
    escaped = _ITALIC.sub(r"<em>\1</em>", escaped)
    return Markup(escaped)  # content was escaped above


def strip_inline(text: str | None) -> str:
    """Remove bold/italic markers, keeping the words (for titles, subjects, prompts)."""
    if not text:
        return ""
    out = _BOLD.sub(r"\1", text)
    out = _ITALIC.sub(r"\1", out)
    return out.replace("**", "")
