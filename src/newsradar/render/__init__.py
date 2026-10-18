"""Render a :class:`~newsradar.models.Digest` as HTML, Markdown or JSON."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from newsradar import i18n
from newsradar.markup import render_inline, strip_inline
from newsradar.models import Digest


def _environment(digest: Digest) -> Environment:
    env = Environment(
        loader=PackageLoader("newsradar.render", "templates"),
        # HTML templates are escaped; the Markdown template is plain text.
        autoescape=select_autoescape(["html.j2", "html"], default_for_string=True),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["inline_md"] = render_inline
    env.filters["plain"] = strip_inline
    fmt = i18n.Formatter(digest.locale, digest.timezone)
    env.filters["date"] = fmt.date
    env.filters["datetime"] = fmt.datetime
    env.filters["number"] = fmt.number
    env.globals["t"] = i18n.strings(digest.language)
    return env


def render_html(digest: Digest) -> str:
    return _environment(digest).get_template("digest.html.j2").render(digest=digest)


def render_markdown(digest: Digest) -> str:
    return _environment(digest).get_template("digest.md.j2").render(digest=digest)


def render_json(digest: Digest) -> str:
    return digest.model_dump_json(indent=2)


__all__ = ["render_html", "render_json", "render_markdown"]
