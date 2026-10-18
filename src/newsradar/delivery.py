"""Write digest files and optionally send the digest by e-mail (SMTP)."""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from newsradar.config import EmailConfig, OutputConfig
from newsradar.models import Digest
from newsradar.render import render_html, render_json, render_markdown

logger = logging.getLogger(__name__)

_RENDERERS = {
    "html": ("html", render_html),
    "markdown": ("md", render_markdown),
    "json": ("json", render_json),
}


def write_outputs(digest: Digest, config: OutputConfig) -> list[Path]:
    out_dir = Path(config.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    stem = f"digest-{digest.digest_date.isoformat()}"
    for fmt in config.formats:
        suffix, render = _RENDERERS[fmt]
        path = out_dir / f"{stem}.{suffix}"
        path.write_text(render(digest), encoding="utf-8")
        written.append(path)
    return written


def build_email(digest: Digest, config: EmailConfig) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = config.subject.format(
        title=digest.title, date=digest.digest_date.isoformat(), count=digest.article_count
    )
    message["From"] = config.sender or ""
    message["To"] = ", ".join(config.to)
    message.set_content(render_markdown(digest))
    message.add_alternative(render_html(digest), subtype="html")
    return message


def send_email(digest: Digest, config: EmailConfig) -> None:
    host = os.environ.get(config.smtp_host_env)
    if not host:
        raise RuntimeError(f"e-mail enabled but {config.smtp_host_env} is not set")
    port = int(os.environ.get(config.smtp_port_env, "587"))
    user = os.environ.get(config.smtp_user_env)
    password = os.environ.get(config.smtp_password_env)
    message = build_email(digest, config)
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        if config.starttls:
            smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(message)
    logger.info("digest e-mailed to %d recipient(s)", len(config.to))
