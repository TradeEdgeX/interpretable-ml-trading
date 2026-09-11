"""CLI locale for operator-facing help / banners (MLBOT_LANG / --lang)."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

Lang = str  # "zh" | "en"

_lang: Lang = "zh"

_MESSAGES: dict[Lang, dict[str, str]] = {
    "zh": {
        "cli.lab.help": "本地法庭 Lab：实验管理 + 问答 + results 浏览。不含辅助盘。",
        "cli.lab.banner": "🧪 MLBot Lab — 本地法庭门户（实验 / 问答 / results）",
        "cli.lab.rd": "   实验管理: http://{host}:{port}/rd",
        "cli.lab.qa": "   问答:     http://{host}:{port}/rd/qa",
        "cli.lab.browse": "   浏览:     http://{host}:{port}/browse",
        "cli.stop": "   Ctrl+C 停止",
        "cli.bind": "   bind:     http://{bind}:{port}/",
    },
    "en": {
        "cli.lab.help": "Local court Lab: experiments, Q&A, results browse. No auxiliary trading.",
        "cli.lab.banner": "🧪 MLBot Lab — local court portal (experiments / Q&A / results)",
        "cli.lab.rd": "   Experiments: http://{host}:{port}/rd",
        "cli.lab.qa": "   Q&A:         http://{host}:{port}/rd/qa",
        "cli.lab.browse": "   Browse:      http://{host}:{port}/browse",
        "cli.stop": "   Ctrl+C to stop",
        "cli.bind": "   bind:     http://{bind}:{port}/",
    },
}


def normalize_lang(raw: Optional[str]) -> Lang:
    s = (raw or "").strip().lower()
    if s.startswith("en"):
        return "en"
    return "zh"


def init_lang_from_env() -> Lang:
    global _lang
    env_lang = os.environ.get("MLBOT_LANG")
    if env_lang:
        _lang = normalize_lang(env_lang)
    else:
        _lang = "zh"
    return _lang


def get_lang() -> Lang:
    return _lang


def set_lang(lang: Lang) -> None:
    global _lang
    _lang = normalize_lang(lang)


def t(key: str, **vars: Any) -> str:
    catalog = _MESSAGES.get(_lang) or _MESSAGES["zh"]
    template = catalog.get(key) or _MESSAGES["zh"].get(key) or key
    if not vars:
        return template
    try:
        return template.format(**vars)
    except (KeyError, ValueError):
        return template


def register_messages(lang: Lang, messages: Mapping[str, str]) -> None:
    bucket = _MESSAGES.setdefault(normalize_lang(lang), {})
    bucket.update(dict(messages))


init_lang_from_env()
