"""CLI locale for operator-facing help / banners (MLBOT_LANG / --lang)."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

Lang = str  # "zh" | "en"

_lang: Lang = "zh"

_MESSAGES: dict[Lang, dict[str, str]] = {
    "zh": {
        "cli.lab.help": "本地研发 Lab：实验管理 + results 浏览 + A股/港股/币圈辅助。",
        "cli.lab.banner": "🧪 MLBot Lab — 本地研发门户 (FastAPI + React)",
        "cli.lab.rd": "   实验管理: http://{host}:{port}/rd",
        "cli.lab.qa": "   问答:     http://{host}:{port}/rd/qa",
        "cli.lab.browse": "   浏览:     http://{host}:{port}/browse",
        "cli.lab.ashare": "   A股辅助:  http://{host}:{port}/ashare",
        "cli.lab.hk": "   港股辅助: http://{host}:{port}/hk",
        "cli.lab.macro": "   大环境:   http://{host}:{port}/macro",
        "cli.stop": "   Ctrl+C 停止",
        "cli.console.help": "实盘 Business CMS（Trade Map / orders / account）；不含本地实验管理。",
        "cli.console.banner": "🌐 MLBot Business Console — 实盘 CMS (FastAPI)",
        "cli.console.hint_lab": "   本地研发:  mlbot lab → :8008/rd",
        "cli.rolling_dashboard.deprecated": (
            "⚠️  DEPRECATED: `mlbot rolling-dashboard` 已更名为 `mlbot lab`。"
            " 正在启动 Lab…"
        ),
        "cli.multileg.help": "多腿管线（研究 / 回测 / 实盘相关子命令）。",
        "cli.pipeline.help": "研究管线（特征 / 扫描 / 回测编排）。",
        "cli.bind": "   bind:     http://{bind}:{port}/",
        "cli.bind_console": "   bind:      http://{bind}:{port}/",
        "cli.trade_map": "   Trade Map: http://{host}:{port}/trade-map",
    },
    "en": {
        "cli.lab.help": "Local R&D Lab: experiments, results browse, A-share / HK / crypto aux.",
        "cli.lab.banner": "🧪 MLBot Lab — local R&D portal (FastAPI + React)",
        "cli.lab.rd": "   Experiments: http://{host}:{port}/rd",
        "cli.lab.qa": "   Q&A:         http://{host}:{port}/rd/qa",
        "cli.lab.browse": "   Browse:      http://{host}:{port}/browse",
        "cli.lab.ashare": "   A-share:     http://{host}:{port}/ashare",
        "cli.lab.hk": "   HK:          http://{host}:{port}/hk",
        "cli.lab.macro": "   Macro:       http://{host}:{port}/macro",
        "cli.stop": "   Ctrl+C to stop",
        "cli.console.help": (
            "Live Business CMS (Trade Map / orders / account); no local experiment UI."
        ),
        "cli.console.banner": "🌐 MLBot Business Console — live CMS (FastAPI)",
        "cli.console.hint_lab": "   Local R&D:  mlbot lab → :8008/rd",
        "cli.rolling_dashboard.deprecated": (
            "⚠️  DEPRECATED: `mlbot rolling-dashboard` was renamed to `mlbot lab`. "
            "Starting Lab…"
        ),
        "cli.multileg.help": "Multi-leg pipeline (research / backtest / live-related subcommands).",
        "cli.pipeline.help": "Research pipeline (features / scans / backtest orchestration).",
        "cli.bind": "   bind:     http://{bind}:{port}/",
        "cli.bind_console": "   bind:      http://{bind}:{port}/",
        "cli.trade_map": "   Trade Map: http://{host}:{port}/trade-map",
    },
}


def normalize_lang(raw: Optional[str]) -> Lang:
    s = (raw or "").strip().lower()
    if s.startswith("en"):
        return "en"
    return "zh"


def init_lang_from_env() -> Lang:
    global _lang
    _lang = normalize_lang(os.environ.get("MLBOT_LANG") or os.environ.get("LANG"))
    # LANG=en_US.UTF-8 → en; LANG=zh_CN.UTF-8 → zh; unset → zh
    env_lang = os.environ.get("MLBOT_LANG")
    if env_lang:
        _lang = normalize_lang(env_lang)
    else:
        # Do not treat system LANG=C / en_US as forcing English for operators;
        # only MLBOT_LANG switches. Default zh.
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


# Initialize on import so Click docstrings can use t() if evaluated late.
init_lang_from_env()
