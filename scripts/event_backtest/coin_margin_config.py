"""COIN-M backtest defaults — shared by event_backtest CLI, matrix, bc_coin_backtest."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BACKTEST_YAML = (
    _REPO_ROOT
    / "config"
    / "experiments"
    / "20260627_coin_layer2_native"
    / "backtest.yaml"
)


@lru_cache(maxsize=4)
def load_backtest_config(path: Optional[str] = None) -> dict[str, Any]:
    p = Path(path) if path else _DEFAULT_BACKTEST_YAML
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def backtest_output_root(config: Optional[Mapping[str, Any]] = None) -> Path:
    cfg = dict(config or load_backtest_config())
    return _REPO_ROOT / str(cfg.get("output_root", "results/coin_margin/bc_native"))


def resolve_contract_multiplier(
    symbol: str,
    *,
    margin_mode: str = "usd_m",
    cli_override: Optional[float] = None,
    usd_m_default: float = 100.0,
) -> float:
    """Return contract multiplier for *symbol*.

    *usd_m* always uses ``cli_override`` or ``usd_m_default``.
    *coin_m* uses ``cli_override`` when set; otherwise built-in dapi mapping table.
    """
    mode = str(margin_mode or "usd_m").lower()
    if mode != "coin_m":
        return float(cli_override if cli_override is not None else usd_m_default)
    if cli_override is not None:
        return float(cli_override)
    from order_management.exchange.symbol_map import contract_multiplier_for

    return float(contract_multiplier_for(str(symbol).upper()))


def resolve_contract_multipliers(
    symbols: list[str],
    *,
    margin_mode: str = "usd_m",
    cli_override: Optional[float] = None,
) -> dict[str, float]:
    return {
        str(sym).upper(): resolve_contract_multiplier(
            sym, margin_mode=margin_mode, cli_override=cli_override
        )
        for sym in symbols
    }
