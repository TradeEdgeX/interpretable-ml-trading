from __future__ import annotations

from pathlib import Path

from scripts.pipeline.strategy_symbols import (
    format_symbol_csv,
    parse_symbol_csv,
    resolve_strategy_symbols,
)


def test_resolve_strategy_symbols_with_include_and_exclude(tmp_path: Path) -> None:
    strat_dir = tmp_path / "dual_add_trend"
    strat_dir.mkdir(parents=True)
    (strat_dir / "meta.yaml").write_text(
        """
strategy:
  symbol_include: [ETHUSDT, BTCUSDT, ADAUSDT]
  symbol_exclude: [ADAUSDT]
""".strip(),
        encoding="utf-8",
    )

    sel = resolve_strategy_symbols(
        strategy="dual_add_trend",
        base_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"],
        strategy_config_dir=strat_dir,
    )

    assert sel.include_symbols == ["ETHUSDT", "BTCUSDT", "ADAUSDT"]
    assert sel.exclude_symbols == ["ADAUSDT"]
    # Preserve base order after include/exclude filtering.
    assert sel.resolved_symbols == ["BTCUSDT", "ETHUSDT"]


def test_resolve_strategy_symbols_without_meta_keeps_base(tmp_path: Path) -> None:
    strat_dir = tmp_path / "bpc"
    strat_dir.mkdir(parents=True)
    sel = resolve_strategy_symbols(
        strategy="bpc",
        base_symbols=["BTCUSDT", "ETHUSDT"],
        strategy_config_dir=strat_dir,
    )
    assert sel.include_symbols == []
    assert sel.exclude_symbols == []
    assert sel.resolved_symbols == ["BTCUSDT", "ETHUSDT"]


def test_parse_and_format_symbol_csv_roundtrip() -> None:
    raw = "btcUSDT, ethusdt,,SOLUSDT "
    symbols = parse_symbol_csv(raw)
    assert symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert format_symbol_csv(symbols) == "BTCUSDT,ETHUSDT,SOLUSDT"
