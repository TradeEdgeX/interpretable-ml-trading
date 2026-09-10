"""Capital-denominated reports for research backtests.

The project has two historical PnL units:

* single-position event backtests: ``pnl_r`` is a raw R-multiple sum. Money needs
  an explicit ``risk_per_r`` assumption.
* multi-leg inventory backtests: ``pnl_per_capital`` is already normalized by the
  strategy capital bucket.

This module writes a small JSON/HTML report so research outputs can answer the
plain question: "if I start with 10,000 USD, what does this run imply?"
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

_COIN_M_ACCOUNTING = {
    "accounting_basis": "strategy_realized_usd",
    "accounting_basis_label": "Strategy realized USD (trade closes only)",
    "accounting_basis_note": (
        "Sums trade-level pnl_usd_realized on a fixed initial_capital base at each "
        "exit. Idle collateral is NOT mark-to-market between trades, so collateral "
        "beta is excluded from total_return."
    ),
    "related_metrics": {
        "wallet_mtm_equity": (
            "event_backtest equity_curve: wallet_coin × collateral_price + open "
            "unrealized, bar-level (includes collateral beta)."
        ),
        "combined_C": (
            "Matrix gate metric from coin_margin_core.compute_equity_from_trades: "
            "(1 + collateral_hold) × (1 + strategy_leg) − 1."
        ),
    },
}


def _apply_accounting_metadata(
    report: Dict[str, Any], *, margin_mode: str | None
) -> None:
    mode = str(margin_mode or "usd_m").lower()
    report["margin_mode"] = mode
    if mode == "coin_m":
        report.update(_COIN_M_ACCOUNTING)
        report["equity_curve_explanation"] = (
            "Strategy-realized equity: each trade adds pnl_usd_realized to a fixed "
            f"initial_capital base ({report.get('equity_curve_explanation', '')}). "
            "This curve excludes collateral mark-to-market between exits. Compare "
            "with event_backtest wallet_MTM equity_curve (includes beta) and matrix "
            "combined_C (post-process sleeve return)."
        )


def write_capital_report_from_trades(
    *,
    trades_path: str | Path,
    out_dir: str | Path,
    unit: str,
    title: str,
    initial_capital: float = 10_000.0,
    risk_per_r: float = 0.01,
    start_date: str = "",
    end_date: str = "",
    total_r: Optional[float] = None,
    compound_sizing: bool = True,
    n_symbols: Optional[int] = None,
    margin_mode: str | None = None,
) -> Dict[str, Any]:
    """Write ``capital_report.json/html`` from a trade CSV.

    Args:
        unit: ``capital_normalized`` for ``pnl_per_capital``; ``r_multiple`` for
            ``pnl_r``. ``r_multiple`` converts to money as
            ``initial_capital * risk_per_r * pnl_r``.
        n_symbols: When set (>1), ``initial_capital`` is total portfolio notional
            and each trade applies ``pnl_per_capital`` to ``initial_capital / n_symbols``.
    """
    trades_path = Path(trades_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "capital_report.json"
    html_path = out_dir / "capital_report.html"

    if not trades_path.exists():
        report = _empty_report(
            title=title,
            unit=unit,
            initial_capital=initial_capital,
            risk_per_r=risk_per_r,
            reason=f"trades file missing: {trades_path}",
        )
        _write_report(report, report_path, html_path)
        return report

    try:
        trades = pd.read_csv(trades_path)
    except pd.errors.EmptyDataError:
        trades = pd.DataFrame()
    if trades.empty:
        report = _empty_report(
            title=title,
            unit=unit,
            initial_capital=initial_capital,
            risk_per_r=risk_per_r,
            reason="no trades",
        )
        _write_report(report, report_path, html_path)
        return report

    time_col = "exit_time" if "exit_time" in trades.columns else None
    if time_col is None:
        report = _empty_report(
            title=title,
            unit=unit,
            initial_capital=initial_capital,
            risk_per_r=risk_per_r,
            reason="missing exit_time",
        )
        _write_report(report, report_path, html_path)
        return report

    if "pnl_usd_realized" in trades.columns:
        dollars = pd.to_numeric(trades["pnl_usd_realized"], errors="coerce").fillna(0.0)
        returns = dollars / max(float(initial_capital), 1e-12)
        unit_explanation = (
            "equity is computed from trade-level realized USDT PnL exported by "
            "the backtest; pnl_r / pnl_per_capital are kept as diagnostics."
        )
        if total_r is None:
            if "pnl_r" in trades.columns:
                effective_total_r = float(
                    pd.to_numeric(trades["pnl_r"], errors="coerce").fillna(0.0).sum()
                )
            elif "pnl_per_capital" in trades.columns:
                effective_total_r = float(
                    pd.to_numeric(trades["pnl_per_capital"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
            else:
                effective_total_r = float(returns.sum())
        else:
            effective_total_r = float(total_r)
    elif unit == "capital_normalized":
        pnl_col = "pnl_per_capital"
        if pnl_col not in trades.columns:
            report = _empty_report(
                title=title,
                unit=unit,
                initial_capital=initial_capital,
                risk_per_r=risk_per_r,
                reason="missing pnl_per_capital",
            )
            _write_report(report, report_path, html_path)
            return report
        returns = pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0)
        ns = n_symbols
        if ns is None and "symbol" in trades.columns:
            ns = int(trades["symbol"].nunique())
        if ns and ns > 1:
            # initial_capital is the total portfolio notional; each symbol owns a
            # 1/ns bucket, so a trade's pnl_per_capital scales by 1/ns at the
            # portfolio level.
            portfolio_returns = returns / float(ns)
            dollars = portfolio_returns * float(initial_capital)
            unit_explanation = (
                "total_r equals timeline portfolio pnl_per_capital (equal weight per "
                "symbol). Money uses initial_capital as total portfolio notional split "
                f"across {ns} symbols."
            )
        else:
            portfolio_returns = returns
            dollars = initial_capital * returns
            unit_explanation = (
                "total_r equals sum(pnl_per_capital): capital-bucket-normalized net "
                "return, not classic per-trade R. 1.0 means +100% on the strategy "
                "capital bucket; 1.1355 means about +113.55%."
            )
        effective_total_r = (
            float(total_r) if total_r is not None else float(portfolio_returns.sum())
        )
    elif unit == "r_multiple":
        pnl_col = "pnl_r"
        if pnl_col not in trades.columns:
            report = _empty_report(
                title=title,
                unit=unit,
                initial_capital=initial_capital,
                risk_per_r=risk_per_r,
                reason="missing pnl_r",
            )
            _write_report(report, report_path, html_path)
            return report
        r_values = pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0)
        unit_explanation = (
            "total_r equals raw sum(pnl_r): cumulative R-multiples. Money is "
            "estimated with initial_capital * risk_per_r * pnl_r."
        )
        effective_total_r = float(r_values.sum() if total_r is None else total_r)
        dollars = initial_capital * float(risk_per_r) * r_values
        returns = float(risk_per_r) * r_values
    else:
        raise ValueError(f"unsupported capital report unit: {unit}")

    df = pd.DataFrame(
        {
            "exit_time": pd.to_datetime(trades[time_col], utc=True, errors="coerce"),
            "pnl_usd_est": dollars,
        }
    ).dropna(subset=["exit_time"])
    df = df.sort_values("exit_time")
    if df.empty:
        report = _empty_report(
            title=title,
            unit=unit,
            initial_capital=initial_capital,
            risk_per_r=risk_per_r,
            reason="no valid exit_time rows",
        )
        _write_report(report, report_path, html_path)
        return report

    df["equity"] = float(initial_capital) + df["pnl_usd_est"].cumsum()
    df["peak"] = df["equity"].cummax()
    df["drawdown_usd"] = df["equity"] - df["peak"]
    df["drawdown_pct"] = df["drawdown_usd"] / df["peak"].replace(0.0, pd.NA)
    equity_curve = _sample_equity_curve(df)

    start_ts = (
        pd.Timestamp(start_date, tz="UTC") if start_date else df["exit_time"].min()
    )
    end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else df["exit_time"].max()
    years = max((end_ts - start_ts).days / 365.25, 1.0 / 365.25)
    final_capital = float(df["equity"].iloc[-1])
    ic = float(initial_capital)
    total_return = final_capital / ic - 1.0
    if final_capital > 0 and ic > 0:
        try:
            cagr = math.pow(final_capital / ic, 1.0 / years) - 1.0
        except (OverflowError, ValueError):
            cagr = float("inf")
    else:
        cagr = -1.0

    report = {
        "title": title,
        "initial_capital": float(initial_capital),
        "final_capital": final_capital,
        "estimated_profit_usd": final_capital - float(initial_capital),
        "total_return": total_return,
        "cagr": cagr,
        "cagr_explanation": (
            "CAGR is compound annual growth rate: the constant yearly return that "
            "turns initial_capital into final_capital over the measured period."
        ),
        "max_drawdown_usd": float(df["drawdown_usd"].min()),
        "max_drawdown_pct": float(df["drawdown_pct"].min()),
        "trades": int(len(df)),
        "start": str(start_ts),
        "end": str(end_ts),
        "years": float(years),
        "unit": unit,
        "unit_explanation": unit_explanation,
        "total_r": float(effective_total_r),
        "risk_per_r": float(risk_per_r),
        "assumptions": {
            "event_r_multiple": "1R risks risk_per_r of initial_capital; default 1%.",
            "multi_leg_capital_normalized": "pnl_per_capital is applied to initial_capital.",
            "compounding": (
                "Position sizing uses current equity × risk_per_slot after each close "
                "(compound sizing; matches constitution)."
                if compound_sizing
                else "Equity curve is additive per trade; position sizing frozen at "
                "initial_capital × risk_per_slot (legacy fixed-base)."
            ),
            "excluded": "Funding/liquidation/margin usage beyond modeled trade PnL unless present in the backtest trades.",
        },
        "trades_path": str(trades_path),
        "equity_curve_explanation": (
            "Equity curve applies each trade's modeled PnL to a fixed initial "
            "capital base. For capital_normalized multi-leg reports, each "
            "pnl_per_capital increment is added as initial_capital * "
            "pnl_per_capital; for r_multiple reports, pnl_r is converted through "
            "risk_per_r."
        ),
        "equity_curve": equity_curve,
        "equity_tail": [
            {
                "exit_time": str(row.exit_time),
                "equity": float(row.equity),
                "drawdown_pct": float(row.drawdown_pct),
            }
            for row in df.tail(20).itertuples(index=False)
        ],
    }
    _apply_accounting_metadata(report, margin_mode=margin_mode)
    _write_report(report, report_path, html_path)
    return report


def _empty_report(
    *,
    title: str,
    unit: str,
    initial_capital: float,
    risk_per_r: float,
    reason: str,
) -> Dict[str, Any]:
    return {
        "title": title,
        "initial_capital": float(initial_capital),
        "final_capital": float(initial_capital),
        "estimated_profit_usd": 0.0,
        "total_return": 0.0,
        "cagr": 0.0,
        "max_drawdown_usd": 0.0,
        "max_drawdown_pct": 0.0,
        "trades": 0,
        "unit": unit,
        "risk_per_r": float(risk_per_r),
        "reason": reason,
    }


def _write_report(report: Dict[str, Any], json_path: Path, html_path: Path) -> None:
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    html_path.write_text(_render_html(report), encoding="utf-8")


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except Exception:
        return "N/A"


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "N/A"


def _sample_equity_curve(
    df: pd.DataFrame, *, max_points: int = 240
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    if len(df) <= max_points:
        sampled = df
    else:
        step = max(1, len(df) // max_points)
        sampled = df.iloc[::step].copy()
        if sampled.index[-1] != df.index[-1]:
            sampled = pd.concat([sampled, df.tail(1)], axis=0)
    return [
        {
            "exit_time": str(row.exit_time),
            "equity": float(row.equity),
            "drawdown_pct": float(row.drawdown_pct),
        }
        for row in sampled.itertuples(index=False)
    ]


def _render_equity_svg(report: Dict[str, Any]) -> str:
    curve = report.get("equity_curve") or []
    if len(curve) < 2:
        return '<p class="note">Equity curve unavailable: not enough points.</p>'

    width = 920
    height = 280
    pad_l = 64
    pad_r = 24
    pad_t = 24
    pad_b = 42
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    values = [float(p.get("equity", 0.0) or 0.0) for p in curve]
    ymin = min(values)
    ymax = max(values)
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0

    def _xy(i: int, v: float) -> tuple[float, float]:
        x = pad_l + plot_w * (i / max(1, len(values) - 1))
        y = pad_t + plot_h * (1.0 - ((v - ymin) / (ymax - ymin)))
        return x, y

    points = " ".join(
        f"{x:.1f},{y:.1f}" for i, v in enumerate(values) for x, y in [_xy(i, v)]
    )
    start_label = _money(values[0])
    end_label = _money(values[-1])
    low_label = _money(ymin)
    high_label = _money(ymax)
    first_time = str(curve[0].get("exit_time", ""))
    last_time = str(curve[-1].get("exit_time", ""))
    return f"""
  <svg class="equity-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Equity curve">
    <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="#d0d7de"/>
    <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#d0d7de"/>
    <text x="8" y="{pad_t + 5}" class="axis-label">{high_label}</text>
    <text x="8" y="{height - pad_b}" class="axis-label">{low_label}</text>
    <polyline fill="none" stroke="#0969da" stroke-width="2.5" points="{points}"/>
    <circle cx="{_xy(0, values[0])[0]:.1f}" cy="{_xy(0, values[0])[1]:.1f}" r="3" fill="#0969da"/>
    <circle cx="{_xy(len(values) - 1, values[-1])[0]:.1f}" cy="{_xy(len(values) - 1, values[-1])[1]:.1f}" r="3" fill="#1a7f37"/>
    <text x="{pad_l}" y="{height - 12}" class="axis-label">{first_time}</text>
    <text x="{width - pad_r}" y="{height - 12}" text-anchor="end" class="axis-label">{last_time}</text>
    <text x="{pad_l + 8}" y="{pad_t + 18}" class="chart-note">start {start_label}</text>
    <text x="{width - pad_r - 8}" y="{pad_t + 18}" text-anchor="end" class="chart-note">end {end_label}</text>
  </svg>
"""


def _render_html(report: Dict[str, Any]) -> str:
    related = report.get("related_metrics") or {}
    related_html = ""
    if related:
        related_html = (
            f'<p class="note"><b>Related metrics:</b> '
            f"{json.dumps(related, ensure_ascii=False)}</p>"
        )
    basis_label = report.get(
        "accounting_basis_label", "Trade-level modeled PnL on fixed initial capital"
    )
    basis_note = report.get(
        "accounting_basis_note", report.get("equity_curve_explanation", "")
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{report.get('title', 'Capital Report')}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .note {{ color: #555; max-width: 980px; line-height: 1.45; }}
    .equity-chart {{ width: 100%; max-width: 980px; height: auto; border: 1px solid #d0d7de; border-radius: 6px; background: #fff; }}
    .axis-label {{ font-size: 12px; fill: #57606a; }}
    .chart-note {{ font-size: 12px; fill: #24292f; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>{report.get('title', 'Capital Report')}</h1>
  <p class="note"><b>Accounting basis:</b> {basis_label}</p>
  <p class="note">{basis_note}</p>
  <table>
    <tr><th>Initial capital</th><td>{_money(report.get('initial_capital'))}</td></tr>
    <tr><th>Final capital</th><td>{_money(report.get('final_capital'))}</td></tr>
    <tr><th>Estimated profit</th><td>{_money(report.get('estimated_profit_usd'))}</td></tr>
    <tr><th>Total return</th><td>{_pct(report.get('total_return'))}</td></tr>
    <tr><th>CAGR</th><td>{_pct(report.get('cagr'))}</td></tr>
    <tr><th>Max drawdown</th><td>{_money(report.get('max_drawdown_usd'))} ({_pct(report.get('max_drawdown_pct'))})</td></tr>
    <tr><th>Trades</th><td>{report.get('trades')}</td></tr>
    <tr><th>Total R</th><td>{float(report.get('total_r', 0.0) or 0.0):.6f}</td></tr>
  </table>
  <h2>Equity Curve</h2>
  {_render_equity_svg(report)}
  <p class="note"><b>Equity curve:</b> {report.get('equity_curve_explanation', '')}</p>
  <h2>Definitions</h2>
  <p class="note"><b>CAGR:</b> {report.get('cagr_explanation', '')}</p>
  <p class="note"><b>Total R unit:</b> {report.get('unit_explanation', '')}</p>
  <p class="note"><b>Assumptions:</b> {json.dumps(report.get('assumptions', {}), ensure_ascii=False)}</p>
  {related_html}
</body>
</html>
"""
