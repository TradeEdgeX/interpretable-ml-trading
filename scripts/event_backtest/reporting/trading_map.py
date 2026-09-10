from __future__ import annotations

import html as html_lib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.event_backtest._bootstrap import logger
from scripts.event_backtest.results import BacktestResult
from scripts.event_backtest.types.trade import ClosedTrade
from scripts.rolling_trading_map_indicators import weekly_ema200
from src.data_tools.data_handler import DataHandler

try:
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import (
        HoverTool,
        Div,
        Tabs,
        TabPanel,
        FixedTicker,
        ColumnDataSource,
        Span,
        Toggle,
        CustomJS,
    )
    from bokeh.layouts import column as bk_column
    from bokeh.layouts import row as bk_row
    from bokeh.resources import INLINE as BK_RESOURCES
    from bokeh.embed import file_html as bk_file_html

    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    bk_row = None  # type: ignore

# Default chart bar size for B-stack backtests (120T); matches FeatureStore / simulate_symbol.
DEFAULT_TRADING_MAP_BAR_FREQ = "2h"
# SR overlay / trade-list: BTC first (extend via --map-sr-symbols).
DEFAULT_MAP_SR_SYMBOLS: Tuple[str, ...] = ("BTCUSDT",)
_SR_SOURCE_COLORS = {"L1": "#16a34a", "L3": "#9333ea", "entry": "#64748b"}
_COLOR_WIN_HEX = "#26a69a"
_COLOR_LOSS_HEX = "#ef5350"


def _compute_sr_overlay_series(
    ohlc: pd.DataFrame, *, l1_lookback: int = 20
) -> pd.DataFrame:
    """Public court has no SR overlay series."""
    del l1_lookback
    return pd.DataFrame(index=getattr(ohlc, "index", None))


def _srb_trade_list_html(trades: Sequence[ClosedTrade], symbol: str) -> str:
    """HTML table: which true-SR level each SRB trade used."""
    rows = [
        t
        for t in trades
        if str(t.symbol).upper() == str(symbol).upper()
        and str(t.archetype or "").lower().startswith("srb")
    ]
    if not rows:
        return (
            f"<p style='color:#64748b;font-size:12px'>"
            f"No SRB trades for {html_lib.escape(symbol)} in this tab.</p>"
        )
    rows = sorted(rows, key=lambda t: pd.Timestamp(t.entry_time))
    head = (
        "<div style='max-height:280px;overflow:auto;margin:6px 0 12px 0'>"
        "<table style='border-collapse:collapse;font-size:11px;width:100%'>"
        "<thead><tr style='background:#f1f5f9;text-align:left'>"
        "<th style='padding:4px 6px'>#</th>"
        "<th style='padding:4px 6px'>Entry</th>"
        "<th style='padding:4px 6px'>Side</th>"
        "<th style='padding:4px 6px'>Entry px</th>"
        "<th style='padding:4px 6px'>SR src</th>"
        "<th style='padding:4px 6px'>true_sr</th>"
        "<th style='padding:4px 6px'>stop</th>"
        "<th style='padding:4px 6px'>L1 sup/res</th>"
        "<th style='padding:4px 6px'>L3 lo/hi</th>"
        "<th style='padding:4px 6px'>R</th>"
        "<th style='padding:4px 6px'>Exit</th>"
        "<th style='padding:4px 6px'>Add?</th>"
        "</tr></thead><tbody>"
    )
    body_parts: List[str] = []
    for i, t in enumerate(rows, 1):
        src = str(getattr(t, "srb_true_sr_source", "") or "") or "—"
        src_c = _SR_SOURCE_COLORS.get(src, "#334155")
        et = pd.Timestamp(t.entry_time)
        et_s = et.strftime("%Y-%m-%d %H:%M") if pd.notna(et) else ""
        true_sr = float(getattr(t, "srb_true_sr_level", 0.0) or 0.0)
        stop_src = str(getattr(t, "sizing_stop_source", "") or "") or "atr"
        l1s = float(getattr(t, "srb_l1_support", 0.0) or 0.0)
        l1r = float(getattr(t, "srb_l1_resistance", 0.0) or 0.0)
        l3l = float(getattr(t, "srb_l3_lower", 0.0) or 0.0)
        l3u = float(getattr(t, "srb_l3_upper", 0.0) or 0.0)
        rc = _COLOR_WIN_HEX if t.pnl_r > 0 else _COLOR_LOSS_HEX
        body_parts.append(
            "<tr style='border-top:1px solid #e2e8f0'>"
            f"<td style='padding:3px 6px'>{i}</td>"
            f"<td style='padding:3px 6px;white-space:nowrap'>{html_lib.escape(et_s)}</td>"
            f"<td style='padding:3px 6px'>{html_lib.escape(str(t.side))}</td>"
            f"<td style='padding:3px 6px'>{t.entry_price:.2f}</td>"
            f"<td style='padding:3px 6px;color:{src_c};font-weight:600'>"
            f"{html_lib.escape(src)}</td>"
            f"<td style='padding:3px 6px'>{true_sr:.2f}</td>"
            f"<td style='padding:3px 6px'>{html_lib.escape(stop_src)}</td>"
            f"<td style='padding:3px 6px'>{l1s:.2f} / {l1r:.2f}</td>"
            f"<td style='padding:3px 6px'>{l3l:.2f} / {l3u:.2f}</td>"
            f"<td style='padding:3px 6px;color:{rc}'>{t.pnl_r:+.2f}</td>"
            f"<td style='padding:3px 6px'>{html_lib.escape(str(t.exit_reason))}</td>"
            f"<td style='padding:3px 6px'>{'Y' if t.is_add_position else ''}</td>"
            "</tr>"
        )
    note = (
        "<p style='color:#475569;font-size:11px;margin:0 0 4px 0'>"
        f"<b>{html_lib.escape(symbol)}</b> SRB trade list — "
        "gate=L3 age_decay（时间窗，非贴价）；"
        "<span style='color:#16a34a'>L1</span>=swing20 meta/true_sr，"
        "<span style='color:#9333ea'>L3</span>=wide_sr240，"
        "<span style='color:#64748b'>entry</span>=两侧缺失。"
        " stop 列=sizing_stop_source（现网多为 atr / initial_r，非 L1 线）；"
        " Toggle 控制主图水平 SR；持仓虚线=true_sr 锚点。"
        "</p>"
    )
    return note + head + "".join(body_parts) + "</tbody></table></div>"


def _compute_fade_pierce_events_from_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Detector pierce→recross pairs on chart OHLC (same R/B/V as prod fade)."""
    empty = pd.DataFrame(
        columns=["pierce_time", "recross_time", "fade_side", "level", "pierce_y"]
    )
    if ohlc is None or ohlc.empty:
        return empty
    need = ("open", "high", "low", "close", "volume")
    if any(c not in ohlc.columns for c in need):
        return empty
    try:
        from src.features.time_series.baseline_features import (
            compute_atr_from_series,
            compute_volume_ratio_pct_from_series,
        )
        from src.features.time_series.chop_largebar_fade_features import (
            compute_fade_pierce_recross_events,
        )
    except Exception:
        return empty
    atr = compute_atr_from_series(
        high=ohlc["high"], low=ohlc["low"], close=ohlc["close"]
    )["atr"]
    volp = compute_volume_ratio_pct_from_series(volume=ohlc["volume"])[
        "volume_ratio_pct"
    ]
    return compute_fade_pierce_recross_events(
        open=ohlc["open"],
        high=ohlc["high"],
        low=ohlc["low"],
        close=ohlc["close"],
        atr=atr,
        volume_ratio_pct=volp,
    )


def _should_draw_fade_pierce_overlay(strat_filter: Optional[str]) -> bool:
    """All tab + fade tab only (not SRB-only / other strategy tabs)."""
    if strat_filter is None:
        return True
    s = str(strat_filter).strip().lower()
    return any(tok in s for tok in ("largebar_fade", "chop_largebar_fade"))


def _add_fade_pierce_overlay(
    p: Any,
    *,
    ohlc_full: pd.DataFrame,
    view_start: pd.Timestamp,
    view_end: pd.Timestamp,
) -> Tuple[int, List[Any]]:
    """Diamonds on pierce bars + level lines to recross.

    Returns ``(event_count, glyph_renderers)`` for optional UI toggles.
    """
    renderers: List[Any] = []
    ev = _compute_fade_pierce_events_from_ohlc(ohlc_full)
    if ev.empty:
        return 0, renderers
    pt = pd.to_datetime(ev["pierce_time"], utc=True)
    rt = pd.to_datetime(ev["recross_time"], utc=True)
    vs = pd.Timestamp(view_start)
    ve = pd.Timestamp(view_end)
    if vs.tzinfo is None:
        vs = vs.tz_localize("UTC")
    else:
        vs = vs.tz_convert("UTC")
    if ve.tzinfo is None:
        ve = ve.tz_localize("UTC")
    else:
        ve = ve.tz_convert("UTC")
    # Keep if pierce or recross intersects the backtest view.
    mask = ((pt >= vs) & (pt <= ve)) | ((rt >= vs) & (rt <= ve))
    ev = ev.loc[mask].reset_index(drop=True)
    if ev.empty:
        return 0, renderers

    long_ev = ev[ev["fade_side"].astype(float) > 0]
    short_ev = ev[ev["fade_side"].astype(float) < 0]

    def _draw_side(batch: pd.DataFrame, *, fade_long: bool) -> None:
        if batch.empty:
            return
        # Bear pierce → await long: amber; bull pierce → await short: violet
        fill = "#f59e0b" if fade_long else "#a855f7"
        label = "FADE pierce→long" if fade_long else "FADE pierce→short"
        src = ColumnDataSource(
            {
                "x": pd.to_datetime(batch["pierce_time"], utc=True),
                "y": batch["pierce_y"].astype(float),
                "level": batch["level"].astype(float),
                "recross": pd.to_datetime(batch["recross_time"], utc=True),
                "side": ["LONG" if fade_long else "SHORT"] * len(batch),
            }
        )
        r = p.scatter(
            x="x",
            y="y",
            source=src,
            marker="diamond",
            size=12,
            fill_color=fill,
            line_color="#1e293b",
            line_width=1.2,
            fill_alpha=0.95,
            legend_label=label,
        )
        renderers.append(r)
        p.add_tools(
            HoverTool(
                renderers=[r],
                tooltips=[
                    ("Pierce", "@x{%F %H:%M}"),
                    ("Recross", "@recross{%F %H:%M}"),
                    ("Fade", "@side"),
                    ("Level", "@level{0.4f}"),
                ],
                formatters={"@x": "datetime", "@recross": "datetime"},
            )
        )
        r_lvl = p.multi_line(
            xs=[
                [pd.Timestamp(a), pd.Timestamp(b)]
                for a, b in zip(batch["pierce_time"], batch["recross_time"])
            ],
            ys=[[float(lv), float(lv)] for lv in batch["level"]],
            line_color=fill,
            line_dash="dotted",
            line_alpha=0.65,
            line_width=1.4,
            legend_label=f"{label} level",
        )
        renderers.append(r_lvl)

    _draw_side(long_ev, fade_long=True)
    _draw_side(short_ev, fade_long=False)
    return int(len(ev)), renderers


def _resample_bars(
    bars_1min: pd.DataFrame, freq: str = DEFAULT_TRADING_MAP_BAR_FREQ
) -> pd.DataFrame:
    """1min bars → 指定 timeframe OHLCV"""
    ohlc = (
        bars_1min.resample(freq)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    if "volume" in bars_1min.columns:
        ohlc["volume"] = bars_1min["volume"].resample(freq).sum()
    return ohlc


def _rolling_tp_vwap(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """Rolling typical-price VWAP: sum(tp*vol)/sum(vol) over ``window`` bars."""
    h = ohlc["high"].astype(float)
    l = ohlc["low"].astype(float)
    c = ohlc["close"].astype(float)
    tp = (h + l + c) / 3.0
    if "volume" in ohlc.columns:
        vol = ohlc["volume"].astype(float).clip(lower=0.0)
    else:
        vol = pd.Series(1.0, index=ohlc.index)
    n = len(ohlc)
    w = max(2, min(int(window), n))
    min_p = max(3, min(w // 10, max(50, w // 20)))
    num = (tp * vol).rolling(window=w, min_periods=min_p).sum()
    den = vol.rolling(window=w, min_periods=min_p).sum()
    out = num / den.replace(0, np.nan)
    return out


def _merge_1min_for_chart(
    sym: str,
    bars_1min: pd.DataFrame,
    data_path: Optional[str],
    extra_months: int,
) -> pd.DataFrame:
    """1m OHLCV from (test_start − extra_months) through test_end; merge with backtest bars."""
    if bars_1min is None or bars_1min.empty:
        return bars_1min
    b = bars_1min.copy()
    b.index = pd.to_datetime(b.index, utc=True)
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in b.columns]
    if not cols or "close" not in cols:
        return bars_1min
    b = b[cols]
    if not data_path or int(extra_months) <= 0:
        return b
    dp = Path(data_path)
    if not dp.is_dir():
        return b
    idx_min = b.index.min()
    idx_max = b.index.max()
    start_ts = idx_min - pd.DateOffset(months=int(extra_months))
    start_s = start_ts.strftime("%Y-%m-%d")
    end_day = pd.Timestamp(idx_max)
    if getattr(end_day, "tz", None) is not None:
        end_day = end_day.tz_convert("UTC")
    end_s = (end_day.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        dh = DataHandler(str(dp))
        ext = dh.load_ohlcv(
            symbol=sym, timeframe="1T", start_date=start_s, end_date=end_s
        )
    except Exception as e:
        logger.warning("trading map: extended 1m load failed %s: %s", sym, e)
        return b
    if ext is None or ext.empty:
        return b
    ext = ext.sort_index()
    ext.index = pd.to_datetime(ext.index, utc=True)
    ec = [c for c in ("open", "high", "low", "close", "volume") if c in ext.columns]
    if not ec or "close" not in ec:
        return b
    ext = ext[ec]
    merged = pd.concat([ext, b]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def generate_trading_map_html(
    result: BacktestResult,
    output_path: str,
    bar_freq: str = DEFAULT_TRADING_MAP_BAR_FREQ,
    compare_trades_csv: Optional[str] = None,
    *,
    data_path: Optional[str] = None,
    map_extra_months: int = 12,
    map_vwap_window_bars: int = 1200,
    map_long_ema_span: int = 1200,
    map_sr_symbols: Optional[Sequence[str]] = None,
) -> str:
    """生成 K线 + 交易标记 HTML 交易地图 (多策略分 Tab)。

    可选从 ``data_path`` 向前多取 ``map_extra_months`` 月 1m 数据，仅在 **计算** VWAP / **长窗 EMA** 时使用；
    **图上 X 轴只覆盖本次回测窗口**（``result.bars_1min`` 对应区间），不把向前扩展的历史整段画出来。
    主图叠画滚动典型价 VWAP，以及 ``close`` 上 span=``map_long_ema_span`` 的 EMA 价格线。
    （地图 K 线由 1m 按策略主周期重采样，如 120T→2h；EMA 与该重采样后的 ``close`` 同源。若需与
    FeatureStore **同一张 120T 表**的 ``ema_1200`` 像素级对齐，请用向量回测生成的 ``trading_map_<strategy>.html``。）

    ``map_sr_symbols``（默认 BTCUSDT）：主图可切换叠 L1 swing / L3 wide_sr 水平线，
    并在图下给出 SRB 成交列表（true_sr 来源 L1/L3/entry）。

    可视化规则:
      入场填充色   = 方向 (多=绿 / 空=红), 描边色 = 策略 (BPC=蓝 / FER=紫 / ME=橙)
      标记形状     = 方向+加仓 (△=多头, ▽=空头, ◇=加仓多, ◈=加仓空)
      连接线颜色   = 盈亏 (绿=盈利, 红=亏损)
      出场标记颜色 = 盈亏 (绿=盈利, 红=亏损)
    面板布局:
      顶部: 执行层(execution.yaml) vs 宪法层(constitution.yaml)说明、漏斗计数摘要、
           仿真组合权益曲线+回撤（含 Kill Switch 触发竖线）；
      Tabs: All | BPC | FER | ME (每个 Tab 内按 symbol 纵向堆叠；
           漏斗附图中 KS 记号 = 当周期间隔内 constitutional kill 阻断新开仓)。
    """
    if not BOKEH_AVAILABLE:
        logger.warning("❌ Bokeh 未安装, 无法生成交易地图. pip install bokeh")
        return ""

    symbols = sorted(set(result.per_symbol.keys()) | set(result.bars_1min.keys()))
    if not symbols:
        logger.warning("❌ 无 symbol（无成交且无 bars_1min）, 无法生成交易地图")
        return ""

    if map_sr_symbols is None:
        _sr_syms = {s.upper() for s in DEFAULT_MAP_SR_SYMBOLS}
    else:
        _sr_syms = {str(s).upper().strip() for s in map_sr_symbols if str(s).strip()}

    def _build_portfolio_overview_above_tabs(result: BacktestResult) -> list:
        """资金曲线 / 回撤 + 宪法与 execution 语义说明（置于 Tabs 上方）。"""
        blocks: list = []
        exe_note = (
            "<b>执行层</b>：<code>archetypes/execution.yaml</code> "
            "由模拟器在每根 1m bar 上应用（初始止损倍数、追踪、保本、结构性出场等）；"
            "决定<strong>开仓之后</strong>的价格路径与平仓，不改变宪法是否允许新开仓。"
        )
        leg_note = (
            "<b>宪法层</b>：<code>constitution.yaml</code> "
            "由 <code>ConstitutionExecutor</code> 在每根时间线上评估（回撤/日损等 → Kill Switch）；"
            "若为「暂停」，PCM 虽已排序出 intent，<strong>新开仓会被跳过</strong>。"
            "顶部权益图红虚竖线、各 symbol 漏斗图 y=KS 记号均表示该时间评估下 Kill 生效。"
        )
        axis_note = (
            "<b>读图</b>：蓝线为<strong>USDT权益</strong>（按成交已实现 PnL 更新，默认仅在<strong>平仓</strong>时阶跃）；"
            "若全程未盈利超起点，则纵轴峰值接近初始现金锚点；不是「被坐标轴挡住」。"
            "Kill 生效后常见<strong>权益走平</strong>：无新仓、也无平仓结算，曲线不再变化，但时间线仍走到回测结束。"
        )
        lines = [exe_note, leg_note, axis_note]

        ces = getattr(result, "constitution_execution_summary", None) or {}
        if ces:
            ks_on = ces.get("kill_switch_enabled")
            lines.append(
                "<b>本回测宪法阈值</b>："
                f"Kill={'开' if ks_on else '关'} | "
                f"risk/slot≤{float(ces.get('risk_per_slot', 0)):g} | "
                f"max_dd≤{float(ces.get('max_dd_limit', 0)):g} "
                f"daily≤{float(ces.get('daily_loss_limit', 0)):g} "
                f"weekly≤{float(ces.get('weekly_loss_limit', 0)):g} "
                f"monthly≤{float(ces.get('monthly_loss_limit', 0)):g} | "
                f"cooldown={int(ces.get('cooldown_minutes') or 0)}min "
                f"(tz={ces.get('daily_reset_timezone')!s}) "
                f"<code>{Path(str(ces.get('constitution_yaml') or '')).name}</code>"
            )

        funnel = getattr(result, "funnel", None) or {}
        top_lines: List[str] = []
        try:
            for k, v in sorted(
                funnel.items(), key=lambda kv: (-abs(float(kv[1] or 0)), str(kv[0]))
            ):
                iv = float(v if v is not None else 0)
                if abs(iv) < 1e-9:
                    continue
                if iv == int(iv):
                    top_lines.append(f"{k}={int(iv)}")
                else:
                    top_lines.append(f"{k}={iv:.4g}")
                if len(top_lines) >= 14:
                    break
        except (TypeError, ValueError):
            top_lines = []
        if top_lines:
            lines.append("<b>漏斗计数（非零）</b>：" + "；".join(top_lines))

        blocks.append(
            Div(
                text="<p style='max-width:1400px;font-size:13px;line-height:1.5'>"
                + "<br/><br/>".join(lines)
                + "</p>",
                width=1400,
            )
        )

        eq_series = getattr(result, "equity_curve", None) or []
        ts_series = getattr(result, "equity_curve_ts", None) or []

        if len(eq_series) >= 2 and len(ts_series) == len(eq_series):
            t_ix = pd.to_datetime(ts_series, utc=True)
            eq_arr = np.asarray(eq_series, dtype=float)
            peak = np.maximum.accumulate(eq_arr)
            dd_arr = np.where(peak > 0, (peak - eq_arr) / peak, 0.0)

            p_eq = bk_figure(
                title=(
                    "组合权益曲线（USDT 口径；优先使用成交已实现 PnL，"
                    "每次平仓结算后更新。红虚竖线≈Kill Switch 触发时刻）"
                ),
                x_axis_type="datetime",
                width=1400,
                height=240,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            src_eq = ColumnDataSource(
                {
                    "t": t_ix,
                    "equity": eq_arr,
                    "dd_pct": dd_arr * 100.0,
                }
            )
            p_eq.line("t", "equity", source=src_eq, line_width=2, color="#2563eb")
            p_eq.add_tools(
                HoverTool(
                    tooltips=[
                        ("Time", "@t{%F %H:%M}"),
                        ("Equity ($)", "@equity{0}"),
                        ("Drawdown%", "@dd_pct{0.2f}"),
                    ],
                    formatters={"@t": "datetime"},
                )
            )
            p_eq.grid.grid_line_alpha = 0.25
            p_eq.yaxis.axis_label = "Equity ($)"

            ks = getattr(result, "kill_switch_stats", None) or {}
            for trig in ks.get("triggers", []) or []:
                t_raw = trig.get("timestamp")
                if not t_raw:
                    continue
                try:
                    tx = pd.Timestamp(t_raw)
                    if tx.tzinfo is None:
                        tx = tx.tz_localize("UTC")
                    x_ms = float(tx.value / 1e6)
                except Exception:
                    continue
                p_eq.add_layout(
                    Span(
                        location=x_ms,
                        dimension="height",
                        line_color="#dc2626",
                        line_width=2,
                        line_alpha=0.55,
                        line_dash="dashed",
                    )
                )

            blocks.append(p_eq)

            # ── 累计已实现 R（按平仓时刻阶跃，与美元权益同源不同单位）──
            if getattr(result, "trades", None):
                _tr_sorted = sorted(
                    result.trades, key=lambda x: getattr(x, "exit_time", None) or ""
                )
                _rx: list = []
                _ry: list = []
                _cum_r = 0.0
                for _ti, _trade in enumerate(_tr_sorted):
                    _et = getattr(_trade, "exit_time", None)
                    if _et is None:
                        continue
                    _tsx = pd.Timestamp(_et)
                    if _tsx.tzinfo is None:
                        _tsx = _tsx.tz_localize("UTC")
                    _pr = float(getattr(_trade, "pnl_r", 0.0) or 0.0)
                    # 平仓前水平线段 → 平仓后跳变（阶梯）
                    if not _rx:
                        _rx.extend([_tsx, _tsx])
                        _ry.extend([0.0, _cum_r + _pr])
                    else:
                        _rx.append(_tsx)
                        _ry.append(_cum_r)
                        _rx.append(_tsx)
                        _ry.append(_cum_r + _pr)
                    _cum_r += _pr
                if _rx:
                    p_r = bk_figure(
                        title="累计已实现 R（每笔平仓计入；与上图美元权益涨跌方向一致）",
                        x_axis_type="datetime",
                        width=1400,
                        height=150,
                        x_range=p_eq.x_range,
                        tools="pan,wheel_zoom,box_zoom,reset,save",
                    )
                    p_r.line(_rx, _ry, line_width=1.8, color="#0d9488")
                    p_r.yaxis.axis_label = "Σ R"
                    p_r.grid.grid_line_alpha = 0.25
                    blocks.append(p_r)

            p_dd = bk_figure(
                title="回撤（峰值权益口径，百分比）",
                x_axis_type="datetime",
                width=1400,
                height=160,
                x_range=p_eq.x_range,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            p_dd.line(t_ix, dd_arr * 100.0, line_width=1.6, color="#b91c1c")
            p_dd.yaxis.axis_label = "DD %"
            p_dd.grid.grid_line_alpha = 0.25
            blocks.append(p_dd)
        elif len(eq_series) >= 2:
            blocks.append(
                Div(
                    text="<p style='color:#92400e'>权益曲线时间点未对齐 equity_curve_ts，跳过资金图。</p>",
                    width=1400,
                )
            )

        return blocks

    # ── 颜色方案 ──
    _STRAT_COLORS: dict = {
        "bpc": "#3274D9",  # 蓝
        "fer": "#B877D9",  # 紫
        "me": "#FF9830",  # 橙
        "me-long": "#FF9830",  # 橙（别名）
        "lv": "#73BF69",  # 绿
    }
    _STRAT_COLOR_DEFAULT = "#aaaaaa"
    _COLOR_WIN = "#26a69a"  # 盈利 绿
    _COLOR_LOSS = "#ef5350"  # 亏损 红
    _COLOR_UP = "#26a69a"  # K线 阳
    _COLOR_DOWN = "#ef5350"  # K线 阴
    # 入场填充: 多空分离 (与 K 线涨跌绿红区分略提高饱和度)
    _ENTRY_FILL_LONG = "#2e7d32"
    _ENTRY_FILL_SHORT = "#c62828"

    # ── 标记映射 ──
    _MARKER_MAP = {
        ("LONG", False): "triangle",  # △ 多头入场
        ("SHORT", False): "inverted_triangle",  # ▽ 空头入场
        ("LONG", True): "diamond",  # ◇ 加仓多
        ("SHORT", True): "diamond_cross",  # ◈ 加仓空
    }

    def _entry_fill_for_side(side: str) -> str:
        return _ENTRY_FILL_LONG if str(side).upper() == "LONG" else _ENTRY_FILL_SHORT

    _FREQ_MS = {
        "15min": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
    }
    bar_w = _FREQ_MS.get(bar_freq, _FREQ_MS["2h"]) * 0.6

    def _arch_family(name: str) -> str:
        s = str(name or "").lower().strip()
        return s.split("-")[0] if s else ""

    def _strat_color(archetype: str) -> str:
        key = str(archetype).lower()
        if key in _STRAT_COLORS:
            return _STRAT_COLORS[key]
        fam = _arch_family(key)
        return _STRAT_COLORS.get(fam, _STRAT_COLOR_DEFAULT)

    # ── 所有出现的 archetype ──
    all_archetypes = sorted(set(t.archetype for t in result.trades if t.archetype))
    _has_fade_arch = any(
        any(tok in str(a).lower() for tok in ("largebar_fade", "chop_largebar_fade"))
        for a in all_archetypes
    ) or any(
        tok in str(getattr(result, "strategy", "") or "").lower()
        for tok in ("largebar_fade", "chop_largebar_fade")
    )

    # ── 构建单个 symbol K线图 ──
    # ── 加载对比交易（向量回测等）──
    _cmp_by_sym: dict = {}
    if compare_trades_csv and Path(compare_trades_csv).exists():
        try:
            _cdf = pd.read_csv(compare_trades_csv)
            for col in ["entry_time", "exit_time"]:
                if col in _cdf.columns:
                    _cdf[col] = pd.to_datetime(_cdf[col], utc=True)
            for _sym, _grp in _cdf.groupby("symbol"):
                _cmp_by_sym[_sym] = _grp.to_dict("records")
            logger.info(
                f"  🔵 对比交易已加载: {len(_cdf)} 笔 from {compare_trades_csv}"
            )
        except Exception as _e:
            logger.warning(f"  ⚠️  对比交易加载失败: {_e}")

    def _funnel_row_matches(
        row: Mapping[str, Any], sf: "str | None", fam: bool
    ) -> bool:
        if sf is None:
            return True
        s = str(row.get("strategy") or "").lower()
        if fam:
            return _arch_family(s) == str(sf).lower()
        return s == str(sf).lower()

    def _build_funnel_figures_for_sym(
        sym: str,
        *,
        strat_filter: "str | None",
        family_mode: bool,
        x_range,
        ref_index: Optional[pd.DatetimeIndex],
    ) -> list:
        rows_all = getattr(result, "funnel_per_bar", None) or []
        rows = [
            r
            for r in rows_all
            if str(r.get("symbol") or "") == sym
            and _funnel_row_matches(r, strat_filter, family_mode)
        ]
        if not rows:
            return []

        def _pcm_y(rec: Mapping[str, Any]) -> float:
            if rec.get("pcm_direction_filter") is False:
                return 0.0
            return 1.0

        def _bool_y(rec: Mapping[str, Any], key: str) -> float:
            v = rec.get(key)
            if v is None:
                return float("nan")
            return 1.0 if v else 0.0

        def _dir_y(rec: Mapping[str, Any]) -> float:
            dv = rec.get("direction_value")
            if dv is None:
                return float("nan")
            try:
                dvi = int(dv)
            except (TypeError, ValueError):
                return float("nan")
            return {-1: 0.0, 0: 0.5, 1: 1.0}.get(dvi, float("nan"))

        def _step_xy(ts: list, vals: list):
            if not ts:
                return [], []
            xs: list = []
            ys: list = []
            for i in range(len(ts)):
                if i > 0:
                    xs.append(ts[i])
                    ys.append(vals[i - 1])
                xs.append(ts[i])
                ys.append(vals[i])
            return xs, ys

        by_strat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_strat[str(r.get("strategy") or "unknown")].append(dict(r))

        figs: list = []
        _STAGES = [
            ("PCM EMA", _pcm_y, "#64748b"),
            ("Prefilter", lambda rec: _bool_y(rec, "prefilter"), "#3274D9"),
            ("Gate", lambda rec: _bool_y(rec, "gate"), "#7c3aed"),
            ("Entry filter", lambda rec: _bool_y(rec, "entry_filter"), "#ca8a04"),
            ("Direction (−1/0/+1)", _dir_y, "#059669"),
        ]

        def _compact_reason(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, (list, tuple)):
                return "; ".join(str(x) for x in value[:6])
            return str(value)

        def _block_points(sub_rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
            out: Dict[str, List[Any]] = {
                "x": [],
                "y": [],
                "stage": [],
                "reason": [],
                "strategy": [],
                "direction": [],
            }
            _prev_kill = False
            for rec in sub_rows:
                stage = ""
                reason = ""
                y = float("nan")
                if rec.get("kill_switch_blocked"):
                    # 连续多根处于 Kill：仅在进入该状态的第一根打点，避免 Hover 洪流
                    if _prev_kill:
                        continue
                    _prev_kill = True
                    stage = "KillSwitch"
                    reason = (
                        "constitutional risk pause → new entries blocked "
                        "(同段 Kill 仅在首根标点)"
                    )
                    y = 5.0
                else:
                    _prev_kill = False
                if not stage and rec.get("prefilter") is False:
                    stage = "Prefilter"
                    reason = _compact_reason(rec.get("prefilter_reason"))
                    y = 1.0
                elif not stage and rec.get("direction_value") == 0:
                    stage = "Direction"
                    reason = _compact_reason(
                        rec.get("direction_reason") or rec.get("direction_rule")
                    )
                    y = 4.0
                elif not stage and rec.get("gate") is False:
                    stage = "Gate"
                    reason = _compact_reason(rec.get("gate_reasons"))
                    y = 2.0
                elif not stage and rec.get("entry_filter") is False:
                    stage = "Entry"
                    reason = _compact_reason(rec.get("entry_filter_reason"))
                    y = 3.0
                elif not stage and int(rec.get("pcm_drop_slot", 0) or 0) > 0:
                    stage = "PCM"
                    reason = "slot_full"
                    y = 0.0
                elif (
                    not stage and int(rec.get("pcm_drop_direction_policy", 0) or 0) > 0
                ):
                    stage = "PCM"
                    reason = "direction_policy"
                    y = 0.0
                elif not stage and int(rec.get("pcm_drop_family_conflict", 0) or 0) > 0:
                    stage = "PCM"
                    reason = "family_conflict"
                    y = 0.0
                elif not stage and int(rec.get("pcm_drop_daily_limit", 0) or 0) > 0:
                    stage = "PCM"
                    reason = "daily_limit"
                    y = 0.0
                if not stage:
                    continue
                out["x"].append(pd.Timestamp(rec["timestamp"]))
                out["y"].append(y)
                out["stage"].append(stage)
                out["reason"].append(reason)
                out["strategy"].append(str(rec.get("strategy") or ""))
                out["direction"].append(str(rec.get("direction_value")))
            return out

        for strat_name in sorted(by_strat.keys()):
            sub = sorted(by_strat[strat_name], key=lambda t: t["timestamp"])
            ts = [pd.Timestamp(t["timestamp"]) for t in sub]
            if ref_index is not None and getattr(ref_index, "tz", None) is not None:
                _tz = ref_index.tz
                ts = [
                    (
                        x.tz_convert(_tz)
                        if x.tzinfo
                        else x.tz_localize("UTC").tz_convert(_tz)
                    )
                    for x in ts
                ]
            pf = bk_figure(
                title=f"{sym} · {strat_name} — gate / prefilter / direction / kill",
                x_axis_type="datetime",
                width=1400,
                height=200,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                x_range=x_range,
                y_range=(-0.15, 5.65),
            )
            pf.yaxis.ticker = FixedTicker(ticks=[0, 1, 2, 3, 4, 5])
            pf.yaxis.major_label_overrides = {
                0: "PCM",
                1: "Prefilter",
                2: "Gate",
                3: "EntryFlt",
                4: "Dir",
                5: "KS",
            }
            pf.grid.grid_line_alpha = 0.25

            for bi, (label, fn, color) in enumerate(_STAGES):
                vals = [float(fn(rec)) for rec in sub]
                xs, ys = _step_xy(ts, [bi + 0.35 * v for v in vals])
                if xs:
                    pf.line(
                        xs, ys, line_color=color, line_width=1.6, legend_label=label
                    )
            block_data = _block_points(sub)
            if block_data["x"]:
                bsrc = ColumnDataSource(block_data)
                blocked = pf.scatter(
                    "x",
                    "y",
                    source=bsrc,
                    marker="x",
                    size=9,
                    line_width=2,
                    color="#dc2626",
                    legend_label="No-entry reason",
                )
                pf.add_tools(
                    HoverTool(
                        renderers=[blocked],
                        tooltips=[
                            ("Time", "@x{%F %H:%M}"),
                            ("Stage", "@stage"),
                            ("Reason", "@reason"),
                            ("Strategy", "@strategy"),
                            ("Dir", "@direction"),
                        ],
                        formatters={"@x": "datetime"},
                    )
                )
            pf.legend.click_policy = "hide"
            pf.legend.label_text_font_size = "8pt"
            pf.legend.location = "top_left"
            figs.append(pf)
        return figs

    def _build_symbol_figure(
        sym: str,
        trades: list,
        cmp_trades: list = None,
        *,
        strat_filter: "str | None" = None,
        family_mode: bool = False,
    ) -> object:
        bars_1min = result.bars_1min.get(sym)
        if bars_1min is None or bars_1min.empty:
            return None
        bars_full = _merge_1min_for_chart(sym, bars_1min, data_path, map_extra_months)
        df = _resample_bars(bars_full, freq=bar_freq)
        if df.empty:
            return None

        # 全量 df 上算指标（含 map_extra_months 向前扩展），图上只画回测窗
        vw_n = int(map_vwap_window_bars)
        _span = max(1, int(map_long_ema_span))
        vwap_price = _rolling_tp_vwap(df, vw_n)
        ema_long_price = df["close"].ewm(span=_span, adjust=False).mean()
        wema200_price = weekly_ema200(df["close"])

        view_start = pd.Timestamp(bars_1min.index.min())
        view_end = pd.Timestamp(bars_1min.index.max())
        idx = df.index
        if idx.tz is not None:
            if view_start.tzinfo is None:
                view_start = view_start.tz_localize("UTC").tz_convert(idx.tz)
            else:
                view_start = view_start.tz_convert(idx.tz)
            if view_end.tzinfo is None:
                view_end = view_end.tz_localize("UTC").tz_convert(idx.tz)
            else:
                view_end = view_end.tz_convert(idx.tz)
        plot_mask = (idx >= view_start) & (idx <= view_end)
        df_plot = df.loc[plot_mask]
        if df_plot.empty:
            df_plot = df
            logger.warning(
                "trading map %s: plot window empty after tz align, using full merged range",
                sym,
            )

        cmp_trades = cmp_trades or []
        sym_r = sum(t.pnl_r for t in trades)
        sym_wr = sum(1 for t in trades if t.pnl_r > 0) / len(trades) if trades else 0
        p = bk_figure(
            title=f"{sym}  |  {len(trades)} trades  |  WR={sym_wr:.1%}  |  Total={sym_r:.2f}R",
            x_axis_type="datetime",
            width=1400,
            height=350,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.grid.grid_line_alpha = 0.3

        try:
            p.x_range.start = df_plot.index.min()
            p.x_range.end = df_plot.index.max()
            p.x_range.range_padding = 0.02
        except Exception:
            pass

        inc = df_plot.close >= df_plot.open
        dec = ~inc
        p.segment(
            df_plot.index[inc],
            df_plot.high[inc],
            df_plot.index[inc],
            df_plot.low[inc],
            color=_COLOR_UP,
            line_width=1,
        )
        p.segment(
            df_plot.index[dec],
            df_plot.high[dec],
            df_plot.index[dec],
            df_plot.low[dec],
            color=_COLOR_DOWN,
            line_width=1,
        )
        p.vbar(
            df_plot.index[inc],
            bar_w,
            df_plot.open[inc],
            df_plot.close[inc],
            fill_color=_COLOR_UP,
            line_color=_COLOR_UP,
            fill_alpha=0.8,
        )
        p.vbar(
            df_plot.index[dec],
            bar_w,
            df_plot.open[dec],
            df_plot.close[dec],
            fill_color=_COLOR_DOWN,
            line_color=_COLOR_DOWN,
            fill_alpha=0.8,
        )

        vp = vwap_price.reindex(df_plot.index)
        p.line(
            df_plot.index,
            vp,
            line_color="#c026d3",
            line_width=1.35,
            line_alpha=0.78,
            legend_label=f"VWAP({vw_n})",
        )
        ep = ema_long_price.reindex(df_plot.index)
        p.line(
            df_plot.index,
            ep,
            line_color="#f59e0b",
            line_width=1.35,
            line_alpha=0.88,
            legend_label=f"EMA({_span})",
        )
        wp = wema200_price.reindex(df_plot.index)
        p.line(
            df_plot.index,
            wp,
            line_color="#38bdf8",
            line_width=2.0,
            line_alpha=0.82,
            legend_label="W-EMA(200) weekly",
        )

        # ── SRB L1 / L3 horizontal SR overlays (toggleable; default symbols=BTC) ──
        sr_toggle_widgets: list = []
        _sym_u = str(sym).upper()
        _want_sr = _sym_u in _sr_syms
        if _want_sr:
            try:
                sr_full = _compute_sr_overlay_series(df)
                sr_plot = sr_full.reindex(df_plot.index)
                r_l3_up = p.line(
                    df_plot.index,
                    sr_plot["l3_upper"],
                    line_color="#9333ea",
                    line_width=1.8,
                    line_alpha=0.9,
                    legend_label="L3 wide upper",
                    visible=True,
                )
                r_l3_lo = p.line(
                    df_plot.index,
                    sr_plot["l3_lower"],
                    line_color="#c084fc",
                    line_width=1.8,
                    line_alpha=0.9,
                    legend_label="L3 wide lower",
                    visible=True,
                )
                r_l1_sup = p.line(
                    df_plot.index,
                    sr_plot["l1_support"],
                    line_color="#16a34a",
                    line_width=1.4,
                    line_dash="dotdash",
                    line_alpha=0.85,
                    legend_label="L1 swing support",
                    visible=True,
                )
                r_l1_res = p.line(
                    df_plot.index,
                    sr_plot["l1_resistance"],
                    line_color="#15803d",
                    line_width=1.4,
                    line_dash="dotdash",
                    line_alpha=0.85,
                    legend_label="L1 swing resistance",
                    visible=True,
                )
                t_l3 = Toggle(label="L3 wide SR", active=True, width=110)
                t_l1 = Toggle(label="L1 swing SR", active=True, width=110)
                t_l3.js_on_click(
                    CustomJS(
                        args=dict(a=r_l3_up, b=r_l3_lo),
                        code="a.visible = cb_obj.active; b.visible = cb_obj.active;",
                    )
                )
                t_l1.js_on_click(
                    CustomJS(
                        args=dict(a=r_l1_sup, b=r_l1_res),
                        code="a.visible = cb_obj.active; b.visible = cb_obj.active;",
                    )
                )
                sr_toggle_widgets = [t_l3, t_l1]
            except Exception as e:
                logger.warning("trading map %s: SR overlay skipped: %s", sym, e)

        # ── CRF: rolling 120 lo/hi band（仅 CRF 语义；勿在「All」Tab 叠到 SRB/BPC 等图上）
        # ``strat_filter is None`` 对应 Tabs 里的 **All**，若此处画 CRF 盒会把「盒通过」
        # 误读成当前 Tab 策略的 prefilter，造成 SRB 图与漏斗严重不一致。
        _draw_box = str(strat_filter or "").strip().lower() == "crf"
        if _draw_box:
            try:
                import numpy as _np
                from src.features.time_series.box_structure_features import (
                    compute_box_structure_from_series as _box_feat,
                )

                _bx = _box_feat(
                    close=df_plot["close"],
                    high=df_plot["high"],
                    low=df_plot["low"],
                )
            except Exception:
                _bx = None
            if _bx is not None and not _bx.empty:
                _bx = _bx.reindex(df_plot.index)
                _stab = pd.to_numeric(_bx.get("box_stability_120"), errors="coerce")
                _widp = pd.to_numeric(_bx.get("box_width_pct_120"), errors="coerce")
                _hi = pd.to_numeric(_bx.get("box_hi_120"), errors="coerce")
                _lo = pd.to_numeric(_bx.get("box_lo_120"), errors="coerce")
                _touch_hi = pd.to_numeric(
                    _bx.get("box_touches_hi_120"), errors="coerce"
                )
                _touch_lo = pd.to_numeric(
                    _bx.get("box_touches_lo_120"), errors="coerce"
                )
                # Keep in sync with config/strategies/crf/archetypes/prefilter.yaml:
                #   stab>=0.85, 0.04 <= width <= 0.30, hi/lo touches>=5
                _qual = (
                    (_stab.fillna(0.0) >= 0.85)
                    & (_widp.fillna(0.0) >= 0.04)
                    & (_widp.fillna(1.0) <= 0.30)
                    & (_touch_hi.fillna(0.0) >= 5)
                    & (_touch_lo.fillna(0.0) >= 5)
                )
                _pass_rate = float(_qual.mean()) if len(_qual) else 0.0
                print(f"   CRF prefilter overlay: pass_rate={_pass_rate:.1%}")
                qidx = df_plot.index[_qual.values]
                if len(qidx) > 0:
                    # Per-bar vbar — true gaps where filter fails, no NaN bridging.
                    p.vbar(
                        x=qidx,
                        width=bar_w,
                        top=_hi.values[_qual.values],
                        bottom=_lo.values[_qual.values],
                        fill_color="#22c55e",
                        fill_alpha=0.18,
                        line_color=None,
                        legend_label="CRF: box (prefilter pass)",
                    )

        # ── fade: pierce arm diamonds + level → recross (toggleable review overlay) ──
        fade_toggle_widgets: list = []
        if _has_fade_arch and _should_draw_fade_pierce_overlay(strat_filter):
            try:
                n_pierce, fade_renderers = _add_fade_pierce_overlay(
                    p,
                    ohlc_full=df,
                    view_start=view_start,
                    view_end=view_end,
                )
                if n_pierce:
                    print(f"   fade pierce overlay {sym}: {n_pierce} events")
                if fade_renderers:
                    t_fade = Toggle(label="FADE pierce/arm", active=True, width=130)
                    t_fade.js_on_click(
                        CustomJS(
                            args=dict(renderers=fade_renderers),
                            code=(
                                "for (var i = 0; i < renderers.length; i++) {"
                                " renderers[i].visible = cb_obj.active; }"
                            ),
                        )
                    )
                    fade_toggle_widgets = [t_fade]
            except Exception as e:
                logger.warning(
                    "trading map %s: fade pierce overlay skipped: %s", sym, e
                )

        if trades:
            # ── 连接线: 颜色 = win/loss ──
            for wl, lc, emoji in [
                ("win", _COLOR_WIN, "📈"),
                ("loss", _COLOR_LOSS, "📉"),
            ]:
                batch = [t for t in trades if ("win" if t.pnl_r > 0 else "loss") == wl]
                if batch:
                    p.multi_line(
                        xs=[[t.entry_time, t.exit_time] for t in batch],
                        ys=[[t.entry_price, t.exit_price] for t in batch],
                        line_color=lc,
                        line_dash="dashed",
                        line_alpha=0.4,
                        line_width=1.5,
                        legend_label=f"{emoji} {wl}",
                    )

            # ── 入场标记: 颜色 = 策略, 形状 = side+is_add ──
            # group by (archetype, side, is_add)
            entry_groups: dict = {}
            for t in trades:
                key = (str(t.archetype).lower(), t.side.upper(), t.is_add_position)
                entry_groups.setdefault(key, []).append(t)

            for (arch, side, is_add), batch in sorted(entry_groups.items()):
                strat_line = _strat_color(arch)
                fill_c = _entry_fill_for_side(side)
                marker = _MARKER_MAP.get((side, is_add), "circle")
                sz = 13 if is_add else 11
                add_txt = "Add " if is_add else ""
                leg_side = "Long" if str(side).upper() == "LONG" else "Short"
                # 图例只写字，形状由 glyph 表达，避免与 Unicode 符号重复叠字
                legend_lbl = f"{arch.upper()} {add_txt}{leg_side}"
                p.scatter(
                    x=[t.entry_time for t in batch],
                    y=[t.entry_price for t in batch],
                    marker=marker,
                    size=sz,
                    fill_color=fill_c,
                    line_color=strat_line,
                    line_width=2,
                    fill_alpha=0.88,
                    legend_label=legend_lbl,
                )

            # ── 出场标记: 颜色 = win/loss ──
            for wl, ec in [("win", _COLOR_WIN), ("loss", _COLOR_LOSS)]:
                batch = [t for t in trades if ("win" if t.pnl_r > 0 else "loss") == wl]
                if batch:
                    p.scatter(
                        x=[t.exit_time for t in batch],
                        y=[t.exit_price for t in batch],
                        marker="square",
                        size=8,
                        color=ec,
                        alpha=0.6,
                        legend_label=f"□ exit_{wl}",
                    )

            # ── SRB true_sr 持仓期水平锚点（按来源着色）──
            if _want_sr:
                for src_key, src_color in _SR_SOURCE_COLORS.items():
                    batch = [
                        t
                        for t in trades
                        if str(t.archetype or "").lower().startswith("srb")
                        and str(getattr(t, "srb_true_sr_source", "") or "") == src_key
                        and float(getattr(t, "srb_true_sr_level", 0.0) or 0.0) > 0
                    ]
                    if not batch:
                        continue
                    p.segment(
                        x0=[t.entry_time for t in batch],
                        y0=[float(t.srb_true_sr_level) for t in batch],
                        x1=[t.exit_time for t in batch],
                        y1=[float(t.srb_true_sr_level) for t in batch],
                        line_color=src_color,
                        line_width=2.0,
                        line_alpha=0.75,
                        line_dash="dashed",
                        legend_label=f"true_sr ({src_key})",
                    )

        # ── 对比交易标记（向量回测）── 蓝色圆圈
        if cmp_trades:
            _cmp_xs = [r["entry_time"] for r in cmp_trades]
            _cmp_ys = [r["entry_price"] for r in cmp_trades]
            p.scatter(
                x=_cmp_xs,
                y=_cmp_ys,
                marker="circle",
                size=10,
                color="#00b4d8",
                alpha=0.7,
                line_color="#0077b6",
                line_width=1.5,
                legend_label="◉ Vector BT entry",
            )

        p.add_tools(
            HoverTool(
                tooltips=[("Time", "@x{%F %H:%M}"), ("Price", "@y{0.2f}")],
                formatters={"@x": "datetime"},
                mode="mouse",
            )
        )
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        p.legend.label_text_font_size = "9pt"
        p.legend.background_fill_alpha = 0.92

        legend_toggle = Toggle(label="显示图例", active=True, width=100)
        legend_toggle.js_on_click(
            CustomJS(
                args=dict(lg=p.legend),
                code="lg.visible = cb_obj.active;",
            )
        )
        _overlay_toggles = [*sr_toggle_widgets, *fade_toggle_widgets]
        toggle_row = (
            bk_row(legend_toggle, *_overlay_toggles)
            if _overlay_toggles
            else legend_toggle
        )
        trade_list_div = None
        if _want_sr:
            trade_list_div = Div(
                text=_srb_trade_list_html(trades, sym),
                width=1400,
            )

        funnel_figs = _build_funnel_figures_for_sym(
            sym,
            strat_filter=strat_filter,
            family_mode=family_mode,
            x_range=p.x_range,
            ref_index=df_plot.index,
        )
        col_kids: list = [toggle_row, p]
        if trade_list_div is not None:
            col_kids.append(trade_list_div)
        if funnel_figs:
            col_kids.extend(funnel_figs)
        return bk_column(*col_kids, sizing_mode="stretch_width")

    # ── 构建单个 Tab ──
    def _build_tab(
        tab_label: str,
        strat_filter: "str | None",
        *,
        family_mode: bool = False,
    ) -> object:
        def _match_trade(t: ClosedTrade) -> bool:
            if strat_filter is None:
                return True
            arch = str(t.archetype).lower()
            if family_mode:
                return _arch_family(arch) == str(strat_filter).lower()
            return arch == str(strat_filter).lower()

        tab_trades = [t for t in result.trades if _match_trade(t)]
        n = len(tab_trades)
        wr = sum(1 for t in tab_trades if t.pnl_r > 0) / n if n else 0
        total = sum(t.pnl_r for t in tab_trades)
        strat_c = _strat_color(strat_filter) if strat_filter else "#888888"

        cmp_n = sum(len(v) for v in _cmp_by_sym.values())
        cmp_label = f" | 🔵 Vector={cmp_n}" if cmp_n else ""
        title_html = (
            f"<h2 style='color:{strat_c}'>🗺️ {tab_label} "
            f"| {n} trades | WR={wr:.1%} | Total={total:.2f}R{cmp_label}</h2>"
        )
        figs: list = [Div(text=title_html)]

        for sym in symbols:
            _bars = result.bars_1min.get(sym)
            if _bars is None or getattr(_bars, "empty", True):
                continue
            sym_trades = result.per_symbol.get(sym, [])
            if strat_filter is not None:
                sym_trades = [t for t in sym_trades if _match_trade(t)]
            fig = _build_symbol_figure(
                sym,
                sym_trades,
                cmp_trades=_cmp_by_sym.get(sym),
                strat_filter=strat_filter,
                family_mode=family_mode,
            )
            if fig is not None:
                figs.append(fig)

        child = bk_column(*figs, sizing_mode="stretch_width")
        return TabPanel(child=child, title=tab_label)

    # ── 组装 Tabs (All + 家族聚合 + 各策略) ──
    tabs_list = [_build_tab("All", None)]
    families = sorted({_arch_family(a) for a in all_archetypes if _arch_family(a)})
    for fam in families:
        tabs_list.append(_build_tab(fam.upper(), fam, family_mode=True))
    for arch in all_archetypes:
        tabs_list.append(_build_tab(arch.upper(), arch))

    _above_tabs = _build_portfolio_overview_above_tabs(result)
    layout = bk_column(*_above_tabs, Tabs(tabs=tabs_list), sizing_mode="stretch_width")
    html = bk_file_html(
        layout, resources=BK_RESOURCES, title=f"Trading Map: {result.strategy}"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"\n  🗺️  Trading map saved → {output_path}")
    return output_path
