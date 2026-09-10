from src.time_series_model.live.archetype_heuristics import (
    evaluate_required_conditions,
)


def _mk_bars_uptrend(n: int = 60, start: float = 100.0, step: float = 0.5):
    bars = []
    px = start
    for i in range(n):
        o = px
        c = px + step
        h = max(o, c) + 0.2
        l = min(o, c) - 0.2
        bars.append(
            {"open": o, "high": h, "low": l, "close": c, "volume": 1.0, "ts": i}
        )
        px = c
    return bars


def test_trend_breakout_pullback_continuation_can_pass_minimal() -> None:
    bars = _mk_bars_uptrend()
    # Force a breakout by pushing last close above rolling high.
    bars[-1]["close"] = bars[-2]["high"] + 5.0
    bars[-1]["high"] = bars[-1]["close"] + 0.1
    feats = {
        "15T_atr": 1.0,
        "pred_dir_prob": 0.8,
        "pred_mfe_atr": 1.2,
        "pred_mae_atr": 0.4,  # rr=3
        "pred_t_to_mfe": 10.0,
        "vpin": 0.2,
        "imbalance": 0.1,
        "total_vol": 10.0,
    }
    hd = evaluate_required_conditions(
        archetype_name="TrendContinuationTC",
        regime="TREND",
        required_conditions=["structure_breakout", "healthy_pullback", "rr_geq_2"],
        feats=feats,
        bars=bars,
    )
    assert hd.ok
    assert hd.side == "BUY"


def test_mean_failed_breakout_fade_requires_absorption_and_failed_breakout() -> None:
    bars = _mk_bars_uptrend()
    # Make a failed breakout: swept above HH but closed back below HH.
    prev_h = max(b["high"] for b in bars[-50:-1])
    bars[-1]["high"] = prev_h + 2.0
    bars[-1]["close"] = prev_h - 0.5
    feats = {
        "15T_atr": 1.0,
        "pred_dir_prob": 0.7,
        "pred_mfe_atr": 0.8,
        "pred_mae_atr": 0.4,
        "pred_t_to_mfe": 12.0,
        "vpin": 0.5,  # absorption_present
        "imbalance": -0.2,
        "total_vol": 10.0,
    }
    hd = evaluate_required_conditions(
        archetype_name="FailureReversionFR",
        regime="MEAN",
        required_conditions=[
            "breakout_failed_close_back",
            "no_follow_through",
            "absorption_present",
            "ttm_break",
        ],
        feats=feats,
        bars=bars,
    )
    assert hd.ok
    assert hd.side in ("BUY", "SELL")


def test_unknown_condition_fails_closed() -> None:
    hd = evaluate_required_conditions(
        archetype_name="X",
        regime="TREND",
        required_conditions=["does_not_exist"],
        feats={"15T_atr": 1.0},
        bars=[],
    )
    assert not hd.ok
