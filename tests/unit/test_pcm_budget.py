from src.time_series_model.portfolio.pcm import (
    CapitalPolicy,
    SymbolDecision,
    compute_pcm_budget_for_decisions,
)


def test_pcm_budget_splits_by_mode_and_scores() -> None:
    decisions = [
        SymbolDecision(symbol="BTCUSDT", mode="TREND", gated=True, score=2.0),
        SymbolDecision(symbol="ETHUSDT", mode="TREND", gated=True, score=1.0),
        SymbolDecision(symbol="SOLUSDT", mode="MEAN", gated=True, score=1.0),
    ]
    policy = CapitalPolicy(base_mode_budgets={"MEAN": 1.0, "TREND": 1.0})
    res = compute_pcm_budget_for_decisions(decisions=decisions, policy=policy)

    per_mode = res.per_mode_budget
    assert per_mode["MEAN"] > 0
    assert per_mode["TREND"] > 0
    assert abs(per_mode["MEAN"] + per_mode["TREND"] - 1.0) < 1e-9

    per_symbol = res.per_symbol_budget
    # TREND budget should be split by score (BTC higher than ETH)
    assert per_symbol["BTCUSDT"] > per_symbol["ETHUSDT"]
    # MEAN symbol gets some budget
    assert per_symbol["SOLUSDT"] > 0
