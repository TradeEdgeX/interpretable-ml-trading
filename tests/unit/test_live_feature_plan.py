from src.time_series_model.live.live_feature_plan import load_live_feature_plan


def test_live_feature_plan_overlay():
    feats = load_live_feature_plan(plan_path="config/live/live_feature_plan.yaml")
    # base minimal_required + overlay additions
    assert "close" in feats
    assert "atr" in feats
    assert "atr_percentile" in feats
    assert "bb_width_normalized" in feats
    # tier-expanded
    assert "compression_score" in feats
