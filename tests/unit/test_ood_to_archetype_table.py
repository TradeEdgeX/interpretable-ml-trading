import pandas as pd

from src.time_series_model.diagnostics.ood_to_archetype_table import (
    build_conditional_survival_table,
    load_ood_to_archetype_table_config,
)


def test_ood_to_archetype_table_weights_sum_to_one_per_bin_when_valid() -> None:
    cfg = load_ood_to_archetype_table_config("config/ood/ood_to_archetype_table.yaml")
    # Lower min_samples for unit test
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "min_samples_per_cell": 2,
        }
    )
    rows = []
    ts = pd.date_range("2025-01-01", periods=12, freq="4H", tz="UTC")
    # Make TC survive more in low_ood, TE survive more in mid_ood
    for t in ts[:4]:
        rows.append(
            {
                "symbol": "BTC",
                "timestamp": t,
                "ood_score": 0.2,
                "active_archetype": "TrendContinuationTC",
                "y_surv": 1,
            }
        )
        rows.append(
            {
                "symbol": "BTC",
                "timestamp": t,
                "ood_score": 0.2,
                "active_archetype": "TrendExpansionTE",
                "y_surv": 0,
            }
        )
    for t in ts[4:8]:
        rows.append(
            {
                "symbol": "BTC",
                "timestamp": t,
                "ood_score": 0.5,
                "active_archetype": "TrendContinuationTC",
                "y_surv": 0,
            }
        )
        rows.append(
            {
                "symbol": "BTC",
                "timestamp": t,
                "ood_score": 0.5,
                "active_archetype": "TrendExpansionTE",
                "y_surv": 1,
            }
        )
    df = pd.DataFrame(rows)

    table, _ = build_conditional_survival_table(df, cfg=cfg)
    # Check sum of weights per bin is ~1 (only over archetypes present in cfg)
    for b in ["low_ood", "mid_ood"]:
        s = float(table[table["bin"] == b]["weight"].sum())
        assert 0.95 <= s <= 1.05
