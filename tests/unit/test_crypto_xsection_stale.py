"""Stale FeatureStore must not keep the crypto xsection short gate open."""

from mlbot_console.services.crypto_top_picks import build_action
from mlbot_console.services.crypto_xsection_metrics import asof_is_stale


def test_march_panel_is_stale_in_september() -> None:
    assert asof_is_stale("2026-03-31", now="2026-09-07")
    assert not asof_is_stale("2026-09-06", now="2026-09-07")
    assert asof_is_stale(None, now="2026-09-07")


def test_stale_bear_snapshot_blocks_short() -> None:
    action = build_action(
        {
            "regime": "bear",
            "roc_90": -0.221,
            "bias": "short",
            "asof": "2026-03-31",
            "stale": True,
        }
    )
    assert action["allow_short"] is False
    assert action["allow_long"] is False
    assert "过期" in action["verdict"]


def test_fresh_bear_still_allows_short() -> None:
    action = build_action(
        {
            "regime": "bear",
            "roc_90": -0.221,
            "bias": "short",
            "asof": "2026-09-06",
            "stale": False,
        }
    )
    assert action["allow_short"] is True
    assert action["allow_long"] is False
