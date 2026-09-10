"""P1 regression tests: E22 ADX(50) labeled regime + exit_by_regime.

Covers the bugs that caused v1/v2 = 47.57R (ADX never loaded, regime = neutral = E9):
  1. Labeled regime ``classify()`` returns correct bull/bear/neutral
  2. ``ExecutionParamGenerator`` respects ``exit_by_regime`` + ``regime_label``
  3. ``extract_features_from_archetypes`` pulls ``adx_50`` from labeled regime
  4. Tick timezone normalization: naive vs UTC-aware produce identical masks
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.time_series_model.archetype.loader import RegimeConfig
from src.time_series_model.live.generic_live_strategy import ExecutionParamGenerator
from src.time_series_model.live.live_feature_plan import (
    extract_features_from_archetypes,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Labeled regime classify() → bull / bear / neutral
# ═══════════════════════════════════════════════════════════════════════════


def _make_e22_regime() -> RegimeConfig:
    """Replicate E22 production regime.yaml (labeled schema)."""
    return RegimeConfig.from_mapping(
        {
            "allowed_regimes": {
                "bull": {
                    "description": "强趋势市",
                    "match": "all",
                    "rules": [
                        {"feature": "adx_50", "operator": ">=", "value": 25},
                        {
                            "feature": "ema_1200_position",
                            "operator": ">=",
                            "value": 0.1,
                        },
                    ],
                },
                "bear": {
                    "description": "弱势/下跌",
                    "match": "any",
                    "rules": [
                        {"feature": "adx_50", "operator": "<=", "value": 20},
                        {
                            "feature": "ema_1200_position",
                            "operator": "<=",
                            "value": -0.1,
                        },
                    ],
                },
                "neutral": {
                    "description": "震荡/无方向",
                    "match": "any",
                    "rules": [
                        {"feature": "adx_50", "operator": "<=", "value": 25},
                    ],
                },
            },
            "allowed_sides": ["long", "short"],
        }
    )


class TestLabeledRegimeClassify:
    """Verify classify() returns correct labels for the E22 ADX schema."""

    def test_classify_bull(self):
        rc = _make_e22_regime()
        # strong trend + price above EMA → bull
        assert rc.classify({"adx_50": 30, "ema_1200_position": 0.15}) == "bull"

    def test_classify_bear_by_adx(self):
        rc = _make_e22_regime()
        # weak ADX → bear (any match: adx_50 <= 20)
        assert rc.classify({"adx_50": 18, "ema_1200_position": 0.05}) == "bear"

    def test_classify_bear_by_ema(self):
        rc = _make_e22_regime()
        # price below EMA → bear (any match: ema_1200_position <= -0.1)
        assert rc.classify({"adx_50": 22, "ema_1200_position": -0.15}) == "bear"

    def test_classify_neutral_deadband(self):
        rc = _make_e22_regime()
        # adx=22 (<=25) matches neutral rule; bull/bear both fail → neutral
        assert rc.classify({"adx_50": 22, "ema_1200_position": 0.05}) == "neutral"

    def test_classify_high_adx_low_ema_neutral(self):
        """High ADX + low EMA → neutral (bull fails EMA, bear fails ADX).

        This is a critical boundary for E22: ADX >= 25 but EMA < 0.1
        means strong trend but price below EMA → should NOT be bull.
        The E21→E22 improvement came from correctly NOT treating
        high-ADX/low-EMA as bull."""
        rc = _make_e22_regime()
        # ADX=30 (strong trend), EMA=0.05 (below bull threshold 0.1)
        assert rc.classify({"adx_50": 30, "ema_1200_position": 0.05}) == "neutral"

    def test_classify_missing_feature_falls_to_neutral(self):
        rc = _make_e22_regime()
        # adx_50 missing → bull fails (all → False), bear fails (any → False),
        # neutral fails (any → False) → returns "neutral" as ultimate fallback
        assert rc.classify({"ema_1200_position": 0.15}) == "neutral"

    def test_classify_or_default_with_empty_features(self):
        rc = _make_e22_regime()
        assert rc.classify_or_default({}, "neutral") == "neutral"

    def test_is_empty_false_for_labeled_regime(self):
        rc = _make_e22_regime()
        assert rc.is_empty is False, (
            "Labeled regime with per-label rules must NOT be empty — "
            "this was the bug that caused live to skip classify()"
        )

    def test_prod_yaml_matches_test_fixture(self):
        """Anchor: _make_e22_regime() must match prod regime.yaml."""
        prod = RegimeConfig.from_yaml(
            Path("config/strategies/tpc/archetypes/regime.yaml")
        )
        fixture = _make_e22_regime()
        # same classify results for key boundary cases
        cases = [
            ({"adx_50": 30, "ema_1200_position": 0.15}, "bull"),
            ({"adx_50": 18, "ema_1200_position": 0.05}, "bear"),
            ({"adx_50": 30, "ema_1200_position": 0.05}, "neutral"),
            ({"adx_50": 22, "ema_1200_position": 0.05}, "neutral"),
        ]
        for feats, expected in cases:
            assert prod.classify(feats) == expected, f"prod mismatch: {feats}"
            assert fixture.classify(feats) == expected, f"fixture mismatch: {feats}"

    def test_is_empty_true_for_default(self):
        rc = RegimeConfig()
        assert rc.is_empty is True


# ═══════════════════════════════════════════════════════════════════════════
# 2. ExecutionParamGenerator + exit_by_regime + regime_label
# ═══════════════════════════════════════════════════════════════════════════


def _make_e22_execution_config() -> dict:
    """Replicate E22 production execution.yaml."""
    return {
        "stop_loss": {
            "type": "trailing",
            "initial_r": 4.0,
            "structural_exit": "ema1200",
            "trailing": {
                "enabled": True,
                "activation_r": 3.5,
                "trail_r": 6.0,
                "expand_with_primary_atr": True,
            },
            "breakeven": {
                "enabled": True,
                "trigger_r": 6.0,
                "lock_level_r": 2,
                "measure": "atr",
            },
            "guardrails": {
                "max_stop_pct": 0.2,
                "min_stop_pct": 0.01,
            },
            "exit_by_regime": {
                "bull": {"trailing": {"enabled": False}},
                "bear": {"trailing": {"enabled": True}},
                "neutral": {"trailing": {"enabled": True}},
            },
        },
        "add_position": {
            "trigger": {"type": "float_r_ladder_only"},
            "add_size_multipliers": [0.25, 0.5, 1.0],
            "min_current_r_by_add": [0.5, 1, 1.5],
            "min_current_r_unit": "atr",
            "sizing_mode": "fixed_multiplier",
            "inherit_parent_stop": True,
        },
        "execution_constraints": {
            "allow_add_on": True,
            "min_order_interval_minutes": 60,
        },
        "holding": {"max_holding_bars": 0, "time_stop_bars": 0},
        "take_profit": {"enabled": False},
        "version": 1,
    }


class TestExitByRegime:
    """Verify exit_by_regime controls trailing based on regime_label."""

    def test_bull_disables_trailing(self):
        gen = ExecutionParamGenerator(_make_e22_execution_config())
        params = gen.generate_params(0.5, regime_label="bull")
        assert (
            params.get("allow_trailing") is False
        ), "Bull regime must disable trailing (structural exit instead)"
        assert params.get("activation_r") is None
        assert params.get("trail_r") is None

    def test_bear_enables_trailing(self):
        gen = ExecutionParamGenerator(_make_e22_execution_config())
        params = gen.generate_params(0.5, regime_label="bear")
        assert (
            params.get("allow_trailing") is True
        ), "Bear regime must keep trailing ON (E23 proved bear structural = -6.14R)"
        assert params.get("activation_r") == 3.5
        assert params.get("trail_r") == 6.0

    def test_neutral_enables_trailing(self):
        gen = ExecutionParamGenerator(_make_e22_execution_config())
        params = gen.generate_params(0.5, regime_label="neutral")
        assert params.get("allow_trailing") is True
        assert params.get("activation_r") == 3.5

    def test_default_regime_label_is_neutral(self):
        gen = ExecutionParamGenerator(_make_e22_execution_config())
        params = gen.generate_params(0.5)  # no regime_label → default "neutral"
        assert params.get("allow_trailing") is True

    def test_no_exit_by_regime_falls_back_to_default_trailing(self):
        """Without exit_by_regime config, trailing stays at execution.yaml default."""
        cfg = _make_e22_execution_config()
        del cfg["stop_loss"]["exit_by_regime"]
        gen = ExecutionParamGenerator(cfg)
        params = gen.generate_params(0.5, regime_label="bull")
        # No exit_by_regime → trailing stays on (execution.yaml global default)
        assert params.get("allow_trailing") is True


# ═══════════════════════════════════════════════════════════════════════════
# 3. Feature extraction from labeled regime → adx_50 appears
# ═══════════════════════════════════════════════════════════════════════════


def _write_archetypes(root: Path, regime_yaml: str) -> None:
    """Write minimal archetype files for feature extraction testing.

    The root path is the STRATEGY directory (contains features.yaml + archetypes/).
    """
    arch = root / "archetypes"
    arch.mkdir(parents=True)
    # Required: features.yaml at strategy root (even if empty requested_features)
    (root / "features.yaml").write_text("feature_pipeline:\n  requested_features: []\n")
    for fname, content in [
        ("gate.yaml", "hard_gates: []\n"),
        ("evidence.yaml", "evidence: []\n"),
        ("execution.yaml", "execution_constraints: {}\n"),
        ("prefilter.yaml", "rules: []\n"),
        ("entry_filters.yaml", "filters: []\n"),
        ("direction.yaml", "direction_rules: []\n"),
    ]:
        (arch / fname).write_text(content)
    (arch / "regime.yaml").write_text(textwrap.dedent(regime_yaml).lstrip("\n"))


class TestLabeledRegimeFeatureExtraction:
    """Verify extract_features_from_archetypes finds adx_50 in labeled regime."""

    def test_adx_50_extracted_from_flat_rules(self, tmp_path: Path):
        """Old schema: top-level rules list — still works."""
        strat = tmp_path / "test_strat"
        _write_archetypes(
            strat,
            """
            allowed_regimes: [bull, bear, neutral]
            rules:
              - feature: adx_50
                operator: ">="
                value: 25
            """,
        )
        cols, _ = extract_features_from_archetypes(str(strat / "archetypes"))
        assert "adx_50" in cols, "adx_50 must be extracted from old flat rules schema"

    def test_adx_50_extracted_from_labeled_regime(self, tmp_path: Path):
        """New schema: allowed_regimes.<label>.rules — the E22 format."""
        strat = tmp_path / "test_strat"
        _write_archetypes(
            strat,
            """
            allowed_regimes:
              bull:
                match: all
                rules:
                  - feature: adx_50
                    operator: ">="
                    value: 25
                  - feature: ema_1200_position
                    operator: ">="
                    value: 0.1
              bear:
                match: any
                rules:
                  - feature: adx_50
                    operator: "<="
                    value: 20
                  - feature: ema_1200_position
                    operator: "<="
                    value: -0.1
              neutral:
                match: any
                rules:
                  - feature: adx_50
                    operator: "<="
                    value: 25
            allowed_sides:
            - long
            - short
            """,
        )
        cols, _ = extract_features_from_archetypes(str(strat / "archetypes"))
        assert "adx_50" in cols, (
            "adx_50 MUST be extracted from labeled regime schema — "
            "this was the bug that caused v1/v2 = 47.57R (ADX never loaded)"
        )
        assert (
            "ema_1200_position" in cols
        ), "ema_1200_position must also be extracted from labeled regime"

    def test_labeled_regime_without_rules_not_extracted(self, tmp_path: Path):
        """Labeled regime labels without rules should add no features."""
        strat = tmp_path / "test_strat"
        _write_archetypes(
            strat,
            """
            allowed_regimes:
              bull:
                description: "just a label, no rules"
              bear:
                description: "also no rules"
            """,
        )
        cols, _ = extract_features_from_archetypes(str(strat / "archetypes"))
        assert "adx_50" not in cols


# ═══════════════════════════════════════════════════════════════════════════
# 4. Tick timezone normalization: naive vs UTC-aware → same mask
# ═══════════════════════════════════════════════════════════════════════════


class TestTickTimezoneNormalization:
    """Verify ``pd.to_datetime(..., utc=True).dt.tz_convert(None)`` produces
    identical filtering results for naive and UTC-aware tick timestamps."""

    @staticmethod
    def _make_ticks_naive() -> pd.DataFrame:
        ts = pd.date_range("2025-06-01", "2025-06-02", freq="1min", inclusive="left")
        n = len(ts)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "price": np.random.uniform(100, 200, n),
                "volume": np.random.uniform(0.1, 10.0, n),
                "side": np.random.choice([1, -1], n),
            }
        )

    @staticmethod
    def _make_ticks_utc() -> pd.DataFrame:
        ts = pd.date_range(
            "2025-06-01", "2025-06-02", freq="1min", inclusive="left", tz="UTC"
        )
        n = len(ts)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "price": np.random.uniform(100, 200, n),
                "volume": np.random.uniform(0.1, 10.0, n),
                "side": np.random.choice([1, -1], n),
            }
        )

    def test_normalize_naive_unchanged(self):
        """tz-naive timestamps remain tz-naive after normalization."""
        ticks = self._make_ticks_naive()
        ts = pd.to_datetime(ticks["timestamp"], utc=True).dt.tz_convert(None)
        assert ts.dt.tz is None
        # should be equal to original (interpreted as UTC)
        assert (ts == ticks["timestamp"]).all()

    def test_normalize_utc_to_naive(self):
        """tz-aware UTC timestamps become tz-naive with same wall-clock."""
        ticks = self._make_ticks_utc()
        ts = pd.to_datetime(ticks["timestamp"], utc=True).dt.tz_convert(None)
        assert ts.dt.tz is None
        # wall-clock values preserved
        expected = ticks["timestamp"].dt.tz_convert(None)
        assert (ts == expected).all()

    def test_filter_same_result_naive_vs_utc(self):
        """Naive and UTC-aware ticks yield identical mask after normalization."""
        naive = self._make_ticks_naive()
        utc = self._make_ticks_utc()

        load_start = pd.Timestamp("2025-06-01 12:00:00")
        load_end = pd.Timestamp("2025-06-01 18:00:00")

        # Normalize both
        ts_naive = pd.to_datetime(naive["timestamp"], utc=True).dt.tz_convert(None)
        ts_utc = pd.to_datetime(utc["timestamp"], utc=True).dt.tz_convert(None)

        mask_naive = (ts_naive >= load_start) & (ts_naive <= load_end)
        mask_utc = (ts_utc >= load_start) & (ts_utc <= load_end)

        assert mask_naive.sum() == mask_utc.sum()
        assert (mask_naive == mask_utc).all()

    def test_utc_without_normalize_raises(self):
        """Without normalization, comparing tz-aware to tz-naive raises TypeError."""
        utc = self._make_ticks_utc()
        load_start = pd.Timestamp("2025-06-01 12:00:00")

        with pytest.raises(TypeError, match="Invalid comparison.*datetime64"):
            _ = utc["timestamp"] >= load_start
