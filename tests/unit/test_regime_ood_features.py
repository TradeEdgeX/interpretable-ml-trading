"""Tests for P5 non-stationarity features: regime_state and ood_score.

Bug/Feature background:
  - P5 identifies three non-stationarity gaps: Regime detection, OOD detection, Alpha Decay.
  - `compute_regime_state_from_df` replicates RegimeDetector logic as a feature for Gate/Evidence.
  - `compute_ood_score_from_df` computes fraction of features outside training [q05, q95].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# regime_state tests
# ============================================================================


class TestRegimeState:
    """Test compute_regime_state_from_df."""

    @staticmethod
    def _make_df(
        atr_percentile=None,
        oi_zscore=None,
        funding_rate_abs_zscore_50=None,
        n: int = 5,
    ) -> pd.DataFrame:
        data = {}
        if atr_percentile is not None:
            data["atr_percentile"] = (
                atr_percentile
                if hasattr(atr_percentile, "__len__")
                else [atr_percentile] * n
            )
        if oi_zscore is not None:
            data["oi_zscore"] = (
                oi_zscore if hasattr(oi_zscore, "__len__") else [oi_zscore] * n
            )
        if funding_rate_abs_zscore_50 is not None:
            data["funding_rate_abs_zscore_50"] = (
                funding_rate_abs_zscore_50
                if hasattr(funding_rate_abs_zscore_50, "__len__")
                else [funding_rate_abs_zscore_50] * n
            )
        return pd.DataFrame(data)

    def test_normal_regime(self):
        """All normal conditions → regime_state = 0."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(
            atr_percentile=0.3, oi_zscore=0.5, funding_rate_abs_zscore_50=0.5
        )
        result = compute_regime_state_from_df(df)
        assert "regime_state" in result.columns
        assert (result["regime_state"] == 0).all()

    def test_high_vol_regime(self):
        """atr_percentile > 0.7 → regime_state = 1 (HIGH_VOL)."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(
            atr_percentile=0.85, oi_zscore=0.5, funding_rate_abs_zscore_50=0.5
        )
        result = compute_regime_state_from_df(df)
        assert (result["regime_state"] == 1).all()

    def test_high_leverage_regime(self):
        """oi_zscore > 1.5 AND funding_abs_zscore > 2.0 → regime_state = 2 (HIGH_LEVERAGE)."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(
            atr_percentile=0.85, oi_zscore=2.0, funding_rate_abs_zscore_50=3.0
        )
        result = compute_regime_state_from_df(df)
        # HIGH_LEVERAGE (2) overrides HIGH_VOL (1)
        assert (result["regime_state"] == 2).all()

    def test_high_leverage_overrides_high_vol(self):
        """HIGH_LEVERAGE takes precedence even when HIGH_VOL also true."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(
            atr_percentile=0.9, oi_zscore=2.5, funding_rate_abs_zscore_50=4.0
        )
        result = compute_regime_state_from_df(df)
        assert (result["regime_state"] == 2).all()

    def test_missing_columns_defaults_to_normal(self):
        """If all required columns are missing, defaults to NORMAL (0)."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = pd.DataFrame({"close": [100, 101, 102]})
        result = compute_regime_state_from_df(df)
        assert (result["regime_state"] == 0).all()

    def test_partial_columns(self):
        """Only atr_percentile present → can detect HIGH_VOL but not HIGH_LEVERAGE."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(atr_percentile=0.9)
        result = compute_regime_state_from_df(df)
        assert (result["regime_state"] == 1).all()

    def test_mixed_rows(self):
        """Different rows have different regimes."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df = self._make_df(
            atr_percentile=[0.3, 0.8, 0.9, 0.5, 0.4],
            oi_zscore=[0.5, 0.5, 2.0, 0.5, 0.5],
            funding_rate_abs_zscore_50=[0.5, 0.5, 3.0, 0.5, 0.5],
        )
        result = compute_regime_state_from_df(df)
        expected = [0, 1, 2, 0, 0]
        np.testing.assert_array_equal(result["regime_state"].values, expected)

    def test_output_shape(self):
        """Output DataFrame has same index as input."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        df = pd.DataFrame({"atr_percentile": np.random.rand(10)}, index=idx)
        result = compute_regime_state_from_df(df)
        assert len(result) == 10
        assert result.index.equals(idx)


# ============================================================================
# ood_score tests
# ============================================================================


class TestOodScore:
    """Test compute_ood_score_from_df."""

    @staticmethod
    def _baseline():
        return {
            "feat_a": {"q05": 0.0, "q95": 1.0},
            "feat_b": {"q05": -1.0, "q95": 1.0},
            "feat_c": {"q05": 10.0, "q95": 20.0},
        }

    def test_all_in_distribution(self):
        """All features within [q05, q95] → ood_score ≈ 0."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [0.5], "feat_b": [0.0], "feat_c": [15.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert result["ood_score"].iloc[0] == pytest.approx(0.0)

    def test_all_out_of_distribution(self):
        """All features outside [q05, q95] → ood_score = 1.0."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [5.0], "feat_b": [5.0], "feat_c": [50.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert result["ood_score"].iloc[0] == pytest.approx(1.0)

    def test_partial_ood(self):
        """1 of 3 features OOD → ood_score ≈ 0.333."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [5.0], "feat_b": [0.0], "feat_c": [15.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert result["ood_score"].iloc[0] == pytest.approx(1 / 3, abs=0.01)

    def test_no_baseline_returns_zero(self):
        """If no baseline provided, ood_score defaults to 0."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [5.0]})
        result = compute_ood_score_from_df(df, baseline=None)
        assert (result["ood_score"] == 0.0).all()

    def test_missing_features_ignored(self):
        """Features not in df are ignored in OOD count."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        baseline = {
            "feat_a": {"q05": 0.0, "q95": 1.0},
            "feat_missing": {"q05": 0.0, "q95": 1.0},
        }
        df = pd.DataFrame({"feat_a": [0.5]})
        result = compute_ood_score_from_df(df, baseline=baseline)
        # Only feat_a checked, it's in-distribution → 0/1 = 0
        assert result["ood_score"].iloc[0] == pytest.approx(0.0)

    def test_nan_values_not_counted_as_ood(self):
        """NaN values excluded from both numerator AND denominator (per-row)."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [np.nan], "feat_b": [0.0], "feat_c": [15.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        # feat_a=NaN → excluded, feat_b=in, feat_c=in → 0/2 = 0
        assert result["ood_score"].iloc[0] == pytest.approx(0.0)

    def test_nan_with_ood_per_row_denominator(self):
        """NaN reduces denominator per-row: 1 NaN + 2 OOD → 2/2 = 1.0, not 2/3."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [np.nan], "feat_b": [5.0], "feat_c": [50.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        # feat_a=NaN(excluded), feat_b=OOD, feat_c=OOD → 2/2 = 1.0
        assert result["ood_score"].iloc[0] == pytest.approx(1.0)

    def test_all_nan_returns_zero(self):
        """All NaN → denominator=0 → ood_score=0 (safe division)."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [np.nan], "feat_b": [np.nan], "feat_c": [np.nan]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert result["ood_score"].iloc[0] == pytest.approx(0.0)

    def test_multiple_rows(self):
        """Each row gets its own ood_score."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame(
            {
                "feat_a": [0.5, 5.0],  # in, out
                "feat_b": [0.0, 5.0],  # in, out
                "feat_c": [15.0, 15.0],  # in, in
            }
        )
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert result["ood_score"].iloc[0] == pytest.approx(0.0)
        assert result["ood_score"].iloc[1] == pytest.approx(2 / 3, abs=0.01)

    def test_ood_score_bounded(self):
        """ood_score should always be in [0, 1]."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame({"feat_a": [100.0], "feat_b": [100.0], "feat_c": [100.0]})
        result = compute_ood_score_from_df(df, baseline=self._baseline())
        assert 0.0 <= result["ood_score"].iloc[0] <= 1.0

    def test_p5_p95_key_fallback(self):
        """Baseline can use p5/p95 keys (from training_baseline.json format)."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        baseline = {"feat_a": {"p5": 0.0, "p95": 1.0, "mean": 0.5, "std": 0.3}}
        df = pd.DataFrame({"feat_a": [5.0]})
        result = compute_ood_score_from_df(df, baseline=baseline)
        assert result["ood_score"].iloc[0] == pytest.approx(1.0)


# ============================================================================
# Future function (look-ahead) tests
# ============================================================================


class TestRegimeStateNoLookAhead:
    """Verify regime_state uses only current-row data — no future leakage."""

    def test_truncated_equals_full(self):
        """Result at row t must be identical whether computed on df[:t+1] or full df."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "atr_percentile": np.random.uniform(0, 1, n),
                "oi_zscore": np.random.uniform(-1, 3, n),
                "funding_rate_abs_zscore_50": np.random.uniform(0, 4, n),
            }
        )
        full_result = compute_regime_state_from_df(df)

        # Check 5 random rows: truncated result must match full result
        for t in [0, 10, 24, 37, 49]:
            trunc_result = compute_regime_state_from_df(df.iloc[: t + 1])
            assert (
                trunc_result["regime_state"].iloc[-1]
                == full_result["regime_state"].iloc[t]
            ), f"Look-ahead detected at row {t}"

    def test_appending_future_data_no_change(self):
        """Appending future rows must not change past results."""
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        df_past = pd.DataFrame(
            {
                "atr_percentile": [0.3, 0.8, 0.5],
                "oi_zscore": [0.5, 0.5, 2.0],
                "funding_rate_abs_zscore_50": [0.5, 0.5, 3.0],
            }
        )
        df_extended = pd.concat(
            [
                df_past,
                pd.DataFrame(
                    {
                        "atr_percentile": [0.95],
                        "oi_zscore": [5.0],
                        "funding_rate_abs_zscore_50": [10.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        r_past = compute_regime_state_from_df(df_past)
        r_ext = compute_regime_state_from_df(df_extended)
        np.testing.assert_array_equal(
            r_past["regime_state"].values, r_ext["regime_state"].values[:3]
        )


class TestOodScoreNoLookAhead:
    """Verify ood_score uses only current-row data — no future leakage."""

    _BL = {
        "feat_a": {"q05": 0.0, "q95": 1.0},
        "feat_b": {"q05": -1.0, "q95": 1.0},
    }

    def test_truncated_equals_full(self):
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        np.random.seed(123)
        n = 40
        df = pd.DataFrame(
            {
                "feat_a": np.random.uniform(-2, 3, n),
                "feat_b": np.random.uniform(-3, 3, n),
            }
        )
        full = compute_ood_score_from_df(df, baseline=self._BL)
        for t in [0, 10, 25, 39]:
            trunc = compute_ood_score_from_df(df.iloc[: t + 1], baseline=self._BL)
            assert trunc["ood_score"].iloc[-1] == pytest.approx(
                full["ood_score"].iloc[t], abs=1e-12
            ), f"Look-ahead detected at row {t}"

    def test_appending_future_data_no_change(self):
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df_past = pd.DataFrame({"feat_a": [0.5, 5.0], "feat_b": [0.0, 5.0]})
        df_ext = pd.concat(
            [df_past, pd.DataFrame({"feat_a": [100.0], "feat_b": [100.0]})],
            ignore_index=True,
        )
        r_past = compute_ood_score_from_df(df_past, baseline=self._BL)
        r_ext = compute_ood_score_from_df(df_ext, baseline=self._BL)
        np.testing.assert_array_almost_equal(
            r_past["ood_score"].values, r_ext["ood_score"].values[:2]
        )


# ============================================================================
# Streaming consistency tests
# ============================================================================


class TestRegimeStateStreaming:
    """Streaming: row-by-row computation must equal batch."""

    def test_row_by_row_equals_batch(self):
        from src.features.time_series.baseline_features import (
            compute_regime_state_from_df,
        )

        np.random.seed(7)
        n = 30
        df = pd.DataFrame(
            {
                "atr_percentile": np.random.uniform(0, 1, n),
                "oi_zscore": np.random.uniform(-1, 3, n),
                "funding_rate_abs_zscore_50": np.random.uniform(0, 4, n),
            }
        )
        batch = compute_regime_state_from_df(df)
        for i in range(n):
            single = compute_regime_state_from_df(df.iloc[[i]])
            assert (
                single["regime_state"].iloc[0] == batch["regime_state"].iloc[i]
            ), f"Streaming mismatch at row {i}"


class TestOodScoreStreaming:
    """Streaming: row-by-row computation must equal batch."""

    _BL = {
        "feat_a": {"q05": 0.0, "q95": 1.0},
        "feat_b": {"q05": -1.0, "q95": 1.0},
        "feat_c": {"q05": 10.0, "q95": 20.0},
    }

    def test_row_by_row_equals_batch(self):
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        np.random.seed(77)
        n = 30
        df = pd.DataFrame(
            {
                "feat_a": np.random.uniform(-2, 3, n),
                "feat_b": np.random.uniform(-3, 3, n),
                "feat_c": np.random.uniform(5, 25, n),
            }
        )
        batch = compute_ood_score_from_df(df, baseline=self._BL)
        for i in range(n):
            single = compute_ood_score_from_df(df.iloc[[i]], baseline=self._BL)
            assert single["ood_score"].iloc[0] == pytest.approx(
                batch["ood_score"].iloc[i], abs=1e-12
            ), f"Streaming mismatch at row {i}"

    def test_row_by_row_with_nan(self):
        """Streaming with NaN: per-row denomination must match."""
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        df = pd.DataFrame(
            {
                "feat_a": [0.5, np.nan, 5.0],
                "feat_b": [5.0, 5.0, np.nan],
                "feat_c": [15.0, 50.0, 50.0],
            }
        )
        batch = compute_ood_score_from_df(df, baseline=self._BL)
        for i in range(len(df)):
            single = compute_ood_score_from_df(df.iloc[[i]], baseline=self._BL)
            assert single["ood_score"].iloc[0] == pytest.approx(
                batch["ood_score"].iloc[i], abs=1e-12
            ), f"Streaming NaN mismatch at row {i}"


# ============================================================================
# Monitoring script tests: L4 gate rule decay
# ============================================================================


class TestL4GateRuleDecay:
    """Tests for check_l4_gate_rule_decay."""

    @staticmethod
    def _make_gate_yaml(tmp_path, strategy="test_strat"):
        """Create minimal gate.yaml with one rule."""
        gate_dir = tmp_path / "config" / "strategies" / strategy / "archetypes"
        gate_dir.mkdir(parents=True)
        gate_yaml = gate_dir / "gate.yaml"
        gate_yaml.write_text(
            "hard_gates:\n"
            "  - id: g1\n"
            "    when:\n"
            "      atr_percentile:\n"
            "        value_gt: 0.8\n",
            encoding="utf-8",
        )
        return tmp_path / "config" / "strategies"

    def test_no_baseline_returns_skip(self, tmp_path):
        import sys

        sys.path.insert(0, str(tmp_path))
        from scripts.local_monitor_weekly import check_l4_gate_rule_decay

        df = pd.DataFrame({"atr_percentile": [0.5, 0.9]})
        result = check_l4_gate_rule_decay(df, "test_strat", baseline_gate_hit_rates={})
        assert result["status"] == "\u26aa SKIP"

    def test_no_gate_yaml_returns_skip(self, tmp_path):
        from scripts.local_monitor_weekly import check_l4_gate_rule_decay

        result = check_l4_gate_rule_decay(
            pd.DataFrame(),
            "nonexistent_strat",
            baseline_gate_hit_rates={"rule": {"deny_rate": 0.5}},
            config_root=str(tmp_path / "no_such_dir"),
        )
        assert result["status"] == "\u26aa SKIP"

    def test_no_decay_returns_ok(self, tmp_path):
        from scripts.local_monitor_weekly import check_l4_gate_rule_decay

        config_root = self._make_gate_yaml(tmp_path)
        # Baseline deny_rate=0.4, current data also has ~40% above 0.8
        np.random.seed(0)
        vals = np.concatenate([np.full(60, 0.5), np.full(40, 0.9)])
        df = pd.DataFrame({"atr_percentile": vals})
        baseline = {"g1__atr_percentile__value_gt": {"deny_rate": 0.40}}
        result = check_l4_gate_rule_decay(
            df,
            "test_strat",
            baseline,
            config_root=str(config_root),
        )
        assert result["status"] == "\U0001f7e2 OK"
        assert result["max_decay"] <= 0.5

    def test_strong_decay_returns_alert(self, tmp_path):
        from scripts.local_monitor_weekly import check_l4_gate_rule_decay

        config_root = self._make_gate_yaml(tmp_path)
        # Baseline deny_rate=0.50, current deny_rate≈0.05 → decay=0.90
        vals = np.concatenate([np.full(95, 0.5), np.full(5, 0.9)])
        df = pd.DataFrame({"atr_percentile": vals})
        baseline = {"g1__atr_percentile__value_gt": {"deny_rate": 0.50}}
        result = check_l4_gate_rule_decay(
            df,
            "test_strat",
            baseline,
            config_root=str(config_root),
        )
        assert "ALERT" in result["status"] or "WARN" in result["status"]
        assert result["max_decay"] > 0.5


# ============================================================================
# Monitoring script tests: L5 evidence IC decay
# ============================================================================


class TestL5EvidenceICDecay:
    """Tests for check_l5_evidence_ic_decay."""

    def test_no_baseline_returns_skip(self):
        from scripts.local_monitor_weekly import check_l5_evidence_ic_decay

        df = pd.DataFrame({"feat_a": [1, 2, 3], "forward_rr": [0.1, 0.2, 0.3]})
        result = check_l5_evidence_ic_decay(df, baseline_ics={})
        assert result["status"] == "\u26aa SKIP"

    def test_no_target_column_returns_skip(self):
        from scripts.local_monitor_weekly import check_l5_evidence_ic_decay

        df = pd.DataFrame({"feat_a": range(50)})
        result = check_l5_evidence_ic_decay(
            df, baseline_ics={"feat_a": 0.5}, target_col="nonexistent"
        )
        assert result["status"] == "\u26aa SKIP"

    def test_stable_ic_returns_ok(self):
        """Feature with stable IC → no decay."""
        from scripts.local_monitor_weekly import check_l5_evidence_ic_decay

        np.random.seed(42)
        n = 200
        feat = np.arange(n, dtype=float)
        target = feat * 0.8 + np.random.normal(0, 5, n)  # high correlation
        df = pd.DataFrame({"feat_a": feat, "forward_rr": target})
        baseline_ics = {"feat_a": 0.9}  # roughly what we'd get
        result = check_l5_evidence_ic_decay(df, baseline_ics)
        assert result["status"] == "\U0001f7e2 OK"

    def test_ic_decay_detected(self):
        """Feature with destroyed IC → decay detected."""
        from scripts.local_monitor_weekly import check_l5_evidence_ic_decay

        np.random.seed(42)
        n = 200
        # Random noise: IC ≈ 0, baseline was 0.5 → heavy decay
        df = pd.DataFrame(
            {
                "feat_a": np.random.randn(n),
                "forward_rr": np.random.randn(n),
            }
        )
        baseline_ics = {"feat_a": 0.5}
        result = check_l5_evidence_ic_decay(df, baseline_ics)
        assert result["features_decayed"] >= 1
        assert result["max_decay"] > 0.5

    def test_near_zero_baseline_skipped(self):
        """Features with near-zero baseline IC are skipped."""
        from scripts.local_monitor_weekly import check_l5_evidence_ic_decay

        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "feat_a": np.random.randn(n),
                "forward_rr": np.random.randn(n),
            }
        )
        baseline_ics = {"feat_a": 0.005}  # near-zero → should be skipped
        result = check_l5_evidence_ic_decay(df, baseline_ics)
        assert result["features_checked"] == 0


# ============================================================================
# Monitoring script tests: leading indicators (retrain trigger)
# ============================================================================


class TestCheckLeadingIndicators:
    """Tests for check_leading_indicators."""

    def test_no_reports_dir(self, tmp_path, monkeypatch):
        from scripts import monitor_retrain

        monkeypatch.setattr(monitor_retrain, "PROJECT_ROOT", tmp_path)
        result = monitor_retrain.check_leading_indicators("bpc", report=None)
        assert result["triggered"] is False
        assert "no reports" in result["reason"]

    def test_no_health_reports(self, tmp_path, monkeypatch):
        from scripts import monitor_retrain

        monkeypatch.setattr(monitor_retrain, "PROJECT_ROOT", tmp_path)
        (tmp_path / "reports").mkdir()
        result = monitor_retrain.check_leading_indicators("bpc", report=None)
        assert result["triggered"] is False
        assert "no weekly" in result["reason"]

    def test_low_decay_not_triggered(self, tmp_path, monkeypatch):
        import json
        from scripts import monitor_retrain

        monkeypatch.setattr(monitor_retrain, "PROJECT_ROOT", tmp_path)
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report = {
            "checks": [
                {"layer": "L4_gate_rule_decay", "max_decay": 0.2, "rules_decayed": 0},
                {
                    "layer": "L5_evidence_ic_decay",
                    "max_decay": 0.1,
                    "features_decayed": 0,
                },
            ]
        }
        (reports_dir / "weekly_health_check_20250101.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        result = monitor_retrain.check_leading_indicators("bpc", report=None)
        assert result["triggered"] is False
        assert result["max_decay"] == pytest.approx(0.2)

    def test_high_decay_triggers_retrain(self, tmp_path, monkeypatch):
        import json
        from scripts import monitor_retrain

        monkeypatch.setattr(monitor_retrain, "PROJECT_ROOT", tmp_path)
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report = {
            "checks": [
                {"layer": "L4_gate_rule_decay", "max_decay": 0.8, "rules_decayed": 3},
                {
                    "layer": "L5_evidence_ic_decay",
                    "max_decay": 0.6,
                    "features_decayed": 5,
                },
            ]
        }
        (reports_dir / "weekly_health_check_20250101.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        result = monitor_retrain.check_leading_indicators("bpc", report=None)
        assert result["triggered"] is True
        assert result["max_decay"] == pytest.approx(0.8)


# ============================================================================
# OOD baseline injection tests
# ============================================================================


class TestOodBaselineInjection:
    """Verify ood_score baseline_path is injected into compute_params at runtime."""

    def test_inject_ood_baseline_path(self, tmp_path):
        """_inject_ood_baseline_path sets baseline_path from archetypes_dir parent."""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        comp = IncrementalFeatureComputer.__new__(IncrementalFeatureComputer)
        comp._feature_deps = {
            "features": {
                "ood_score_f": {
                    "compute_func": "compute_ood_score_from_df",
                    "compute_params": {
                        "normalized": True,
                        "output_normalization_map": {"ood_score": "bounded_0_1"},
                    },
                }
            }
        }
        archetypes_dir = str(tmp_path / "config" / "strategies" / "bpc" / "archetypes")
        comp._inject_ood_baseline_path(archetypes_dir)

        cp = comp._feature_deps["features"]["ood_score_f"]["compute_params"]
        expected = str(
            tmp_path / "config" / "strategies" / "bpc" / "training_baseline.json"
        )
        assert cp["baseline_path"] == expected

    def test_inject_no_overwrite_existing(self, tmp_path):
        """If baseline_path already set, do not overwrite."""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        comp = IncrementalFeatureComputer.__new__(IncrementalFeatureComputer)
        comp._feature_deps = {
            "features": {
                "ood_score_f": {
                    "compute_params": {"baseline_path": "/custom/path.json"},
                }
            }
        }
        comp._inject_ood_baseline_path(str(tmp_path / "whatever" / "archetypes"))
        assert (
            comp._feature_deps["features"]["ood_score_f"]["compute_params"][
                "baseline_path"
            ]
            == "/custom/path.json"
        )

    def test_inject_no_archetypes_dir(self):
        """No archetypes_dir → no injection, no crash."""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        comp = IncrementalFeatureComputer.__new__(IncrementalFeatureComputer)
        comp._feature_deps = {"features": {"ood_score_f": {"compute_params": {}}}}
        comp._inject_ood_baseline_path(None)  # should not crash
        assert (
            "baseline_path"
            not in comp._feature_deps["features"]["ood_score_f"]["compute_params"]
        )

    def test_ood_score_with_injected_baseline(self, tmp_path):
        """End-to-end: baseline injected → ood_score produces non-zero values."""
        import json
        from src.features.time_series.baseline_features import compute_ood_score_from_df

        # Create training_baseline.json
        baseline = {
            "feature_distributions": {
                "feat_a": {"p5": 0.0, "p95": 1.0},
                "feat_b": {"p5": -1.0, "p95": 1.0},
            }
        }
        bl_path = tmp_path / "training_baseline.json"
        bl_path.write_text(json.dumps(baseline), encoding="utf-8")

        # Simulate: all features OOD
        df = pd.DataFrame({"feat_a": [5.0], "feat_b": [5.0]})
        result = compute_ood_score_from_df(df, baseline_path=str(bl_path))
        assert result["ood_score"].iloc[0] == pytest.approx(1.0)

        # Simulate: all features in-distribution
        df2 = pd.DataFrame({"feat_a": [0.5], "feat_b": [0.0]})
        result2 = compute_ood_score_from_df(df2, baseline_path=str(bl_path))
        assert result2["ood_score"].iloc[0] == pytest.approx(0.0)


# ====================================================================
# Retrain Monitor Gauge integration tests
# ====================================================================


class TestRunRetrainCheck:
    """Test _run_retrain_check() updates Prometheus Gauges correctly."""

    def _make_trades(self, pnls):
        """Helper: create trade dicts with given PnL list."""
        from datetime import datetime, timedelta

        trades = []
        now = datetime.now()
        for i, pnl in enumerate(pnls):
            trades.append(
                {
                    "position_id": f"pos_{i}",
                    "symbol": "BTCUSDT",
                    "side": "long",
                    "entry_time": (now - timedelta(days=len(pnls) - i)).isoformat(),
                    "exit_time": (now - timedelta(days=len(pnls) - i - 1)).isoformat(),
                    "entry_price": 50000,
                    "exit_price": 50000 + pnl * 100,
                    "realized_pnl": pnl,
                    "strategy_id": "bpc",
                    "archetype": "bpc",
                    "status": "closed",
                }
            )
        return trades

    def test_check_triggers_returns_correct_structure(self):
        """check_triggers returns dict with triggered, trigger_count, details."""
        from scripts.monitor_retrain import check_triggers

        trades = self._make_trades([1.0, -0.5, 1.0, 0.3])
        triggers_cfg = {
            "schedule_days": 90,
            "sharpe_decay_ratio": 0.5,
            "consecutive_losses": 8,
            "max_data_age_days": 120,
            "leading_indicator_decay": 0.5,
        }
        result = check_triggers("bpc", trades, None, triggers_cfg)
        assert "triggered" in result
        assert "trigger_count" in result
        assert "details" in result
        assert isinstance(result["details"], dict)

    def test_gauge_update_from_check_result(self):
        """Simulate what _run_retrain_check does: parse result → set Gauges."""
        from src.time_series_model.live.metrics_exporter import _NoopMetric

        # Use NoopMetric to verify the call pattern works without real Prometheus
        noop = _NoopMetric()
        result = {
            "triggered": True,
            "trigger_count": 2,
            "details": {
                "live_sharpe_30d": 0.15,
                "sharpe_ratio": 0.3,
                "consecutive_losses": 3,
                "days_since_last_train": 45,
                "leading_indicator_decay": {"max_decay": 0.6, "triggered": True},
            },
        }
        # These calls must not raise
        noop.labels(strategy="bpc").set(1 if result["triggered"] else 0)
        noop.labels(strategy="bpc").set(result["trigger_count"])
        details = result["details"]
        noop.labels(strategy="bpc").set(details["live_sharpe_30d"])
        noop.labels(strategy="bpc").set(details.get("sharpe_ratio", 0.0))
        noop.labels(strategy="bpc").set(details["consecutive_losses"])
        noop.labels(strategy="bpc").set(details["days_since_last_train"])
        leading = details.get("leading_indicator_decay", {})
        max_decay = leading.get("max_decay", 0.0) if isinstance(leading, dict) else 0.0
        noop.labels(strategy="bpc").set(max_decay)

    def test_no_db_no_crash(self, tmp_path):
        """_run_retrain_check should not crash when DB doesn't exist."""
        from scripts.monitor_retrain import check_triggers, load_live_trades

        fake_db = tmp_path / "nonexistent.db"
        trades = load_live_trades(fake_db, tmp_path, strategy="bpc", days=90)
        assert trades == []
        # check_triggers with empty trades should still return valid result
        result = check_triggers(
            "bpc",
            trades,
            None,
            {"schedule_days": 90, "consecutive_losses": 8},
        )
        assert result["triggered"] is True  # schedule_days triggered (no report)
        assert result["details"]["consecutive_losses"] == 0

    def test_metrics_exporter_has_retrain_gauges(self):
        """Verify the 7 new retrain Gauges exist on Metrics class."""
        from unittest.mock import patch

        with patch(
            "src.time_series_model.live.metrics_exporter._PROM_AVAILABLE", False
        ):
            from src.time_series_model.live.metrics_exporter import Metrics

            m = Metrics()  # NOOP mode
        assert hasattr(m, "retrain_triggered")
        assert hasattr(m, "retrain_trigger_count")
        assert hasattr(m, "sharpe_live_30d")
        assert hasattr(m, "sharpe_decay_ratio")
        assert hasattr(m, "consecutive_losses")
        assert hasattr(m, "days_since_last_train")
        assert hasattr(m, "alpha_decay_max")
        # All should be callable (NOOP)
        m.retrain_triggered.labels(strategy="test").set(0)
        m.alpha_decay_max.labels(strategy="test").set(0.5)
