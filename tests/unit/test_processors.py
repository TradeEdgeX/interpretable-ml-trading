"""Unit tests for data processors."""

import numpy as np
import pandas as pd
import pytest

from src.data_tools.processors import (
    FillNaProcessor,
    ClipOutlierProcessor,
    DtypeDowncastProcessor,
    ReplaceInfProcessor,
    ProcessorChain,
    Pipeline,
    get_default_processor_chain,
    get_live_processor_chain,
    DataValidator,
    validate_dataframe,
    ValidateQualityProcessor,
    ValidateLeakageProcessor,
    LookaheadLeakageDetector,
    detect_lookahead_leakage,
    ResearchValidator,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with various edge cases."""
    return pd.DataFrame(
        {
            "normal": [1.0, 2.0, 3.0, 4.0, 5.0],
            "with_nan": [1.0, np.nan, 3.0, np.nan, 5.0],
            "with_inf": [1.0, np.inf, 3.0, -np.inf, 5.0],
            "outliers": [1.0, 2.0, 100.0, 3.0, 4.0],  # 100 is outlier
            "int_col": [1, 2, 3, 4, 5],
        }
    )


class TestFillNaProcessor:
    def test_ffill(self, sample_df):
        proc = FillNaProcessor(method="ffill")
        result = proc.process(sample_df)
        assert result["with_nan"].isna().sum() == 0
        assert result["with_nan"].iloc[1] == 1.0  # filled with previous value

    def test_bfill(self, sample_df):
        proc = FillNaProcessor(method="bfill")
        result = proc.process(sample_df)
        assert result["with_nan"].isna().sum() == 0
        assert result["with_nan"].iloc[1] == 3.0  # filled with next value

    def test_zero_fill(self, sample_df):
        proc = FillNaProcessor(method="zero")
        result = proc.process(sample_df)
        assert result["with_nan"].isna().sum() == 0
        assert result["with_nan"].iloc[1] == 0.0

    def test_specific_columns(self, sample_df):
        proc = FillNaProcessor(method="zero", columns=["with_nan"])
        result = proc.process(sample_df)
        assert result["with_nan"].iloc[1] == 0.0


class TestClipOutlierProcessor:
    def test_std_mode(self):
        # Create a larger sample with clear outlier
        df = pd.DataFrame(
            {
                "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1000.0],
            }
        )
        proc = ClipOutlierProcessor(mode="std", std_threshold=2.0)
        result = proc.process(df)
        # 1000 should be clipped
        assert result["values"].max() < 1000

    def test_percentile_mode(self, sample_df):
        proc = ClipOutlierProcessor(mode="percentile", lower_pct=10, upper_pct=90)
        result = proc.process(sample_df)
        assert result["outliers"].max() < 100

    def test_exclude_columns(self, sample_df):
        proc = ClipOutlierProcessor(
            mode="std", std_threshold=2.0, exclude_columns=["outliers"]
        )
        result = proc.process(sample_df)
        # outliers column should not be clipped
        assert result["outliers"].max() == 100


class TestDtypeDowncastProcessor:
    def test_float64_to_float32(self, sample_df):
        # Ensure input is float64
        df = sample_df.copy()
        df["normal"] = df["normal"].astype(np.float64)

        proc = DtypeDowncastProcessor(float_dtype="float32")
        result = proc.process(df)
        assert result["normal"].dtype == np.float32

    def test_int64_to_int32(self, sample_df):
        df = sample_df.copy()
        df["int_col"] = df["int_col"].astype(np.int64)

        proc = DtypeDowncastProcessor(int_dtype="int32")
        result = proc.process(df)
        assert result["int_col"].dtype == np.int32


class TestReplaceInfProcessor:
    def test_replace_with_nan(self, sample_df):
        proc = ReplaceInfProcessor(replace_with="nan")
        result = proc.process(sample_df)
        assert not np.isinf(result["with_inf"]).any()
        assert result["with_inf"].isna().sum() == 2  # Two inf values replaced with nan

    def test_replace_with_clip(self, sample_df):
        proc = ReplaceInfProcessor(replace_with="clip")
        result = proc.process(sample_df)
        assert not np.isinf(result["with_inf"]).any()
        assert result["with_inf"].max() == 5.0  # Clipped to max finite value
        assert result["with_inf"].min() == 1.0  # Clipped to min finite value


class TestProcessorChain:
    def test_chain_order(self, sample_df):
        # First replace inf, then fill nan
        chain = ProcessorChain(
            [
                ReplaceInfProcessor(replace_with="nan"),
                FillNaProcessor(method="ffill"),
            ]
        )
        result = chain.process(sample_df)

        # Should have no inf and no nan
        assert not np.isinf(result["with_inf"]).any()
        assert result["with_inf"].isna().sum() == 0

    def test_add_processor(self):
        chain = ProcessorChain()
        chain.add(FillNaProcessor(method="zero"))
        chain.add(DtypeDowncastProcessor())

        assert len(chain.processors) == 2

    def test_empty_chain(self, sample_df):
        chain = ProcessorChain([])
        result = chain.process(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_fluent_api(self, sample_df):
        """Test fluent/chained API."""
        chain = (
            ProcessorChain()
            .replace_inf()
            .fill_na(method="ffill")
            .clip_outliers(std_threshold=3.0)
            .downcast()
        )
        result = chain.process(sample_df)
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
        assert result["normal"].dtype == np.float32

    def test_fluent_api_with_validation(self, sample_df):
        """Test fluent API with validation."""
        chain = ProcessorChain().replace_inf().validate_quality(print_report=False)
        result = chain.process(sample_df)
        # Validation should not modify data (except inf replaced)
        reports = chain.get_validation_reports()
        assert len(reports) == 1


class TestPipeline:
    def test_pipeline_basic(self, sample_df):
        """Test Pipeline fluent API."""
        result = Pipeline(sample_df).replace_inf().fill_na().result()
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

    def test_pipeline_with_validation(self, sample_df):
        """Test Pipeline with validation."""
        pipeline = (
            Pipeline(sample_df).replace_inf().validate(quality=True, leakage=False)
        )
        result = pipeline.result()
        reports = pipeline.reports()
        assert len(reports) == 1


class TestDefaultChains:
    def test_default_chain(self, sample_df):
        chain = get_default_processor_chain()
        result = chain.process(sample_df)

        # Should handle inf and nan
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
        # Dtypes should be downcast
        assert result["normal"].dtype == np.float32

    def test_live_chain(self, sample_df):
        chain = get_live_processor_chain()
        result = chain.process(sample_df)

        # Should handle inf and nan
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()


class TestDataValidator:
    @pytest.fixture
    def df_with_issues(self):
        """DataFrame with various data quality issues."""
        return pd.DataFrame(
            {
                "good": [1.0, 2.0, 3.0, 4.0, 5.0],
                "with_nan": [1.0, np.nan, 3.0, np.nan, 5.0],
                "with_inf": [1.0, np.inf, 3.0, -np.inf, 5.0],
                "high_nan": [np.nan, np.nan, np.nan, 4.0, 5.0],
                "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

    def test_detect_nan(self, df_with_issues):
        validator = DataValidator(
            check_nan=True, check_inf=False, check_outliers=False, check_constant=False
        )
        report = validator.validate(df_with_issues)

        nan_issues = [i for i in report.issues if i.issue_type in ("nan", "high_nan")]
        assert len(nan_issues) == 2  # with_nan and high_nan

    def test_detect_inf(self, df_with_issues):
        validator = DataValidator(
            check_nan=False, check_inf=True, check_outliers=False, check_constant=False
        )
        report = validator.validate(df_with_issues)

        inf_issues = [i for i in report.issues if i.issue_type == "inf"]
        assert len(inf_issues) == 1
        assert inf_issues[0].column == "with_inf"
        assert inf_issues[0].count == 2

    def test_detect_constant(self, df_with_issues):
        validator = DataValidator(
            check_nan=False, check_inf=False, check_outliers=False, check_constant=True
        )
        report = validator.validate(df_with_issues)

        const_issues = [i for i in report.issues if i.issue_type == "constant"]
        assert len(const_issues) == 1
        assert const_issues[0].column == "constant"

    def test_high_nan_threshold(self, df_with_issues):
        validator = DataValidator(check_nan=True, nan_threshold_pct=50.0)
        report = validator.validate(df_with_issues)

        high_nan_issues = [i for i in report.issues if i.issue_type == "high_nan"]
        assert len(high_nan_issues) == 1
        assert high_nan_issues[0].column == "high_nan"

    def test_no_issues_clean_data(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        report = validate_dataframe(df, print_report=False)
        assert not report.has_issues()

    def test_report_to_dataframe(self, df_with_issues):
        report = validate_dataframe(df_with_issues, print_report=False)
        df_report = report.to_dataframe()
        assert len(df_report) > 0
        assert "column" in df_report.columns
        assert "issue_type" in df_report.columns

    def test_exclude_columns(self, df_with_issues):
        validator = DataValidator(
            exclude_columns=["with_nan", "with_inf", "high_nan", "constant"]
        )
        report = validator.validate(df_with_issues)
        # Should only check "good" column, which has no issues
        assert not report.has_issues()


class TestLookaheadLeakageDetector:
    @pytest.fixture
    def clean_features_df(self):
        """DataFrame with legitimate features (no leakage)."""
        np.random.seed(42)
        n = 300
        df = pd.DataFrame(
            {
                "close": np.cumsum(np.random.randn(n)) + 100,
                "volume": np.abs(np.random.randn(n)) * 1000,
                "sma_20": 0.0,
                "rsi": np.random.rand(n) * 100,
            }
        )
        df["sma_20"] = df["close"].rolling(20).mean()
        return df

    @pytest.fixture
    def leaky_features_df(self):
        """DataFrame with features that have lookahead leakage."""
        np.random.seed(42)
        n = 300
        df = pd.DataFrame(
            {
                "close": np.cumsum(np.random.randn(n)) + 100,
                "volume": np.abs(np.random.randn(n)) * 1000,
                "good_feature": np.random.randn(n),
            }
        )
        # Add feature that uses future returns
        future_ret = df["close"].pct_change(5).shift(-5)
        df["leaky_alpha"] = future_ret + np.random.randn(n) * 0.01
        return df

    def test_no_leakage_clean_data(self, clean_features_df):
        report = detect_lookahead_leakage(
            clean_features_df,
            target_col="close",
            print_report=False,
            exclude_columns=["close", "volume"],
        )
        # SMA and RSI should not trigger leakage
        critical = [i for i in report.issues if i.severity == "critical"]
        assert len(critical) == 0

    def test_detect_future_return_leakage(self, leaky_features_df):
        report = detect_lookahead_leakage(
            leaky_features_df,
            target_col="close",
            print_report=False,
            exclude_columns=["close", "volume"],
            future_return_threshold=0.4,
        )
        # Should detect leaky_alpha
        leaky_issues = [i for i in report.issues if i.column == "leaky_alpha"]
        assert len(leaky_issues) > 0
        assert any(i.severity == "critical" for i in leaky_issues)

    def test_report_methods(self, leaky_features_df):
        report = detect_lookahead_leakage(
            leaky_features_df,
            target_col="close",
            print_report=False,
            future_return_threshold=0.4,
        )
        # Test report methods
        assert report.columns_tested > 0
        assert report.tests_run > 0
        summary = report.summary()
        assert "total_issues" in summary
        df_report = report.to_dataframe()
        if report.has_issues():
            assert len(df_report) > 0


class TestResearchValidator:
    def test_combined_validation(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "close": np.cumsum(np.random.randn(n)) + 100,
                "good_feature": np.random.randn(n),
                "feature_with_nan": [np.nan if i % 10 == 0 else i for i in range(n)],
            }
        )

        validator = ResearchValidator(fail_on_critical=True)
        is_valid, reports = validator.validate(
            df, target_col="close", print_report=False
        )

        assert "quality" in reports
        assert "leakage" in reports
        # Should be valid (no critical leakage)
        assert is_valid
