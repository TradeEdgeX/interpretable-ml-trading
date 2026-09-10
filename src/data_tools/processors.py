"""
Data processors for consistent preprocessing.

This module provides a set of composable processors for data preprocessing,
ensuring consistency across training, backtesting, and live trading.

Usage:
    from src.data_tools.processors import ProcessorChain, FillNaProcessor, ClipOutlierProcessor

    chain = ProcessorChain([
        FillNaProcessor(method="ffill"),
        ClipOutlierProcessor(std_threshold=5.0),
        DtypeDowncastProcessor(),
    ])
    df_processed = chain.process(df)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any

import numpy as np
import pandas as pd


class Processor(ABC):
    """Base class for data processors."""

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame
        """
        pass

    @property
    def name(self) -> str:
        """Return processor name for logging."""
        return self.__class__.__name__


class FillNaProcessor(Processor):
    """
    Fill NaN values in numeric columns.

    Supports multiple fill methods:
    - "ffill": Forward fill (carry last valid value forward)
    - "bfill": Backward fill
    - "zero": Fill with 0
    - "mean": Fill with column mean
    - "median": Fill with column median
    """

    def __init__(
        self,
        method: str = "ffill",
        columns: Optional[List[str]] = None,
        fill_value: Optional[float] = None,
    ):
        """
        Initialize FillNaProcessor.

        Args:
            method: Fill method ("ffill", "bfill", "zero", "mean", "median")
            columns: Specific columns to fill (None = all numeric columns)
            fill_value: Custom fill value (overrides method if provided)
        """
        self.method = method
        self.columns = columns
        self.fill_value = fill_value

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Determine which columns to process
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if self.fill_value is not None:
                df[col] = df[col].fillna(self.fill_value)
            elif self.method == "ffill":
                df[col] = df[col].ffill()
            elif self.method == "bfill":
                df[col] = df[col].bfill()
            elif self.method == "zero":
                df[col] = df[col].fillna(0.0)
            elif self.method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif self.method == "median":
                df[col] = df[col].fillna(df[col].median())

        return df


class ClipOutlierProcessor(Processor):
    """
    Clip outliers based on standard deviation or percentile.

    Modes:
    - "std": Clip values beyond mean ± std_threshold * std
    - "percentile": Clip values beyond [lower_pct, upper_pct] percentiles
    - "iqr": Clip values beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR
    """

    def __init__(
        self,
        mode: str = "std",
        std_threshold: float = 5.0,
        lower_pct: float = 0.01,
        upper_pct: float = 99.99,
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Initialize ClipOutlierProcessor.

        Args:
            mode: Clipping mode ("std", "percentile", "iqr")
            std_threshold: Number of std deviations for "std" mode
            lower_pct: Lower percentile for "percentile" mode
            upper_pct: Upper percentile for "percentile" mode
            columns: Specific columns to clip (None = all numeric columns)
            exclude_columns: Columns to exclude from clipping
        """
        self.mode = mode
        self.std_threshold = std_threshold
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.columns = columns
        self.exclude_columns = exclude_columns or []

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Determine which columns to process
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove excluded columns
        cols = [c for c in cols if c not in self.exclude_columns]

        for col in cols:
            series = df[col]
            valid_data = series.dropna()

            if len(valid_data) == 0:
                continue

            if self.mode == "std":
                mean = valid_data.mean()
                std = valid_data.std()
                if std > 0:
                    lower = mean - self.std_threshold * std
                    upper = mean + self.std_threshold * std
                    df[col] = series.clip(lower=lower, upper=upper)

            elif self.mode == "percentile":
                lower = valid_data.quantile(self.lower_pct / 100)
                upper = valid_data.quantile(self.upper_pct / 100)
                df[col] = series.clip(lower=lower, upper=upper)

            elif self.mode == "iqr":
                q1 = valid_data.quantile(0.25)
                q3 = valid_data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[col] = series.clip(lower=lower, upper=upper)

        return df


class DtypeDowncastProcessor(Processor):
    """
    Downcast numeric dtypes to reduce memory usage.

    Converts:
    - float64 → float32
    - int64 → int32 (if values fit)
    """

    def __init__(
        self,
        float_dtype: str = "float32",
        int_dtype: str = "int32",
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Initialize DtypeDowncastProcessor.

        Args:
            float_dtype: Target float dtype
            int_dtype: Target int dtype
            columns: Specific columns to downcast (None = all numeric)
            exclude_columns: Columns to exclude from downcasting
        """
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.columns = columns
        self.exclude_columns = exclude_columns or []

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Determine which columns to process
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove excluded columns
        cols = [c for c in cols if c not in self.exclude_columns]

        for col in cols:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(self.float_dtype)
            elif df[col].dtype == np.int64:
                # Check if values fit in int32
                if (
                    df[col].min() >= np.iinfo(np.int32).min
                    and df[col].max() <= np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(self.int_dtype)

        return df


class ReplaceInfProcessor(Processor):
    """
    Replace inf/-inf values with NaN or specified values.
    """

    def __init__(
        self,
        replace_with: str = "nan",
        pos_inf_value: Optional[float] = None,
        neg_inf_value: Optional[float] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Initialize ReplaceInfProcessor.

        Args:
            replace_with: What to replace inf with ("nan", "clip", "value")
            pos_inf_value: Custom value for +inf (when replace_with="value")
            neg_inf_value: Custom value for -inf (when replace_with="value")
            columns: Specific columns to process (None = all numeric)
        """
        self.replace_with = replace_with
        self.pos_inf_value = pos_inf_value
        self.neg_inf_value = neg_inf_value
        self.columns = columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Determine which columns to process
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if self.replace_with == "nan":
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            elif self.replace_with == "clip":
                # Replace with max/min finite values
                finite_vals = df[col][np.isfinite(df[col])]
                if len(finite_vals) > 0:
                    max_val = finite_vals.max()
                    min_val = finite_vals.min()
                    df[col] = df[col].replace(np.inf, max_val)
                    df[col] = df[col].replace(-np.inf, min_val)
            elif self.replace_with == "value":
                if self.pos_inf_value is not None:
                    df[col] = df[col].replace(np.inf, self.pos_inf_value)
                if self.neg_inf_value is not None:
                    df[col] = df[col].replace(-np.inf, self.neg_inf_value)

        return df


class ProcessorChain(Processor):
    """
    Chain multiple processors together with fluent API.

    Processors are applied in order, with each processor receiving
    the output of the previous one.

    Usage (traditional):
        chain = ProcessorChain([
            ReplaceInfProcessor(),
            FillNaProcessor(),
        ])
        df = chain.process(df)

    Usage (fluent API):
        chain = (ProcessorChain()
            .replace_inf()
            .fill_na(method="ffill")
            .clip_outliers(std_threshold=5.0)
            .validate_quality()
            .downcast()
        )
        df = chain.process(df)
    """

    def __init__(self, processors: Optional[List[Processor]] = None):
        """
        Initialize ProcessorChain.

        Args:
            processors: List of processors to apply in order
        """
        self.processors = processors or []
        self._validation_reports: List[Any] = []

    def add(self, processor: Processor) -> "ProcessorChain":
        """Add a processor to the chain."""
        self.processors.append(processor)
        return self

    # =========================================================================
    # Fluent API - Processing Methods
    # =========================================================================

    def replace_inf(
        self,
        replace_with: str = "nan",
        **kwargs,
    ) -> "ProcessorChain":
        """Add ReplaceInfProcessor to chain."""
        self.processors.append(ReplaceInfProcessor(replace_with=replace_with, **kwargs))
        return self

    def fill_na(
        self,
        method: str = "ffill",
        **kwargs,
    ) -> "ProcessorChain":
        """Add FillNaProcessor to chain."""
        self.processors.append(FillNaProcessor(method=method, **kwargs))
        return self

    def clip_outliers(
        self,
        mode: str = "std",
        std_threshold: float = 5.0,
        **kwargs,
    ) -> "ProcessorChain":
        """Add ClipOutlierProcessor to chain."""
        self.processors.append(
            ClipOutlierProcessor(mode=mode, std_threshold=std_threshold, **kwargs)
        )
        return self

    def downcast(
        self,
        float_dtype: str = "float32",
        **kwargs,
    ) -> "ProcessorChain":
        """Add DtypeDowncastProcessor to chain."""
        self.processors.append(
            DtypeDowncastProcessor(float_dtype=float_dtype, **kwargs)
        )
        return self

    # =========================================================================
    # Fluent API - Validation Methods (validate but don't modify)
    # =========================================================================

    def validate_quality(
        self,
        check_nan: bool = True,
        check_inf: bool = True,
        check_outliers: bool = True,
        check_constant: bool = True,
        print_report: bool = True,
        fail_on_issues: bool = False,
        **kwargs,
    ) -> "ProcessorChain":
        """
        Add data quality validation to chain.

        Validation does not modify the data, only checks and optionally prints report.
        """
        self.processors.append(
            ValidateQualityProcessor(
                check_nan=check_nan,
                check_inf=check_inf,
                check_outliers=check_outliers,
                check_constant=check_constant,
                print_report=print_report,
                fail_on_issues=fail_on_issues,
                **kwargs,
            )
        )
        return self

    def validate_leakage(
        self,
        target_col: str = "close",
        print_report: bool = True,
        fail_on_critical: bool = False,
        **kwargs,
    ) -> "ProcessorChain":
        """
        Add lookahead leakage detection to chain.

        Detection does not modify the data, only checks and optionally prints report.
        """
        self.processors.append(
            ValidateLeakageProcessor(
                target_col=target_col,
                print_report=print_report,
                fail_on_critical=fail_on_critical,
                **kwargs,
            )
        )
        return self

    def validate(
        self,
        quality: bool = True,
        leakage: bool = False,
        target_col: str = "close",
        print_report: bool = True,
        fail_on_critical: bool = False,
    ) -> "ProcessorChain":
        """
        Add combined validation to chain.

        Shorthand for validate_quality() and/or validate_leakage().
        """
        if quality:
            self.validate_quality(print_report=print_report)
        if leakage:
            self.validate_leakage(
                target_col=target_col,
                print_report=print_report,
                fail_on_critical=fail_on_critical,
            )
        return self

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all processors in sequence."""
        result = df
        self._validation_reports = []
        for proc in self.processors:
            result = proc.process(result)
            # Collect validation reports
            if hasattr(proc, "last_report") and proc.last_report is not None:
                self._validation_reports.append(proc.last_report)
        return result

    def get_validation_reports(self) -> List[Any]:
        """Get validation reports from the last process() call."""
        return self._validation_reports

    @property
    def name(self) -> str:
        names = [p.name for p in self.processors]
        return f"ProcessorChain({' -> '.join(names)})"


# =============================================================================
# Validation Processors (inherit from Processor, don't modify data)
# =============================================================================


class ValidateQualityProcessor(Processor):
    """
    Data quality validation as a Processor.

    Does NOT modify data - only validates and optionally raises on issues.
    """

    def __init__(
        self,
        check_nan: bool = True,
        check_inf: bool = True,
        check_outliers: bool = True,
        check_constant: bool = True,
        print_report: bool = True,
        fail_on_issues: bool = False,
        **kwargs,
    ):
        self.validator = DataValidator(
            check_nan=check_nan,
            check_inf=check_inf,
            check_outliers=check_outliers,
            check_constant=check_constant,
            **kwargs,
        )
        self.print_report = print_report
        self.fail_on_issues = fail_on_issues
        self.last_report: Optional[DataQualityReport] = None

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality (does not modify df)."""
        self.last_report = self.validator.validate(df)
        if self.print_report:
            self.last_report.print_report()
        if self.fail_on_issues and self.last_report.has_issues():
            raise ValueError(
                f"Data quality issues found: {len(self.last_report.issues)} issues"
            )
        return df  # Return unchanged


class ValidateLeakageProcessor(Processor):
    """
    Lookahead leakage detection as a Processor.

    Does NOT modify data - only validates and optionally raises on critical issues.
    """

    def __init__(
        self,
        target_col: str = "close",
        print_report: bool = True,
        fail_on_critical: bool = False,
        **kwargs,
    ):
        self.detector = LookaheadLeakageDetector(**kwargs)
        self.target_col = target_col
        self.print_report = print_report
        self.fail_on_critical = fail_on_critical
        self.last_report: Optional[LookaheadLeakageReport] = None

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect lookahead leakage (does not modify df)."""
        self.last_report = self.detector.detect(df, target_col=self.target_col)
        if self.print_report:
            self.last_report.print_report()
        if self.fail_on_critical and self.last_report.has_critical_issues():
            raise ValueError(
                f"Lookahead leakage detected: {len([i for i in self.last_report.issues if i.severity == 'critical'])} critical issues"
            )
        return df  # Return unchanged


# =============================================================================
# Pipeline - Fluent API starting from DataFrame
# =============================================================================


class Pipeline:
    """
    Fluent data processing pipeline starting from a DataFrame.

    Usage:
        result = (Pipeline(df)
            .replace_inf()
            .fill_na()
            .clip_outliers()
            .validate(quality=True, leakage=True)
            .downcast()
            .result()
        )

        # Or get reports
        pipeline = Pipeline(df).replace_inf().validate(quality=True)
        result = pipeline.result()
        reports = pipeline.reports()
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._chain = ProcessorChain()

    def replace_inf(self, **kwargs) -> "Pipeline":
        self._chain.replace_inf(**kwargs)
        return self

    def fill_na(self, **kwargs) -> "Pipeline":
        self._chain.fill_na(**kwargs)
        return self

    def clip_outliers(self, **kwargs) -> "Pipeline":
        self._chain.clip_outliers(**kwargs)
        return self

    def downcast(self, **kwargs) -> "Pipeline":
        self._chain.downcast(**kwargs)
        return self

    def validate_quality(self, **kwargs) -> "Pipeline":
        self._chain.validate_quality(**kwargs)
        return self

    def validate_leakage(self, **kwargs) -> "Pipeline":
        self._chain.validate_leakage(**kwargs)
        return self

    def validate(self, **kwargs) -> "Pipeline":
        self._chain.validate(**kwargs)
        return self

    def add(self, processor: Processor) -> "Pipeline":
        """Add a custom processor."""
        self._chain.add(processor)
        return self

    def result(self) -> pd.DataFrame:
        """Execute the pipeline and return processed DataFrame."""
        return self._chain.process(self._df)

    def reports(self) -> List[Any]:
        """Get validation reports (call after result())."""
        return self._chain.get_validation_reports()


# Pre-configured processor chains for common use cases
def get_default_processor_chain() -> ProcessorChain:
    """
    Get the default processor chain for training/backtesting.

    Chain:
    1. Replace inf with NaN
    2. Forward-fill NaN
    3. Clip outliers (5 std)
    4. Downcast to float32
    """
    return ProcessorChain(
        [
            ReplaceInfProcessor(replace_with="nan"),
            FillNaProcessor(method="ffill"),
            ClipOutlierProcessor(mode="std", std_threshold=5.0),
            DtypeDowncastProcessor(float_dtype="float32"),
        ]
    )


def get_live_processor_chain() -> ProcessorChain:
    """
    Get processor chain for live trading (more conservative).

    Chain:
    1. Replace inf with NaN
    2. Forward-fill NaN (limited lookback in production)
    3. Clip outliers (3 std - more aggressive)
    """
    return ProcessorChain(
        [
            ReplaceInfProcessor(replace_with="nan"),
            FillNaProcessor(method="ffill"),
            ClipOutlierProcessor(mode="std", std_threshold=3.0),
        ]
    )


# =============================================================================
# Validation Mode - Detect issues without modifying data
# =============================================================================


class DataQualityIssue:
    """Represents a single data quality issue."""

    def __init__(
        self,
        column: str,
        issue_type: str,
        count: int,
        percentage: float,
        sample_indices: Optional[List] = None,
        details: Optional[str] = None,
    ):
        self.column = column
        self.issue_type = issue_type
        self.count = count
        self.percentage = percentage
        self.sample_indices = sample_indices or []
        self.details = details

    def __repr__(self) -> str:
        return (
            f"DataQualityIssue(column='{self.column}', type='{self.issue_type}', "
            f"count={self.count}, pct={self.percentage:.2%})"
        )


class DataQualityReport:
    """Collection of data quality issues with summary methods."""

    def __init__(self):
        self.issues: List[DataQualityIssue] = []
        self.total_rows: int = 0
        self.total_columns: int = 0

    def add_issue(self, issue: DataQualityIssue):
        self.issues.append(issue)

    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            "total_issues": len(self.issues),
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "issues_by_type": self._group_by_type(),
            "affected_columns": list(set(i.column for i in self.issues)),
        }

    def _group_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for issue in self.issues:
            counts[issue.issue_type] = counts.get(issue.issue_type, 0) + 1
        return counts

    def print_report(self, max_issues: int = 20):
        """Print a formatted report."""
        print("=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"Total rows: {self.total_rows}")
        print(f"Total columns: {self.total_columns}")
        print(f"Issues found: {len(self.issues)}")
        print()

        if not self.issues:
            print("✅ No data quality issues detected!")
            return

        # Group by type
        by_type: Dict[str, List[DataQualityIssue]] = {}
        for issue in self.issues:
            if issue.issue_type not in by_type:
                by_type[issue.issue_type] = []
            by_type[issue.issue_type].append(issue)

        for issue_type, issues in by_type.items():
            print(f"⚠️  {issue_type.upper()} ({len(issues)} columns)")
            print("-" * 40)
            for issue in issues[:max_issues]:
                print(
                    f"   {issue.column}: {issue.count} ({issue.percentage:.2%})"
                    + (f" | {issue.details}" if issue.details else "")
                )
            if len(issues) > max_issues:
                print(f"   ... and {len(issues) - max_issues} more")
            print()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert issues to DataFrame for further analysis."""
        if not self.issues:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "column": i.column,
                    "issue_type": i.issue_type,
                    "count": i.count,
                    "percentage": i.percentage,
                    "details": i.details,
                }
                for i in self.issues
            ]
        )


class DataValidator:
    """
    Validate data quality without modifying data.

    Checks for:
    - NaN values
    - Inf values
    - Outliers (beyond threshold)
    - Constant columns

    Usage:
        validator = DataValidator()
        report = validator.validate(df_features)
        report.print_report()
    """

    def __init__(
        self,
        check_nan: bool = True,
        check_inf: bool = True,
        check_outliers: bool = True,
        check_constant: bool = True,
        outlier_std_threshold: float = 5.0,
        nan_threshold_pct: float = 50.0,
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Initialize DataValidator.

        Args:
            check_nan: Check for NaN values
            check_inf: Check for inf values
            check_outliers: Check for outliers
            check_constant: Check for constant columns
            outlier_std_threshold: Threshold for outlier detection
            nan_threshold_pct: Warn separately if NaN percentage exceeds this
            columns: Specific columns to validate (None = all numeric)
            exclude_columns: Columns to exclude from validation
        """
        self.check_nan = check_nan
        self.check_inf = check_inf
        self.check_outliers = check_outliers
        self.check_constant = check_constant
        self.outlier_std_threshold = outlier_std_threshold
        self.nan_threshold_pct = nan_threshold_pct
        self.columns = columns
        self.exclude_columns = exclude_columns or ["_symbol"]

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Validate DataFrame and return quality report.

        Args:
            df: Input DataFrame

        Returns:
            DataQualityReport with all detected issues
        """
        report = DataQualityReport()
        report.total_rows = len(df)
        report.total_columns = len(df.columns)

        # Determine which columns to validate
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove excluded columns
        cols = [c for c in cols if c not in self.exclude_columns]

        for col in cols:
            series = df[col]
            total = len(series)

            if total == 0:
                continue

            # Check NaN
            if self.check_nan:
                nan_count = int(series.isna().sum())
                if nan_count > 0:
                    nan_pct = nan_count / total
                    issue_type = (
                        "high_nan" if nan_pct * 100 > self.nan_threshold_pct else "nan"
                    )
                    report.add_issue(
                        DataQualityIssue(
                            column=col,
                            issue_type=issue_type,
                            count=nan_count,
                            percentage=nan_pct,
                            sample_indices=list(series[series.isna()].index[:5]),
                            details=(
                                f">{self.nan_threshold_pct}% NaN"
                                if issue_type == "high_nan"
                                else None
                            ),
                        )
                    )

            # Check Inf
            if self.check_inf:
                inf_mask = np.isinf(series)
                inf_count = int(inf_mask.sum())
                if inf_count > 0:
                    report.add_issue(
                        DataQualityIssue(
                            column=col,
                            issue_type="inf",
                            count=inf_count,
                            percentage=inf_count / total,
                            sample_indices=list(series[inf_mask].index[:5]),
                        )
                    )

            # Check outliers
            if self.check_outliers:
                valid_data = series.dropna()
                valid_data = valid_data[np.isfinite(valid_data)]
                if len(valid_data) > 10:
                    mean = valid_data.mean()
                    std = valid_data.std()
                    if std > 0:
                        lower = mean - self.outlier_std_threshold * std
                        upper = mean + self.outlier_std_threshold * std
                        outlier_mask = (series < lower) | (series > upper)
                        outlier_count = int(outlier_mask.sum())
                        if outlier_count > 0:
                            report.add_issue(
                                DataQualityIssue(
                                    column=col,
                                    issue_type="outlier",
                                    count=outlier_count,
                                    percentage=outlier_count / total,
                                    sample_indices=list(series[outlier_mask].index[:5]),
                                    details=f"beyond {self.outlier_std_threshold}σ",
                                )
                            )

            # Check constant
            if self.check_constant:
                unique_count = series.nunique(dropna=True)
                if unique_count <= 1:
                    report.add_issue(
                        DataQualityIssue(
                            column=col,
                            issue_type="constant",
                            count=total,
                            percentage=1.0,
                            details=f"only {unique_count} unique value(s)",
                        )
                    )

        return report


def validate_dataframe(
    df: pd.DataFrame,
    print_report: bool = True,
    **kwargs,
) -> DataQualityReport:
    """
    Convenience function to validate a DataFrame.

    Args:
        df: Input DataFrame
        print_report: Whether to print the report
        **kwargs: Additional arguments for DataValidator

    Returns:
        DataQualityReport

    Usage:
        report = validate_dataframe(df_features)
        if report.has_issues():
            print("Found issues!")
    """
    validator = DataValidator(**kwargs)
    report = validator.validate(df)
    if print_report:
        report.print_report()
    return report


# =============================================================================
# Lookahead Leakage Detection (Future Data Contamination)
# =============================================================================


class LookaheadLeakageIssue:
    """Represents a potential lookahead leakage issue."""

    def __init__(
        self,
        column: str,
        test_type: str,
        metric: float,
        threshold: float,
        severity: str = "warning",
        details: Optional[str] = None,
    ):
        self.column = column
        self.test_type = test_type
        self.metric = metric
        self.threshold = threshold
        self.severity = severity  # "warning" or "critical"
        self.details = details

    def __repr__(self) -> str:
        return (
            f"LookaheadLeakageIssue(column='{self.column}', test='{self.test_type}', "
            f"metric={self.metric:.4f}, threshold={self.threshold:.4f}, severity='{self.severity}')"
        )


class LookaheadLeakageReport:
    """Report of lookahead leakage detection results."""

    def __init__(self):
        self.issues: List[LookaheadLeakageIssue] = []
        self.columns_tested: int = 0
        self.tests_run: int = 0

    def add_issue(self, issue: LookaheadLeakageIssue):
        self.issues.append(issue)

    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def has_critical_issues(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)

    def summary(self) -> Dict[str, Any]:
        return {
            "columns_tested": self.columns_tested,
            "tests_run": self.tests_run,
            "total_issues": len(self.issues),
            "critical_issues": sum(1 for i in self.issues if i.severity == "critical"),
            "warning_issues": sum(1 for i in self.issues if i.severity == "warning"),
            "issues_by_test": self._group_by_test(),
        }

    def _group_by_test(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for issue in self.issues:
            counts[issue.test_type] = counts.get(issue.test_type, 0) + 1
        return counts

    def print_report(self, max_issues: int = 20):
        """Print a formatted report."""
        print("=" * 60)
        print("LOOKAHEAD LEAKAGE DETECTION REPORT")
        print("=" * 60)
        print(f"Columns tested: {self.columns_tested}")
        print(f"Tests run: {self.tests_run}")
        print(f"Issues found: {len(self.issues)}")
        print()

        if not self.issues:
            print("✅ No lookahead leakage detected!")
            return

        # Group by severity
        critical = [i for i in self.issues if i.severity == "critical"]
        warnings = [i for i in self.issues if i.severity == "warning"]

        if critical:
            print("🚨 CRITICAL ISSUES (likely leakage)")
            print("-" * 40)
            for issue in critical[:max_issues]:
                print(
                    f"   {issue.column}: {issue.test_type} "
                    f"(metric={issue.metric:.4f} > threshold={issue.threshold:.4f})"
                )
                if issue.details:
                    print(f"      └─ {issue.details}")
            if len(critical) > max_issues:
                print(f"   ... and {len(critical) - max_issues} more")
            print()

        if warnings:
            print("⚠️  WARNINGS (potential leakage)")
            print("-" * 40)
            for issue in warnings[:max_issues]:
                print(
                    f"   {issue.column}: {issue.test_type} "
                    f"(metric={issue.metric:.4f} > threshold={issue.threshold:.4f})"
                )
            if len(warnings) > max_issues:
                print(f"   ... and {len(warnings) - max_issues} more")
            print()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert issues to DataFrame."""
        if not self.issues:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "column": i.column,
                    "test_type": i.test_type,
                    "metric": i.metric,
                    "threshold": i.threshold,
                    "severity": i.severity,
                    "details": i.details,
                }
                for i in self.issues
            ]
        )


class LookaheadLeakageDetector:
    """
    Detect potential lookahead (future data) leakage in features.

    Tests performed:
    1. Future correlation: Check if feature correlates too well with future returns
    2. Perfect prediction: Check if feature perfectly predicts future values
    3. Information coefficient: Check if IC is suspiciously high
    4. Shift invariance: Check if feature at t contains info about t+1

    Usage:
        detector = LookaheadLeakageDetector()
        report = detector.detect(df_features, target_col='close')
        report.print_report()
    """

    def __init__(
        self,
        future_return_threshold: float = 0.5,
        perfect_pred_threshold: float = 0.99,
        ic_threshold: float = 0.3,
        shift_corr_threshold: float = 0.95,
        forward_periods: List[int] = None,
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Initialize LookaheadLeakageDetector.

        Args:
            future_return_threshold: Correlation threshold with future returns (critical)
            perfect_pred_threshold: Threshold for perfect prediction (critical)
            ic_threshold: Information coefficient threshold (warning)
            shift_corr_threshold: Correlation with shifted self (critical)
            forward_periods: Periods to check for leakage (default: [1, 5, 10])
            columns: Specific columns to test (None = all numeric)
            exclude_columns: Columns to exclude from testing
        """
        self.future_return_threshold = future_return_threshold
        self.perfect_pred_threshold = perfect_pred_threshold
        self.ic_threshold = ic_threshold
        self.shift_corr_threshold = shift_corr_threshold
        self.forward_periods = forward_periods or [1, 5, 10]
        self.columns = columns
        self.exclude_columns = exclude_columns or [
            "_symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def detect(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
    ) -> LookaheadLeakageReport:
        """
        Run lookahead leakage detection.

        Args:
            df: DataFrame with features
            target_col: Target column to compute returns from

        Returns:
            LookaheadLeakageReport with detected issues
        """
        report = LookaheadLeakageReport()

        # Determine which columns to test
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove excluded columns
        cols = [c for c in cols if c not in self.exclude_columns]
        report.columns_tested = len(cols)

        if target_col not in df.columns:
            print(
                f"Warning: target_col '{target_col}' not found, skipping return-based tests"
            )
            target_col = None

        for col in cols:
            feature = df[col].dropna()
            if len(feature) < 100:
                continue

            # Test 1: Future return correlation
            if target_col:
                for period in self.forward_periods:
                    future_ret = df[target_col].pct_change(period).shift(-period)
                    valid_mask = feature.notna() & future_ret.notna()
                    if valid_mask.sum() > 50:
                        corr = feature[valid_mask].corr(future_ret[valid_mask])
                        report.tests_run += 1
                        if abs(corr) > self.future_return_threshold:
                            report.add_issue(
                                LookaheadLeakageIssue(
                                    column=col,
                                    test_type=f"future_return_corr_{period}",
                                    metric=abs(corr),
                                    threshold=self.future_return_threshold,
                                    severity="critical",
                                    details=f"Feature correlates {corr:.3f} with {period}-bar future return",
                                )
                            )

            # Test 2: Perfect prediction of future feature value
            # Skip this test for slow-moving features (high autocorrelation is normal)
            # Only flag if forward corr is HIGHER than backward corr
            for period in self.forward_periods[:2]:  # Only check 1 and 5
                future_val = df[col].shift(-period)
                past_val = df[col].shift(period)
                valid_fwd = feature.notna() & future_val.notna()
                valid_bwd = feature.notna() & past_val.notna()
                if valid_fwd.sum() > 50 and valid_bwd.sum() > 50:
                    corr_fwd = feature[valid_fwd].corr(future_val[valid_fwd])
                    corr_bwd = feature[valid_bwd].corr(past_val[valid_bwd])
                    report.tests_run += 1
                    # Only flag if forward corr > backward corr (suspicious asymmetry)
                    if (
                        abs(corr_fwd) > self.perfect_pred_threshold
                        and corr_fwd > corr_bwd + 0.01
                    ):
                        report.add_issue(
                            LookaheadLeakageIssue(
                                column=col,
                                test_type=f"perfect_prediction_{period}",
                                metric=abs(corr_fwd),
                                threshold=self.perfect_pred_threshold,
                                severity="critical",
                                details=f"corr(t,t+{period})={corr_fwd:.3f} > corr(t,t-{period})={corr_bwd:.3f}",
                            )
                        )

            # Test 3: High IC (Information Coefficient) - suspiciously high
            if target_col:
                for period in self.forward_periods[:1]:  # Only check 1-bar
                    future_ret = df[target_col].pct_change(1).shift(-period)
                    valid_mask = feature.notna() & future_ret.notna()
                    if valid_mask.sum() > 100:
                        # Rank IC
                        f_rank = feature[valid_mask].rank(pct=True)
                        r_rank = future_ret[valid_mask].rank(pct=True)
                        ic = f_rank.corr(r_rank)
                        report.tests_run += 1
                        if abs(ic) > self.ic_threshold:
                            report.add_issue(
                                LookaheadLeakageIssue(
                                    column=col,
                                    test_type="high_rank_ic",
                                    metric=abs(ic),
                                    threshold=self.ic_threshold,
                                    severity="warning",
                                    details=f"Rank IC = {ic:.3f} is suspiciously high",
                                )
                            )

            # Test 4: Shift invariance (feature[t] ~ feature[t-1] is normal,
            #         but feature[t] ~ feature[t+1] at >0.95 is suspicious)
            feature_forward = df[col].shift(-1)
            valid_mask = feature.notna() & feature_forward.notna()
            if valid_mask.sum() > 50:
                corr_forward = feature[valid_mask].corr(feature_forward[valid_mask])
                report.tests_run += 1
                # High correlation with FUTURE value is suspicious
                # (unless it's a slow-moving feature, so we check if it's higher than lag-1 corr)
                feature_backward = df[col].shift(1)
                valid_back = feature.notna() & feature_backward.notna()
                if valid_back.sum() > 50:
                    corr_backward = feature[valid_back].corr(
                        feature_backward[valid_back]
                    )
                    # If forward corr is significantly higher than backward, suspicious
                    if (
                        corr_forward > self.shift_corr_threshold
                        and corr_forward > corr_backward + 0.02
                    ):
                        report.add_issue(
                            LookaheadLeakageIssue(
                                column=col,
                                test_type="forward_shift_anomaly",
                                metric=corr_forward,
                                threshold=self.shift_corr_threshold,
                                severity="critical",
                                details=f"corr(t,t+1)={corr_forward:.3f} > corr(t,t-1)={corr_backward:.3f}",
                            )
                        )

        return report


def detect_lookahead_leakage(
    df: pd.DataFrame,
    target_col: str = "close",
    print_report: bool = True,
    **kwargs,
) -> LookaheadLeakageReport:
    """
    Convenience function to detect lookahead leakage.

    Args:
        df: DataFrame with features
        target_col: Target column for return calculation
        print_report: Whether to print the report
        **kwargs: Additional arguments for LookaheadLeakageDetector

    Returns:
        LookaheadLeakageReport

    Usage:
        report = detect_lookahead_leakage(df_features)
        if report.has_critical_issues():
            raise ValueError("Lookahead leakage detected!")
    """
    detector = LookaheadLeakageDetector(**kwargs)
    report = detector.detect(df, target_col=target_col)
    if print_report:
        report.print_report()
    return report


# =============================================================================
# Research Validation Suite
# =============================================================================


class ResearchValidator:
    """
    Comprehensive validation for research/backtesting.

    Combines data quality checks and lookahead leakage detection.

    Usage:
        validator = ResearchValidator()
        is_valid, report = validator.validate(df_features, target_col='close')
    """

    def __init__(
        self,
        quality_config: Optional[Dict[str, Any]] = None,
        leakage_config: Optional[Dict[str, Any]] = None,
        fail_on_critical: bool = True,
    ):
        """
        Initialize ResearchValidator.

        Args:
            quality_config: Config for DataValidator
            leakage_config: Config for LookaheadLeakageDetector
            fail_on_critical: Whether to return False if critical issues found
        """
        self.quality_config = quality_config or {}
        self.leakage_config = leakage_config or {}
        self.fail_on_critical = fail_on_critical

    def validate(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        print_report: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run full validation suite.

        Args:
            df: DataFrame to validate
            target_col: Target column for leakage detection
            print_report: Whether to print reports

        Returns:
            Tuple of (is_valid, reports_dict)
        """
        # Run quality validation
        quality_validator = DataValidator(**self.quality_config)
        quality_report = quality_validator.validate(df)

        # Run leakage detection
        leakage_detector = LookaheadLeakageDetector(**self.leakage_config)
        leakage_report = leakage_detector.detect(df, target_col=target_col)

        if print_report:
            quality_report.print_report()
            print()
            leakage_report.print_report()

        # Determine if valid
        is_valid = True
        if self.fail_on_critical:
            if leakage_report.has_critical_issues():
                is_valid = False

        return is_valid, {
            "quality": quality_report,
            "leakage": leakage_report,
        }
