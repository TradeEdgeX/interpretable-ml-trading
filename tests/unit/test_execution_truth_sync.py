from __future__ import annotations

import pytest

from src.order_management.execution_truth_sync import (
    RECONCILIATION_ISSUE_BUCKETS,
    SELF_HEALING_RECONCILIATION_ISSUES,
    UNRESOLVED_RECONCILIATION_ISSUES,
    publish_reconciliation_metrics,
    reconciliation_ok_from_issues,
    self_healing_issue_promql_re,
)


@pytest.mark.parametrize(
    "issue",
    sorted(UNRESOLVED_RECONCILIATION_ISSUES),
)
def test_reconciliation_ok_false_for_each_unresolved_issue(issue: str) -> None:
    assert not reconciliation_ok_from_issues({issue: 1})


def test_reconciliation_ok_true_when_only_self_healing_issues_present() -> None:
    assert reconciliation_ok_from_issues({"open_reconcile_updated": 99})
    assert "open_reconcile_updated" in SELF_HEALING_RECONCILIATION_ISSUES


def test_self_healing_issue_promql_re_covers_all_self_healing_buckets() -> None:
    matcher = self_healing_issue_promql_re()
    assert matcher == "duplicate_position_row_closed|open_reconcile_updated"
    for issue in SELF_HEALING_RECONCILIATION_ISSUES:
        assert issue in matcher.split("|")


def test_reconciliation_ok_treats_invalid_counts_as_zero() -> None:
    assert reconciliation_ok_from_issues({"stale_local_order": "bad"})
    assert reconciliation_ok_from_issues({"api_error": None})


def test_publish_reconciliation_metrics_derives_ok_when_not_provided(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _FakeMetrics:
        def update_reconciliation_metrics(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "src.time_series_model.live.metrics_exporter.METRICS",
        _FakeMetrics(),
    )

    publish_reconciliation_metrics(
        scope="trend",
        strategy="all",
        symbol="ALL",
        issue_counts={"open_reconcile_updated": 4, "api_error": 0},
        source="test",
    )

    assert len(calls) == 1
    assert calls[0]["ok"] is True
    assert calls[0]["issue_counts"] == {
        "open_reconcile_updated": 4,
        "api_error": 0,
    }


def test_publish_reconciliation_metrics_honors_explicit_ok_override(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _FakeMetrics:
        def update_reconciliation_metrics(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "src.time_series_model.live.metrics_exporter.METRICS",
        _FakeMetrics(),
    )

    publish_reconciliation_metrics(
        scope="hedge",
        strategy="chop_grid",
        symbol="BTCUSDT",
        ok=False,
        issue_counts={"open_reconcile_updated": 0},
        source="test",
    )

    assert calls[0]["ok"] is False


def test_publish_reconciliation_metrics_passes_timestamp(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _FakeMetrics:
        def update_reconciliation_metrics(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "src.time_series_model.live.metrics_exporter.METRICS",
        _FakeMetrics(),
    )

    publish_reconciliation_metrics(
        scope="spot",
        strategy="accum",
        symbol="ALL",
        issue_counts={},
        source="test",
        ts_seconds=1_700_000_123.0,
    )

    assert calls[0]["ts_seconds"] == 1_700_000_123.0


def test_publish_reconciliation_metrics_swallows_metrics_errors(
    monkeypatch,
) -> None:
    class _BrokenMetrics:
        def update_reconciliation_metrics(self, **kwargs) -> None:
            raise RuntimeError("metrics down")

    monkeypatch.setattr(
        "src.time_series_model.live.metrics_exporter.METRICS",
        _BrokenMetrics(),
    )

    publish_reconciliation_metrics(
        scope="trend",
        strategy="all",
        symbol="ALL",
        issue_counts={"api_error": 1},
        source="test",
    )


def test_reconciliation_buckets_are_stable_tuple() -> None:
    assert isinstance(RECONCILIATION_ISSUE_BUCKETS, tuple)
    assert reconciliation_ok_from_issues({}) is True
    assert len(RECONCILIATION_ISSUE_BUCKETS) == len(
        UNRESOLVED_RECONCILIATION_ISSUES
    ) + len(SELF_HEALING_RECONCILIATION_ISSUES)


def test_p3_trend_buckets_present() -> None:
    """P3: Trend-specific buckets are in RECONCILIATION_ISSUE_BUCKETS."""
    assert "bootstrap_from_exchange" in RECONCILIATION_ISSUE_BUCKETS
    assert "duplicate_position_row_closed" in RECONCILIATION_ISSUE_BUCKETS
    assert "sqlite_orphan_open" in RECONCILIATION_ISSUE_BUCKETS
    assert "tracker_exchange_qty_mismatch" in RECONCILIATION_ISSUE_BUCKETS


def test_p3_duplicate_position_row_closed_is_self_healing() -> None:
    """P3: duplicate_position_row_closed is self-healing (does not flip ok=False)."""
    assert "duplicate_position_row_closed" in SELF_HEALING_RECONCILIATION_ISSUES
    assert reconciliation_ok_from_issues({"duplicate_position_row_closed": 5})


def test_p3_sqlite_orphan_open_is_unresolved() -> None:
    """P3: sqlite_orphan_open is unresolved (flips ok=False)."""
    assert "sqlite_orphan_open" in UNRESOLVED_RECONCILIATION_ISSUES
    assert not reconciliation_ok_from_issues({"sqlite_orphan_open": 1})
