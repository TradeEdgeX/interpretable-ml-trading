"""FeatureStore layer newer than experiment date → promote evidence stale."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from src.research.evidence_staleness import (
    evaluate_staleness,
    experiment_date_from_id,
    layer_rebuilt_at,
    scan_promote_staleness,
)
from src.research.experiment_gate import evaluate_experiment


def _write_promote(exp: Path, *, layer: str | None = None) -> None:
    exp.mkdir(parents=True)
    extra = f"feature_store_layer: {layer}\n" if layer else ""
    (exp / "DECISION.md").write_text(
        "---\n"
        "strategy: srb\n"
        "harness: event_backtest\n"
        "segments: [bear_2022, bull_2023_2024, recent_range_to_bear]\n"
        "kill_switch: false\n"
        f"{extra}"
        "verdict: promote\n"
        "---\n",
        encoding="utf-8",
    )


def test_experiment_date_from_id() -> None:
    assert experiment_date_from_id("20260805_srb_risk_vs_add_size") == date(2026, 8, 5)
    assert experiment_date_from_id("not_a_date") is None


def test_layer_prefers_last_write_at_stamp(tmp_path: Path) -> None:
    store = tmp_path / "feature_store"
    store.mkdir()
    (store / "features_srb_120T_deadbeef01.meta.json").write_text(
        json.dumps({"last_write_at": "2026-08-23T15:50:00Z"}),
        encoding="utf-8",
    )
    got = layer_rebuilt_at(store, "features_srb_120T_deadbeef01")
    assert got == date(2026, 8, 23)


def test_layer_uses_newer_parquet_even_when_metadata_stamp_is_older(
    tmp_path: Path,
) -> None:
    store = tmp_path / "feature_store"
    layer = "features_srb_120T_deadbeef01"
    store.mkdir()
    (store / f"{layer}.meta.json").write_text(
        json.dumps({"last_write_at": "2026-08-23T15:50:00Z"}),
        encoding="utf-8",
    )
    month = store / layer / "BTCUSDT" / "120T"
    month.mkdir(parents=True)
    pq = month / "2024-01.parquet"
    pq.write_bytes(b"parquet")
    stamp = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc).timestamp()
    import os

    os.utime(pq, (stamp, stamp))

    assert layer_rebuilt_at(store, layer) == date(2026, 9, 1)


def test_promote_is_stale_when_layer_newer(tmp_path: Path) -> None:
    repo = tmp_path
    exp = repo / "config" / "experiments" / "20260805_srb_risk_vs_add_size"
    layer = "features_srb_120T_deadbeef01"
    _write_promote(exp, layer=layer)
    store = repo / "feature_store"
    store.mkdir()
    (store / f"{layer}.meta.json").write_text(
        json.dumps({"last_write_at": "2026-08-23T15:50:00Z"}),
        encoding="utf-8",
    )
    row = evaluate_staleness(exp, repo_root=repo, store_root=store)
    assert row.stale
    assert row.layer == layer
    assert "2026-08-23" in row.reasons[0]


def test_promote_is_fresh_when_layer_same_day_or_older(tmp_path: Path) -> None:
    repo = tmp_path
    exp = repo / "config" / "experiments" / "20260823_srb_replay"
    layer = "features_srb_120T_deadbeef01"
    _write_promote(exp, layer=layer)
    store = repo / "feature_store"
    store.mkdir()
    (store / f"{layer}.meta.json").write_text(
        json.dumps({"last_write_at": "2026-08-23T15:50:00Z"}),
        encoding="utf-8",
    )
    row = evaluate_staleness(exp, repo_root=repo, store_root=store)
    assert row.status == "fresh"
    assert not row.stale


def test_unknown_when_layer_missing(tmp_path: Path) -> None:
    repo = tmp_path
    exp = repo / "config" / "experiments" / "20260805_ashare_rank_ops"
    _write_promote(exp)
    (exp / "DECISION.md").write_text(
        "---\nstrategy: ashare_oversold\nharness: ashare_slot_policy\n"
        "segments: [bear_2022, bull_2023_2024, recent_range_to_bear]\n"
        "kill_switch: false\nverdict: promote\n---\n",
        encoding="utf-8",
    )
    row = evaluate_staleness(exp, repo_root=repo, store_root=repo / "feature_store")
    assert row.status == "unknown"
    assert not row.stale


def test_gate_row_flags_stale_promote(tmp_path: Path) -> None:
    repo = tmp_path
    exp = repo / "config" / "experiments" / "20260805_srb_demo"
    layer = "features_srb_120T_deadbeef01"
    _write_promote(exp, layer=layer)
    (exp / "capital_report.json").write_text(
        json.dumps({"cagr": 0.1, "max_drawdown_pct": -0.2}),
        encoding="utf-8",
    )
    store = repo / "feature_store"
    store.mkdir()
    (store / f"{layer}.meta.json").write_text(
        json.dumps({"last_write_at": "2026-09-01T00:00:00Z"}),
        encoding="utf-8",
    )
    gate = evaluate_experiment(exp, repo_root=repo)
    assert gate.evidence_stale is True
    assert gate.feature_store_layer == layer
    assert any("feature store layer" in r for r in gate.stale_reasons)


def test_scan_skips_reject(tmp_path: Path) -> None:
    repo = tmp_path
    root = repo / "config" / "experiments"
    promo = root / "20260805_keep"
    _write_promote(promo, layer="features_srb_120T_deadbeef01")
    reject = root / "20260806_drop"
    reject.mkdir(parents=True)
    (reject / "DECISION.md").write_text(
        "---\nstrategy: srb\nharness: event_backtest\n"
        "segments: [bear_2022, bull_2023_2024, recent_range_to_bear]\n"
        "kill_switch: false\nverdict: reject\n---\n",
        encoding="utf-8",
    )
    store = repo / "feature_store"
    store.mkdir()
    (store / "features_srb_120T_deadbeef01.meta.json").write_text(
        json.dumps({"last_write_at": "2026-09-01T00:00:00Z"}),
        encoding="utf-8",
    )
    rows = scan_promote_staleness(
        repo_root=repo, experiments_root=root, store_root=store
    )
    assert [r.experiment_id for r in rows] == ["20260805_keep"]
    assert rows[0].stale


def test_parquet_mtime_used_when_stamp_absent(tmp_path: Path) -> None:
    store = tmp_path / "feature_store"
    month = store / "features_srb_120T_deadbeef01" / "BTCUSDT" / "120T"
    month.mkdir(parents=True)
    pq = month / "2024-01.parquet"
    pq.write_bytes(b"parquet")
    stamp = datetime(2026, 8, 23, 15, 50, tzinfo=timezone.utc).timestamp()
    import os

    os.utime(pq, (stamp, stamp))
    got = layer_rebuilt_at(store, "features_srb_120T_deadbeef01")
    assert got == date(2026, 8, 23)
