import json
from pathlib import Path

import scripts.auto_research_pipeline as arp
import scripts.plot_monthly_threshold_drift as drift_report


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_semantic_guard_repairs_missing_prefilter_locked_rule(tmp_path: Path):
    current_root = tmp_path / "current"
    candidate_root = tmp_path / "candidate"
    strategy = "bpc"
    _write_yaml(
        current_root / strategy / "archetypes" / "prefilter.yaml",
        """
rules:
  - feature: bpc_recent_breakout_strength
    operator: ">="
    value: 0.25
    locked: true
""",
    )
    _write_yaml(
        candidate_root / strategy / "archetypes" / "prefilter.yaml",
        """
rules:
  - feature: bpc_volume_compression_pct
    operator: ">="
    value: 0.8
""",
    )
    _write_yaml(current_root / strategy / "archetypes" / "gate.yaml", "hard_gates: []")
    _write_yaml(
        candidate_root / strategy / "archetypes" / "gate.yaml", "hard_gates: []"
    )

    report = arp._evaluate_semantic_contract_for_strategy(
        strategy=strategy,
        current_root=current_root,
        candidate_root=candidate_root,
        semantic_cfg={
            "enabled": True,
            "on_prefilter_locked_missing": "repair_and_continue",
            "on_gate_locked_violation": "red_no_adopt",
        },
        report_path=tmp_path / "semantic_guard.json",
    )
    merged_prefilter = (
        candidate_root / strategy / "archetypes" / "prefilter.yaml"
    ).read_text(encoding="utf-8")
    assert report["level"] == "green"
    assert report["prefilter_locked_repaired"] == 1
    assert "bpc_recent_breakout_strength" in merged_prefilter


def test_semantic_guard_marks_red_when_locked_gate_disabled(tmp_path: Path):
    current_root = tmp_path / "current"
    candidate_root = tmp_path / "candidate"
    strategy = "bpc"
    _write_yaml(
        current_root / strategy / "archetypes" / "prefilter.yaml",
        """
rules:
  - feature: bpc_recent_breakout_strength
    operator: ">="
    value: 0.25
    locked: true
""",
    )
    _write_yaml(
        candidate_root / strategy / "archetypes" / "prefilter.yaml",
        """
rules:
  - feature: bpc_recent_breakout_strength
    operator: ">="
    value: 0.25
    locked: true
""",
    )
    _write_yaml(
        current_root / strategy / "archetypes" / "gate.yaml",
        """
hard_gates:
  - id: gate_locked_a
    locked: true
    when:
      foo:
        value_gt: 1.0
""",
    )
    _write_yaml(
        candidate_root / strategy / "archetypes" / "gate.yaml",
        """
hard_gates:
  - id: gate_locked_a
    locked: true
    disabled: true
    when:
      foo:
        value_gt: 1.0
""",
    )
    report = arp._evaluate_semantic_contract_for_strategy(
        strategy=strategy,
        current_root=current_root,
        candidate_root=candidate_root,
        semantic_cfg={
            "enabled": True,
            "on_prefilter_locked_missing": "repair_and_continue",
            "on_gate_locked_violation": "red_no_adopt",
        },
        report_path=tmp_path / "semantic_guard_red.json",
    )
    assert report["level"] == "red"
    assert report["gate_locked_disabled"] == ["gate_locked_a"]


def test_prefilter_drift_guard_reports_red_on_large_nonlocked_change(tmp_path: Path):
    strategy = "bpc"
    current_prefilter = tmp_path / "current.yaml"
    candidate_prefilter = tmp_path / "candidate.yaml"
    _write_yaml(
        current_prefilter,
        """
rules:
  - feature: bpc_volume_compression_pct
    operator: ">="
    value: 1.0
  - feature: bpc_recent_breakout_strength
    operator: ">="
    value: 0.25
    locked: true
""",
    )
    _write_yaml(
        candidate_prefilter,
        """
rules:
  - feature: bpc_volume_compression_pct
    operator: ">="
    value: 1.7
  - feature: bpc_recent_breakout_strength
    operator: ">="
    value: 0.10
    locked: true
""",
    )
    report = arp._evaluate_prefilter_drift_for_strategy(
        strategy=strategy,
        current_prefilter=current_prefilter,
        candidate_prefilter=candidate_prefilter,
        drift_cfg={
            "enabled": True,
            "warn_relative_change": 0.20,
            "max_relative_change": 0.35,
        },
        report_path=tmp_path / "prefilter_drift.json",
    )
    assert report["level"] == "red"
    assert "rule_0" in report["red_rules"]
    # locked rule change should not be counted as normal drift
    assert all("rule_1" not in x for x in report["red_rules"])


def test_threshold_report_writes_drift_artifacts_with_adoption_status(
    tmp_path: Path, monkeypatch
):
    run_root = tmp_path / "run"
    strategy = "bpc"
    _write_yaml(
        run_root
        / "fast_month_2024-01"
        / "strategies_calibrated"
        / strategy
        / "archetypes"
        / "prefilter.yaml",
        """
rules:
  - feature: bpc_volume_compression_pct
    operator: ">="
    value: 1.0
""",
    )
    _write_yaml(
        run_root
        / "fast_month_2024-01"
        / "strategies_calibrated"
        / strategy
        / "archetypes"
        / "gate.yaml",
        "hard_gates: []",
    )
    _write_yaml(
        run_root
        / "fast_month_2024-01"
        / "strategies_calibrated"
        / strategy
        / "archetypes"
        / "entry_filters.yaml",
        "filters: []",
    )
    _write_yaml(
        run_root
        / "fast_month_2024-02"
        / "strategies_calibrated"
        / strategy
        / "archetypes"
        / "prefilter.yaml",
        """
rules:
  - feature: bpc_volume_compression_pct
    operator: ">="
    value: 1.6
""",
    )
    _write_yaml(
        run_root
        / "fast_month_2024-02"
        / "strategies_calibrated"
        / strategy
        / "archetypes"
        / "gate.yaml",
        "hard_gates: []",
    )
    _write_yaml(
        run_root
        / "fast_month_2024-02"
        / "strategies_calibrated"
        / strategy
        / "archetypes"
        / "entry_filters.yaml",
        "filters: []",
    )
    (run_root / "monthly_ledger.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "month": "2024-01",
                        "slow_guard_by_strategy": {
                            strategy: {
                                "guard_level": "green",
                                "adoption_status": "adopted",
                                "fallback_used": False,
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "month": "2024-02",
                        "slow_guard_by_strategy": {
                            strategy: {
                                "guard_level": "red",
                                "adoption_status": "fallback_previous",
                                "fallback_used": True,
                                "reason": "red_prefilter_drift",
                            }
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_monthly_threshold_drift.py",
            "--run-root",
            str(run_root),
            "--strategy",
            strategy,
            "--yellow-rel",
            "0.20",
            "--red-rel",
            "0.35",
        ],
    )
    rc = drift_report.main()
    assert rc == 0
    out_dir = run_root / "threshold_tracking" / strategy
    drift_json = json.loads(
        (out_dir / "threshold_drift_report.json").read_text(encoding="utf-8")
    )
    assert (out_dir / "threshold_drift_report.csv").exists()
    assert (out_dir / "threshold_timeseries.html").exists()
    assert drift_json["drift_rows"][0]["adoption_status"] == "fallback_previous"


def test_resolve_slow_adoption_gate_cfg_includes_semantic_and_drift() -> None:
    cfg = {
        "rolling": {
            "slow_realistic": {
                "adoption_gate": {
                    "enabled": True,
                    "validation_months": 2,
                    "semantic_guard": {
                        "enabled": False,
                        "on_prefilter_locked_missing": "red_no_adopt",
                    },
                    "prefilter_drift_guard": {
                        "enabled": True,
                        "warn_relative_change": 0.15,
                        "max_relative_change": 0.42,
                        "watched_features": ["bpc_volume_compression_pct"],
                        "on_red_drift": "previous_accepted",
                    },
                },
            },
        },
    }
    g = arp._resolve_slow_adoption_gate_cfg(cfg)
    assert g["enabled"] is True
    assert g["validation_months"] == 2
    assert g["semantic_guard"]["enabled"] is False
    assert g["semantic_guard"]["on_prefilter_locked_missing"] == "red_no_adopt"
    assert g["prefilter_drift_guard"]["enabled"] is True
    assert g["prefilter_drift_guard"]["warn_relative_change"] == 0.15
    assert g["prefilter_drift_guard"]["max_relative_change"] == 0.42
    assert g["prefilter_drift_guard"]["watched_features"] == [
        "bpc_volume_compression_pct"
    ]
    assert g["prefilter_drift_guard"]["on_red_drift"] == "previous_accepted"


def test_prefilter_drift_guard_yellow_between_warn_and_max(tmp_path: Path) -> None:
    strategy = "bpc"
    current_prefilter = tmp_path / "cur.yaml"
    candidate_prefilter = tmp_path / "cand.yaml"
    _write_yaml(
        current_prefilter,
        """
rules:
  - feature: bpc_volume_compression_pct
    operator: ">="
    value: 1.0
""",
    )
    _write_yaml(
        candidate_prefilter,
        """
rules:
  - feature: bpc_volume_compression_pct
    operator: ">="
    value: 1.24
""",
    )
    report = arp._evaluate_prefilter_drift_for_strategy(
        strategy=strategy,
        current_prefilter=current_prefilter,
        candidate_prefilter=candidate_prefilter,
        drift_cfg={
            "enabled": True,
            "warn_relative_change": 0.20,
            "max_relative_change": 0.35,
        },
        report_path=tmp_path / "drift.json",
    )
    assert report["level"] == "yellow"
    assert report["yellow_rules"]
    assert not report["red_rules"]


def test_prefilter_drift_guard_watched_features_restricts_evaluation(
    tmp_path: Path,
) -> None:
    strategy = "bpc"
    current_prefilter = tmp_path / "cur.yaml"
    candidate_prefilter = tmp_path / "cand.yaml"
    _write_yaml(
        current_prefilter,
        """
rules:
  - feature: bpc_volume_compression_pct
    operator: ">="
    value: 1.0
  - feature: bpc_recent_breakout_strength
    operator: ">="
    value: 0.25
""",
    )
    _write_yaml(
        candidate_prefilter,
        """
rules:
  - feature: bpc_volume_compression_pct
    operator: ">="
    value: 2.0
  - feature: bpc_recent_breakout_strength
    operator: ">="
    value: 0.05
""",
    )
    report = arp._evaluate_prefilter_drift_for_strategy(
        strategy=strategy,
        current_prefilter=current_prefilter,
        candidate_prefilter=candidate_prefilter,
        drift_cfg={
            "enabled": True,
            "watched_features": ["bpc_volume_compression_pct"],
            "warn_relative_change": 0.20,
            "max_relative_change": 0.35,
        },
        report_path=tmp_path / "drift_watch.json",
    )
    feats = {r["feature"] for r in report["rows"]}
    assert feats == {"bpc_volume_compression_pct"}
    assert report["level"] == "red"


def test_semantic_guard_disabled_short_circuits_greenish(tmp_path: Path) -> None:
    strategy = "bpc"
    current_root = tmp_path / "current"
    candidate_root = tmp_path / "candidate"
    _write_yaml(
        current_root / strategy / "archetypes" / "prefilter.yaml",
        """
rules:
  - feature: bpc_recent_breakout_strength
    operator: ">="
    value: 0.25
    locked: true
""",
    )
    _write_yaml(
        candidate_root / strategy / "archetypes" / "prefilter.yaml", "rules: []"
    )
    _write_yaml(current_root / strategy / "archetypes" / "gate.yaml", "hard_gates: []")
    _write_yaml(
        candidate_root / strategy / "archetypes" / "gate.yaml", "hard_gates: []"
    )

    report = arp._evaluate_semantic_contract_for_strategy(
        strategy=strategy,
        current_root=current_root,
        candidate_root=candidate_root,
        semantic_cfg={"enabled": False},
        report_path=tmp_path / "off.json",
    )
    assert report["level"] == "none"


def test_load_month_status_reads_slow_guard_by_strategy(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    strat = "bpc-long-120T"
    (run_root / "monthly_ledger.jsonl").write_text(
        json.dumps(
            {
                "month": "2024-06",
                "slow_guard_by_strategy": {
                    strat: {
                        "guard_level": "yellow",
                        "adoption_status": "adopted",
                        "fallback_used": False,
                        "reason": "",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ms = drift_report._load_month_status(run_root, strat)
    assert ms["2024-06"]["guard_level"] == "yellow"
    assert ms["2024-06"]["adoption_status"] == "adopted"


def test_compute_drift_rows_assigns_green_yellow_red() -> None:
    rows = [
        {
            "month": "2024-01",
            "strategy": "bpc",
            "layer": "prefilter",
            "key": "k",
            "feature": "x",
            "operator": ">=",
            "value": 100.0,
            "rule_id": "rule_0",
            "locked": False,
            "enabled": True,
        },
        {
            "month": "2024-02",
            "strategy": "bpc",
            "layer": "prefilter",
            "key": "k",
            "feature": "x",
            "operator": ">=",
            "value": 110.0,
            "rule_id": "rule_0",
            "locked": False,
            "enabled": True,
        },
        {
            "month": "2024-03",
            "strategy": "bpc",
            "layer": "prefilter",
            "key": "k",
            "feature": "x",
            "operator": ">=",
            "value": 145.0,
            "rule_id": "rule_0",
            "locked": False,
            "enabled": True,
        },
    ]
    month_status = {"2024-02": {}, "2024-03": {}}
    out = drift_report._compute_drift_rows(
        rows,
        month_status,
        yellow_rel=0.15,
        red_rel=0.40,
    )
    # 2024-02 vs 01: +10% < 15% yellow band -> green
    # 2024-03 vs 02: +35/110 ~ 31.8% -> yellow (< 40% red)
    by_month = {(r["month"], r["drift_level"]) for r in out}
    assert ("2024-02", "green") in by_month
    assert ("2024-03", "yellow") in by_month
