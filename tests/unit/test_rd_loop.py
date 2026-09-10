"""Smoke test: rd_loop step dispatch (mocked subprocess)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.rd_loop import run_loop


def test_rd_loop_runs_three_steps(tmp_path: Path) -> None:
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "test_loop",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "condition-set",
                        "features_parquet": "dummy.parquet",
                        "condition": ["H: x>0"],
                    }
                ],
                "variant_grid": "config/experiments/_smoke/tpc_variant_grid_smoke.yaml",
                "decision_doc": {
                    "topic": "test_loop",
                    "topic_template": "default",
                    "experiment_index": "idx.json",
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        rc = run_loop(hyp, output_dir=tmp_path / "out")

    assert rc == 0
    assert len(calls) == 3
    joined0 = " ".join(calls[0])
    assert "research" in joined0 and "scan" in joined0 and "condition-set" in joined0
    assert "variant-grid" in " ".join(calls[1])
    assert "_new_decision_doc.py" in " ".join(calls[2])
    state = (tmp_path / "out" / "rd_loop_state.json").read_text(encoding="utf-8")
    assert "research_scan" in state
    assert "quick_scan_html" in state
    assert "tree_pipeline" in state
    assert "decision_doc" in state


def test_rd_loop_quick_scan_html_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "out"
    scan = out / "quick_scan"
    scan.mkdir(parents=True)
    (scan / "depth_plateau.md").write_text(
        "# feature_plateau · depth >= ?\n\n| threshold | n_hit |\n|---:|---:|\n| 0.5 | 10 |\n",
        encoding="utf-8",
    )
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "html_loop",
                "output_dir": str(out),
                "quick_layer_scans": [
                    {
                        "mode": "condition-set",
                        "features_parquet": "dummy.parquet",
                        "condition": ["H: x>0"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        rc = run_loop(hyp, output_dir=out)

    assert rc == 0
    report = scan / "report.html"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "feature_plateau" in text
    assert "depth_plateau.md" in text


def test_rd_loop_resume_skips_completed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "rd_loop_state.json").write_text(
        '{"completed_steps": ["research_scan", "variant_grid"], "steps": {}}',
        encoding="utf-8",
    )
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "resume_test",
                "output_dir": str(out),
                "decision_doc": {"topic": "resume_test"},
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        rc = run_loop(hyp, output_dir=out, resume=True)

    assert rc == 0
    assert len(calls) == 1
    assert "_new_decision_doc.py" in " ".join(calls[0])


def test_rd_loop_pair_scan_cmd(tmp_path: Path) -> None:
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "pair",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "pair-scan",
                        "features_parquet": "dummy.parquet",
                        "pair_a": "a:<=:0,1",
                        "pair_b": "b:>=:0,1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        run_loop(hyp, output_dir=tmp_path / "out")

    joined = " ".join(calls[0])
    assert "pair-scan" in joined
    assert "--pair-a" in joined


def test_rd_loop_snotio_plateau_passes_subject(tmp_path: Path) -> None:
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "snotio",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "snotio-plateau",
                        "features_parquet": "dummy.parquet",
                        "feature": "pulse_z",
                        "grid": "-1,0,1",
                        "subject": "feature:pulse_z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        run_loop(hyp, output_dir=tmp_path / "out")

    joined = " ".join(calls[0])
    assert "plateau" in joined
    assert "--subject" in joined
    assert "feature:pulse_z" in joined


def test_rd_loop_entry_plateau_runs_batch(tmp_path: Path) -> None:
    import pandas as pd

    pq = tmp_path / "logs.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(pq)
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "entry_plateau",
                "strategy": "srb",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "entry-plateau",
                        "features_parquet": str(pq),
                        "snotio_mode": "entry_rr",
                        "steps": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict] = []

    def fake_batch(*args, **kwargs):  # noqa: ANN002, ANN003
        batch_calls.append({"args": args, "kwargs": kwargs})
        return {"summary_path": str(tmp_path / "out" / "summary.json")}

    with patch(
        "scripts.research.entry_plateau_scan.run_entry_plateau_batch",
        side_effect=fake_batch,
    ):
        rc = run_loop(hyp, output_dir=tmp_path / "out")

    assert rc == 0
    assert len(batch_calls) == 1
    assert batch_calls[0]["kwargs"].get("snotio_mode") == "entry_rr"


def test_rd_loop_gate_plateau_batch(tmp_path: Path) -> None:
    import pandas as pd

    pq = tmp_path / "features.parquet"
    pd.DataFrame({"x": [1.0, 2.0]}).to_parquet(pq)
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "gate_plateau",
                "strategy": "tpc",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "gate-plateau",
                        "features_parquet": str(pq),
                        "skip_locked": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict] = []

    def fake_batch(*args, **kwargs):  # noqa: ANN002, ANN003
        batch_calls.append({"args": args, "kwargs": kwargs})
        return {"kpi": "lift", "strategy": "tpc", "rules": {}}

    with patch(
        "scripts.research.gate_plateau_scan.run_gate_plateau_batch",
        side_effect=fake_batch,
    ):
        rc = run_loop(hyp, output_dir=tmp_path / "out")

    assert rc == 0
    assert len(batch_calls) == 1
    assert batch_calls[0]["args"][1] == "tpc"


def test_rd_loop_gate_plateau_single_feature_cmd(tmp_path: Path) -> None:
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "gate_lift",
                "strategy": "tpc",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "gate-plateau",
                        "features_parquet": "dummy.parquet",
                        "feature": "tpc_semantic_chop",
                        "operator": "gt",
                        "grid": "0.2,0.4,0.6",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        run_loop(hyp, output_dir=tmp_path / "out")

    joined = " ".join(calls[0])
    assert "plateau" in joined
    assert "--kpi" in joined and "lift" in joined
    assert "--operator" in joined and "gt" in joined


def test_rd_loop_locked_prefilter_tune(tmp_path: Path) -> None:
    import pandas as pd

    pq = tmp_path / "features.parquet"
    pd.DataFrame({"success_no_rr_extreme": [0, 1, 0, 1]}).to_parquet(pq)
    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "pf_tune",
                "strategy": "tpc",
                "output_dir": str(tmp_path / "out"),
                "quick_layer_scans": [
                    {
                        "mode": "locked-prefilter-tune",
                        "features_parquet": str(pq),
                        "prefilter_path": str(tmp_path / "missing_prefilter.yaml"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rc = run_loop(hyp, output_dir=tmp_path / "out")
    assert rc == 3


def test_rd_loop_tree_pipeline_steps(tmp_path: Path) -> None:
    import pandas as pd

    pq = tmp_path / "features.parquet"
    pred = tmp_path / "preds.parquet"
    pd.DataFrame({"label": [0.1, 0.2], "close": [1.0, 1.1]}).to_parquet(pq)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-10-01", periods=2, freq="120min"),
            "_symbol": ["BTCUSDT", "BTCUSDT"],
            "pred": [0.2, 0.3],
            "close": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
        }
    ).to_parquet(pred)

    hyp = tmp_path / "hyp.yaml"
    hyp.write_text(
        yaml.safe_dump(
            {
                "topic": "tree_test",
                "strategy": "fast_scalp",
                "output_dir": str(tmp_path / "out"),
                "tree_steps": [
                    {
                        "mode": "ic-prune",
                        "features_parquet": str(pq),
                        "out": "ic_out",
                        "write_features_yaml": False,
                    },
                    {
                        "mode": "filter-predictions",
                        "predictions": str(pred),
                        "symbols": "BTCUSDT",
                        "out": "filtered/preds.parquet",
                    },
                    {
                        "mode": "tau-scan",
                        "config": "config/strategies/tree_strategies/fast_scalp",
                        "predictions": "filtered/preds.parquet",
                        "out": "tau_out",
                        "segment_label": "test_holdout",
                        "filter_split": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    with patch("scripts.rd_loop.subprocess.run", side_effect=fake_run):
        with patch(
            "scripts.research.tree_holdout_tau_rr_scan.run_tau_scan"
        ) as mock_tau:
            mock_tau.return_value = {"json": tmp_path / "tau.json"}
            rc = run_loop(hyp, output_dir=tmp_path / "out")

    assert rc == 0
    assert mock_tau.call_count == 1
    ic_cmds = [c for c in calls if "ic-prune" in c]
    assert len(ic_cmds) == 1
    assert "research" in ic_cmds[0]
    assert "ic-prune" in ic_cmds[0]
    filtered = tmp_path / "out" / "filtered" / "preds.parquet"
    assert filtered.is_file()
    state = json.loads((tmp_path / "out" / "rd_loop_state.json").read_text())
    assert state.get("tree_pipeline_completed") == [0, 1, 2]
