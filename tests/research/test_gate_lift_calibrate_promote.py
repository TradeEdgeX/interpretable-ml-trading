"""Tests for gate lift plateau and calibrate/promote upgrades."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.research import calibrate as calibrate_mod
from scripts.research import promote as promote_mod
from scripts.research.gate_lift_scan import gate_lift_plateau_payload
from scripts.pre_deploy_contract_checks import _check_cross_regime_evidence
from scripts.research.drift_suggestions import build_rd_loop_snippet


def _synthetic_gate_df(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    feat = rng.uniform(0, 1, size=n)
    # High chop → bad; deny high chop (operator '>' → deny 'lt', pass when feat >= τ)
    is_good = (feat < 0.45).astype(int)
    return pd.DataFrame(
        {
            "tpc_semantic_chop": feat,
            "is_good": is_good,
            "forward_rr": np.where(is_good, 1.0, -1.5),
        }
    )


def test_gate_lift_plateau_payload_finds_plateau() -> None:
    df = _synthetic_gate_df()
    mask = pd.Series(True, index=df.index)
    payload = gate_lift_plateau_payload(
        df,
        "tpc_semantic_chop",
        "gt",
        base_mask=mask,
        label_col="is_good",
        grid=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    )
    assert payload["deny_operator"] == "gt"
    assert payload["kpi"] == "lift"
    assert len(payload.get("rows", [])) >= 3
    assert payload.get("recommended") is not None or payload.get("is_plateau") is False


def test_calibrate_single_lift_draft(tmp_path: Path) -> None:
    src = tmp_path / "lift.json"
    src.write_text(
        json.dumps(
            {
                "kpi": "lift",
                "feature": "tpc_semantic_chop",
                "deny_operator": "gt",
                "recommended": 0.4,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "draft.yaml"
    rc = calibrate_mod.main(["--from-plateau", str(src), "--output", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "hard_gates" in text
    assert "tpc_semantic_chop" in text


def test_promote_preserves_locked_rule(tmp_path: Path) -> None:
    prod = tmp_path / "gate.yaml"
    prod.write_text(
        yaml.safe_dump(
            {
                "hard_gates": [
                    {
                        "id": "gate_locked",
                        "locked": True,
                        "when": {"feat_a": {"value_gt": 0.99}},
                        "then": {"action": "deny"},
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    draft = tmp_path / "draft.yaml"
    draft.write_text(
        "# DRAFT\n"
        + yaml.safe_dump(
            {
                "hard_gates": [
                    {
                        "id": "gate_locked",
                        "when": {"feat_a": {"value_gt": 0.1}},
                        "then": {"action": "deny"},
                    },
                    {
                        "id": "gate_new",
                        "when": {"feat_b": {"value_gt": 0.2}},
                        "then": {"action": "deny"},
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    merged_text, notes = promote_mod.promote_yaml(draft, prod, layer="gate")
    merged = yaml.safe_load(merged_text)
    locked = merged["hard_gates"][0]
    assert locked["when"]["feat_a"]["value_gt"] == 0.99
    assert any("preserved locked" in n for n in notes)
    assert len(merged["hard_gates"]) == 2


def test_calibrate_main_writes_skip_manifest(tmp_path: Path) -> None:
    arch = tmp_path / "strategies" / "tpc" / "archetypes"
    arch.mkdir(parents=True)
    arch.joinpath("gate.yaml").write_text(
        yaml.safe_dump(
            {
                "hard_gates": [
                    {
                        "id": "gate_anyof",
                        "when": {
                            "any_of": [
                                {"vol_persistence": {"value_gt": 0.01}},
                            ]
                        },
                        "then": {"action": "deny"},
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    batch = {
        "kpi": "lift",
        "strategy": "tpc",
        "rules": {
            "gate_anyof": {
                "feature": "vol_persistence",
                "operator": "gt",
                "status": "stable_plateau_found",
                "recommended_threshold": 0.02,
            }
        },
    }
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    draft_path = tmp_path / "draft.yaml"
    rc = calibrate_mod.main(
        [
            "--from-plateau",
            str(batch_path),
            "--output",
            str(draft_path),
            "--strategy",
            "tpc",
            "--strategies-root",
            str(tmp_path / "strategies"),
        ]
    )
    assert rc == 0
    skips_path = draft_path.with_suffix(draft_path.suffix + ".skips.json")
    assert skips_path.is_file()
    payload = json.loads(skips_path.read_text(encoding="utf-8"))
    assert payload["skip_count"] == 1
    assert payload["skips"][0]["reason"] == "unsafe_any_of"


def test_cross_regime_evidence_missing(tmp_path: Path) -> None:
    idx = tmp_path / "EXPERIMENT_INDEX.json"
    idx.write_text(json.dumps([{"name": "recent_only", "tags": ["recent"]}]))
    ok, detail, _ = _check_cross_regime_evidence(
        "tpc",
        {"experiment_index": str(idx), "required_windows": ["recent", "bull"]},
        project_root=tmp_path,
    )
    assert ok is False
    assert "bull" in detail


def test_drift_rd_loop_snippet() -> None:
    snippet = build_rd_loop_snippet(
        strategy="tpc",
        drift_items=[
            {
                "feature": "ema_1200_position",
                "status": "DRIFT",
                "window_p25": 0.05,
                "window_p50": 0.12,
                "window_p75": 0.20,
            },
        ],
        features_parquet="results/recent.parquet",
    )
    assert snippet["strategy"] == "tpc"
    assert len(snippet["research_scans"]) >= 2
    fp = snippet["research_scans"][0]
    assert fp["mode"] == "feature-plateau"
    assert "0.05" in fp["grid"] or "0.12" in fp["grid"]
    cond = snippet["research_scans"][1]["condition"][0]
    assert "0.12" in cond
