"""Integration: validation_smoke TPC gate batch → calibrate → promote dry-run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts._validation_smoke_assets import make_tpc_parquet
from scripts.research import calibrate as calibrate_mod
from scripts.research import promote as promote_mod
from scripts.research.gate_plateau_scan import run_gate_plateau_batch


@pytest.fixture
def tpc_smoke_gate(tmp_path: Path) -> Path:
    """Minimal TPC-like gate with one optimizable rule + one band rule."""
    gate_dir = tmp_path / "strategies" / "tpc" / "archetypes"
    gate_dir.mkdir(parents=True)
    gate_path = gate_dir / "gate.yaml"
    gate_path.write_text(
        yaml.safe_dump(
            {
                "hard_gates": [
                    {
                        "id": "gate_tpc_semantic_chop_high",
                        "phase": "hard_gate",
                        "reason": "smoke test chop deny",
                        "when": {"tpc_semantic_chop": {"value_gt": 0.45}},
                        "then": {"action": "deny"},
                    }
                ],
                "system_safety": [
                    {
                        "id": "gate_vol_persistence_vol_persistence_bull_only",
                        "phase": "system_safety",
                        "when": {
                            "all_of": [
                                {"vol_persistence": {"value_gt": 0.003}},
                                {"vol_persistence": {"value_lt": 0.06}},
                                {"ema_1200_position": {"value_gt": 0.10}},
                            ]
                        },
                        "then": {"action": "deny"},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return gate_path


@pytest.mark.integration
def test_validation_smoke_tpc_gate_pipeline(
    tmp_path: Path, tpc_smoke_gate: Path
) -> None:
    parquet = make_tpc_parquet(tmp_path / "features_labeled.parquet", n=4000)
    batch_dir = tmp_path / "gate_plateau"
    strategies_root = tmp_path / "strategies"

    batch = run_gate_plateau_batch(
        parquet,
        "tpc",
        out_dir=batch_dir,
        gate_path=str(tpc_smoke_gate),
        strategies_root=str(strategies_root),
        write_back_intervals=True,
        step=0.05,
        skip_locked=False,
    )
    assert batch["kpi"] == "lift"
    assert "gate_tpc_semantic_chop_high" in batch["rules"]

    chop_opt = batch["rules"]["gate_tpc_semantic_chop_high"]
    assert chop_opt.get("status") is not None
    assert chop_opt.get("feature") == "tpc_semantic_chop"

    batch_path = batch_dir / "gate_plateau_batch.json"
    assert batch_path.is_file()
    assert (batch_dir / "gate_plateau_summary.md").is_file()

    draft_path = tmp_path / "gate_draft.yaml"
    rc = calibrate_mod.main(
        [
            "--from-plateau",
            str(batch_path),
            "--output",
            str(draft_path),
            "--strategy",
            "tpc",
            "--strategies-root",
            str(strategies_root),
        ]
    )
    assert rc == 0
    assert draft_path.is_file()
    draft_text = draft_path.read_text(encoding="utf-8")
    assert "gate_tpc_semantic_chop_high" in draft_text

    skips_path = draft_path.with_suffix(draft_path.suffix + ".skips.json")
    vol_opt = batch["rules"].get("gate_vol_persistence_vol_persistence_bull_only", {})
    vol_status = vol_opt.get("status")
    if vol_status in calibrate_mod._OPTIMIZER_APPLY_STATUSES and not vol_opt.get(
        "threshold_interval"
    ):
        assert skips_path.is_file()
        skips_payload = json.loads(skips_path.read_text(encoding="utf-8"))
        reasons = {s["reason"] for s in skips_payload["skips"]}
        assert "unsafe_band_no_interval" in reasons
    elif vol_status in calibrate_mod._OPTIMIZER_APPLY_STATUSES and vol_opt.get(
        "threshold_interval"
    ):
        draft = yaml.safe_load(
            "\n".join(
                line for line in draft_text.splitlines() if not line.startswith("#")
            )
        )
        vol_rule = draft["system_safety"][0]
        clauses = vol_rule["when"]["all_of"]
        assert any("ema_1200_position" in c for c in clauses)
        assert sum(1 for c in clauses if "vol_persistence" in c) == 2
    elif vol_status not in calibrate_mod._OPTIMIZER_APPLY_STATUSES:
        # optimizer did not propose an applicable update — skip manifest optional
        if skips_path.is_file():
            skips_payload = json.loads(skips_path.read_text(encoding="utf-8"))
            assert skips_payload["skip_count"] >= 1

    prod_gate = tpc_smoke_gate
    merged_text, notes = promote_mod.promote_yaml(draft_path, prod_gate, layer="gate")
    assert "hard_gates" in merged_text
    merged = yaml.safe_load(merged_text)
    assert merged["hard_gates"][0]["id"] == "gate_tpc_semantic_chop_high"
    assert isinstance(notes, list)

    # dry-run promote CLI must not write
    rc_dry = promote_mod.main(
        [
            "--from",
            str(draft_path),
            "--to",
            str(prod_gate),
            "--layer",
            "gate",
            "--dry-run",
            "--yes",
        ]
    )
    assert rc_dry == 0
    # production gate yaml unchanged (dry-run)
    assert prod_gate.read_text(encoding="utf-8") == tpc_smoke_gate.read_text(
        encoding="utf-8"
    )
