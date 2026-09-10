#!/usr/bin/env python3

"""Unit tests for offline locked-prefilter parquet plateau suggestion."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.locked_prefilter_parquet_tune import (
    prefilter_rules_pass_mask,
    suggest_locked_prefilter_params_parquet,
)


def test_prefilter_rules_pass_mask_and(tmp_path: Path):
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [0.0, 2.0, 4.0],
        }
    )
    rules = [
        {"feature": "a", "operator": ">=", "value": 1.5},
        {"feature": "b", "operator": "<=", "value": 3.0},
    ]
    m = prefilter_rules_pass_mask(df, rules)
    assert m.tolist() == [False, True, False]


def test_infer_writeback_picks_up_new_locked_atomic():
    from scripts.locked_prefilter_utils import infer_writeback_bindings_from_prefilter

    raw = {
        "rules": [
            {
                "feature": "bpc_new_semantic_knob",
                "operator": ">=",
                "value": 0.1,
                "locked": True,
            },
        ]
    }
    b = infer_writeback_bindings_from_prefilter(raw)
    assert b == [
        {
            "param": "bpc_new_semantic_knob_min",
            "feature": "bpc_new_semantic_knob",
            "operator": ">=",
        }
    ]


def test_suggest_bpc_from_parquet(tmp_path: Path):
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "bpc_recent_breakout_strength": rng.uniform(0, 1, n),
            "bpc_pullback_depth": rng.uniform(0, 1, n),
            "bpc_recovery_strength": rng.uniform(0, 1, n),
            "success_no_rr_extreme": rng.integers(0, 2, n),
        }
    )
    pq = tmp_path / "features_labeled.parquet"
    df.to_parquet(pq, index=False)

    raw_pf = {
        "rules": [
            {
                "feature": "bpc_recent_breakout_strength",
                "operator": ">=",
                "value": 0.45,
                "locked": True,
            },
            {
                "feature": "bpc_pullback_depth",
                "operator": "<=",
                "value": 0.7,
                "locked": True,
            },
            {
                "feature": "bpc_recovery_strength",
                "operator": ">=",
                "value": 0.55,
                "locked": True,
            },
        ]
    }
    prod = tmp_path / "prefilter.yaml"
    prod.write_text(yaml.safe_dump(raw_pf, allow_unicode=True, sort_keys=False))

    norm, meta = suggest_locked_prefilter_params_parquet(
        prod_prefilter_path=prod,
        labeled_parquet_path=pq,
        template="bpc",
        tcfg={
            "plateau_scan_points": 15,
            "plateau_coord_rounds": 2,
        },
        prefilter_gates={"min_pass_rate": 0.01},
    )
    assert meta["archetype_template_guess"] == "bpc"
    assert meta["template"] == "bindings"
    assert set(norm.keys()) == {
        "bpc_recent_breakout_strength_min",
        "bpc_pullback_depth_max",
        "bpc_recovery_strength_min",
    }


def test_suggest_writeback_bindings_generic(tmp_path: Path):
    n = 300
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "tpc_pullback_depth": rng.uniform(0, 1, n),
            "success_no_rr_extreme": rng.integers(0, 2, n),
        }
    )
    pq = tmp_path / "features_labeled.parquet"
    df.to_parquet(pq, index=False)

    raw_pf = {
        "rules": [
            {
                "feature": "tpc_pullback_depth",
                "operator": "<=",
                "value": 0.85,
                "locked": True,
            },
        ]
    }
    prod = tmp_path / "prefilter.yaml"
    prod.write_text(yaml.safe_dump(raw_pf, allow_unicode=True, sort_keys=False))

    bindings = [
        {"param": "tpc_depth_max", "feature": "tpc_pullback_depth", "operator": "<="},
    ]
    norm, meta = suggest_locked_prefilter_params_parquet(
        prod_prefilter_path=prod,
        labeled_parquet_path=pq,
        template="auto",
        tcfg={
            "writeback_bindings": bindings,
            "plateau_scan_points": 12,
            "plateau_coord_rounds": 2,
        },
        prefilter_gates={"min_pass_rate": 0.01},
    )
    assert meta["template"] == "bindings"
    assert "tpc_depth_max" in norm
