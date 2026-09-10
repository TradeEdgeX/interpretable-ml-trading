"""Compare research plateau artifacts (snotio_mode annotation)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.research import compare as compare_mod


def test_compare_plateau_includes_snotio_mode(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.json"
    entry_rr = tmp_path / "entry_rr.json"
    proxy.write_text(
        json.dumps(
            {
                "kpi": "snotio",
                "snotio_mode": "proxy",
                "sim": "proxy",
                "feature": "pulse_z",
                "operator": "<=",
                "recommended": 0.4,
                "is_plateau": True,
            }
        ),
        encoding="utf-8",
    )
    entry_rr.write_text(
        json.dumps(
            {
                "kpi": "snotio",
                "snotio_mode": "entry_rr",
                "sim": "entry_rr",
                "feature": "pulse_z",
                "operator": "<=",
                "recommended": 0.5,
                "is_plateau": True,
            }
        ),
        encoding="utf-8",
    )
    rc = compare_mod.main([str(proxy), str(entry_rr)])
    assert rc == 0


def test_compare_snotio_mode_mismatch_flag(capsys) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = root / "a.json"
        b = root / "b.json"
        a.write_text(
            json.dumps(
                {
                    "snotio_mode": "proxy",
                    "recommended": 0.4,
                    "feature": "x",
                    "operator": "<=",
                }
            ),
            encoding="utf-8",
        )
        b.write_text(
            json.dumps(
                {
                    "snotio_mode": "entry_rr",
                    "recommended": 0.5,
                    "feature": "x",
                    "operator": "<=",
                }
            ),
            encoding="utf-8",
        )
        compare_mod.main([str(a), str(b)])
        out = capsys.readouterr().out
        assert "snotio_mode_mismatch" in out
