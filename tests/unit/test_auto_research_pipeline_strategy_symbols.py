from __future__ import annotations

from pathlib import Path

import scripts.auto_research_pipeline as arp


def test_dual_add_backtest_stage_respects_strategy_symbol_filters(
    tmp_path: Path, monkeypatch
) -> None:
    strat_dir = tmp_path / "dual_add_trend"
    (strat_dir / "research").mkdir(parents=True)
    (strat_dir / "research" / "calibrate_roll.default.yaml").write_text(
        "strategy_type: dual_add_trend\n", encoding="utf-8"
    )
    (strat_dir / "meta.yaml").write_text(
        """
strategy:
  symbol_include: [BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT]
  symbol_exclude: [ADAUSDT]
""".strip(),
        encoding="utf-8",
    )
    captured = {}

    class _FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, cwd=None, capture_output=True, text=True):
        captured["cmd"] = list(cmd)
        return _FakeProc()

    monkeypatch.setattr(arp.subprocess, "run", _fake_run)

    cfg = {
        "strategies": {
            "dual_add_trend": {
                "strategy_type": "dual_add_trend",
                "config": str(strat_dir),
                "timeframe": "120T",
            }
        },
        "dual_add_backtest": {
            "enabled": True,
            "symbols": "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT",
        },
    }
    arp._run_dual_add_backtest_stage(
        cfg=cfg,
        strategies=["dual_add_trend"],
        history_dir=tmp_path / "history",
        timestamp="20260511_000001",
        dry_run=False,
        data_path="data/parquet_data",
        symbols="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT",
        start_date="2022-01-01",
        end_date="2022-12-31",
    )

    cmd = captured["cmd"]
    sym_idx = cmd.index("--symbols")
    assert cmd[sym_idx + 1] == "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
