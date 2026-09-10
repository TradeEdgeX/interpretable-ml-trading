from __future__ import annotations

from pathlib import Path

from src.research.replay import replay_topic, scaffold_replay
from src.research.review import review_experiment


def test_chop_grid_is_not_replayable(tmp_path: Path) -> None:
    exp = tmp_path / "config" / "experiments" / "20260101_chop_x"
    exp.mkdir(parents=True)
    (exp / "DECISION.md").write_text(
        "---\nstrategy: chop_grid\nharness: chop_grid_backtest\nkill_switch: false\n---\n",
        encoding="utf-8",
    )
    row = review_experiment(exp, repo_root=tmp_path)
    assert row.action == "do_not_replay"
    assert row.replayable is False


def test_trusted_is_skipped(tmp_path: Path) -> None:
    exp = tmp_path / "config" / "experiments" / "20260101_done"
    exp.mkdir(parents=True)
    (exp / "DECISION.md").write_text(
        "---\nstrategy: srb\nharness: event_backtest\n"
        "segments: [bear_2022, bull_2023_2024, recent_range_to_bear]\n"
        "kill_switch: false\nverdict: reject\n---\n",
        encoding="utf-8",
    )
    row = review_experiment(exp, repo_root=tmp_path)
    assert row.action == "skip"
    assert "trusted_closed" in row.flags


def test_ma_cross_legacy_can_open_new_court(tmp_path: Path) -> None:
    parent = tmp_path / "config" / "experiments" / "20260101_ma_cross_old"
    parent.mkdir(parents=True)
    (parent / "DECISION.md").write_text(
        "---\nstrategy: ma_cross\nharness: event_backtest\nkill_switch: false\n---\n# old\n",
        encoding="utf-8",
    )
    row = review_experiment(parent, repo_root=tmp_path)
    assert row.replayable is True
    dest = scaffold_replay("20260101_ma_cross_old", repo_root=tmp_path, today="20260823")
    assert dest.name == "20260823_replay_ma_cross_old"
    text = (dest / "DECISION.md").read_text(encoding="utf-8")
    assert "replays: [20260101_ma_cross_old]" in text
    assert "verdict:" in text
    assert "promote" not in text.split("verdict:")[1][:20]
    assert (
        (parent / "DECISION.md")
        .read_text(encoding="utf-8")
        .startswith("---\nstrategy: ma_cross")
    )


def test_replay_topic_strips_date() -> None:
    assert replay_topic("20260602_trend_scalp_segment_validate", today="20260823") == (
        "20260823_replay_trend_scalp_segment_validate"
    )


def test_scaffold_without_grid_does_not_invent_run_grid(tmp_path: Path) -> None:
    parent = tmp_path / "config" / "experiments" / "20260101_srb_old"
    parent.mkdir(parents=True)
    (parent / "DECISION.md").write_text(
        "---\nstrategy: srb\nharness: event_backtest\nkill_switch: false\n---\n# old\n",
        encoding="utf-8",
    )
    dest = scaffold_replay("20260101_srb_old", repo_root=tmp_path, today="20260823")
    assert not (dest / "run_grid.py").exists()
    text = (dest / "DECISION.md").read_text(encoding="utf-8")
    assert "Do not add run_grid.py" in text
    assert "mlbot research run 20260823_replay_srb_old" in text
