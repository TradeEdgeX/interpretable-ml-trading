"""R&D experiment API on local rolling_dashboard (/rd)."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts.rolling_dashboard_server import build_request_handler

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _serve(handler_cls, port: int = 0):
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server


def test_rd_list_hides_inferred_verdict_on_legacy() -> None:
    from scripts.rolling_dashboard import rd_experiments

    rd_experiments.clear_experiments_cache()
    rows = rd_experiments.list_experiments()
    trusted = [r for r in rows if r.get("record_class") == "trusted"]
    legacy = [r for r in rows if r.get("record_class") == "legacy"]
    assert trusted
    assert all(r.get("display_verdict") for r in trusted)
    assert legacy
    assert all(r.get("display_verdict") in (None, "") for r in legacy[:20])
    filled = [r for r in rows if r.get("scorecard_fill")]
    assert filled
    assert any(
        r.get("scorecard_fill", {}).get("label") in ("空", "齐")
        or str(r.get("scorecard_fill", {}).get("label", "")).startswith("P3")
        for r in filled
    )


def test_rd_experiments_list_api(tmp_path: Path) -> None:
    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/rd/experiments")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        assert isinstance(body["data"], list)
        assert body["meta"]["count"] == len(body["data"])
        assert body["meta"]["count"] >= 10
    finally:
        server.shutdown()


def test_rd_experiment_detail_known(tmp_path: Path) -> None:
    exp_id = "20260531_tpc_gate_validate"
    if not (PROJECT_ROOT / "config" / "experiments" / exp_id).is_dir():
        pytest.skip("missing fixture experiment")

    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/rd/experiment/{exp_id}"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        data = body["data"]
        assert data["id"] == exp_id
        assert data["has_decision"] is True
        assert "decision_text" in data
        card = (data.get("court") or {}).get("scorecard") or {}
        assert set(card.get("phase1") or {}) >= {"ic", "icir", "auc", "lift_pp"}
        assert set(card.get("phase3") or {}) >= {
            "cagr",
            "calmar",
            "win_rate",
            "maxdd",
            "sharpe",
        }
    finally:
        server.shutdown()


def test_rd_page_served(tmp_path: Path) -> None:
    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/rd")
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "R&D 实验" in html
        assert "/api/rd/experiments" not in html
        assert "rd-page.js" in html
    finally:
        server.shutdown()


def test_list_experiments_newest_dated_first(tmp_path: Path) -> None:
    from scripts.rolling_dashboard import rd_experiments

    root = tmp_path / "config" / "experiments"
    old = root / "20260527_old_demo"
    new = root / "20260823_new_demo"
    undated = root / "chop_no_date_demo"
    old.mkdir(parents=True)
    new.mkdir()
    undated.mkdir()
    for p in (old, new, undated):
        (p / "README.md").write_text("## 假设\ncard\n", encoding="utf-8")

    ids = [
        r["id"]
        for r in rd_experiments.list_experiments(
            experiments_root=root, repo_root=tmp_path
        )
    ]
    assert ids[0] == "20260823_new_demo"
    assert ids[1] == "20260527_old_demo"
    assert ids[-1] == "chop_no_date_demo"


def test_query_matches_mangled_date_prefix() -> None:
    from scripts.rolling_dashboard.rd_experiments import _query_matches

    hay = "20260823_replay_bpc_entry_v2 replay_bpc_entry_v2 bpc"
    assert _query_matches("20260823_replay_bpc_entry_v2", hay)
    assert _query_matches("replay_bpc", hay)
    assert _query_matches("123_replay_bpc_entry_v2", hay)
    assert not _query_matches("chop_grid_semantic_proxy", hay)


def test_list_experiments_picks_up_decision_edit_without_manual_clear(
    tmp_path: Path,
) -> None:
    from scripts.rolling_dashboard import rd_experiments

    root = tmp_path / "config" / "experiments"
    exp = root / "20260823_edit_demo"
    exp.mkdir(parents=True)
    decision = exp / "DECISION.md"
    decision.write_text(
        "---\nstrategy: ma_cross\nverdict:\n---\n# old title\n",
        encoding="utf-8",
    )
    rows1 = rd_experiments.list_experiments(experiments_root=root, repo_root=tmp_path)
    assert rows1[0]["decision_title"] == "old title"
    assert rows1[0]["verdict"] in (None, "")

    decision.write_text(
        "---\nstrategy: ma_cross\nverdict: reject\n---\n# new title\n",
        encoding="utf-8",
    )
    os.utime(decision, (decision.stat().st_mtime + 5, decision.stat().st_mtime + 5))

    rows2 = rd_experiments.list_experiments(experiments_root=root, repo_root=tmp_path)
    assert rows2[0]["decision_title"] == "new title"
    assert rows2[0]["verdict"] == "reject"


def test_list_experiments_drops_deleted_dir_without_manual_clear(
    tmp_path: Path,
) -> None:
    from scripts.rolling_dashboard import rd_experiments

    root = tmp_path / "config" / "experiments"
    keep = root / "20260101_keep_demo"
    gone = root / "20260102_gone_demo"
    keep.mkdir(parents=True)
    gone.mkdir()
    (keep / "README.md").write_text("## 假设\nkeep card\n", encoding="utf-8")
    (gone / "README.md").write_text(
        "## 假设\ngone card should vanish\n", encoding="utf-8"
    )

    ids1 = {
        r["id"]
        for r in rd_experiments.list_experiments(
            experiments_root=root, repo_root=tmp_path
        )
    }
    assert ids1 == {"20260101_keep_demo", "20260102_gone_demo"}

    for child in gone.iterdir():
        child.unlink()
    gone.rmdir()

    ids2 = {
        r["id"]
        for r in rd_experiments.list_experiments(
            experiments_root=root, repo_root=tmp_path
        )
    }
    assert ids2 == {"20260101_keep_demo"}


def test_rd_refresh_post(tmp_path: Path) -> None:
    handler = build_request_handler(tmp_path)
    server = _serve(handler, 0)
    port = server.server_address[1]
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/rd/refresh",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        assert body["data"]["refreshed"] is True
    finally:
        server.shutdown()
