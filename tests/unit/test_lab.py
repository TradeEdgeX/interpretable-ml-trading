"""Court Lab: experiment cards + browse. No auxiliary / CMS routes."""

from __future__ import annotations

from pathlib import Path

from src.lab import browse as browse_mod
from src.lab import experiments as rd
from src.lab.app import api_route_prefixes, app, page_routes

REPO = Path(__file__).resolve().parents[2]


def _route_paths() -> set[str]:
    out: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if path:
            out.add(str(path))
    return out


def test_lab_pages_are_court_only() -> None:
    assert page_routes() == frozenset({"/", "/rd", "/rd/qa", "/browse"})
    paths = _route_paths()
    assert "/rd" in paths
    assert "/rd/qa" in paths
    assert "/browse" in paths
    for banned in (
        "/ashare",
        "/hk",
        "/macro",
        "/crypto-xsection",
        "/trade-map",
        "/orders",
        "/account",
        "/monitoring",
        "/grafana",
    ):
        assert banned not in paths
        assert not any(p.startswith(banned + "/") for p in paths)


def test_lab_api_prefixes_are_court_only() -> None:
    assert api_route_prefixes() == frozenset(
        {"/api/rd", "/api/browse", "/results", "/static"}
    )


def test_list_experiments_skips_underscore_dirs() -> None:
    rd.clear_experiments_cache()
    rows = rd.list_experiments(repo_root=REPO, experiments_root=REPO / "config" / "experiments")
    ids = {r["id"] for r in rows}
    assert "20260910_ma50_ma200_cross" in ids
    assert not any(i.startswith("_") for i in ids)


def test_list_experiments_hides_undeclared_verdict() -> None:
    rd.clear_experiments_cache()
    rows = rd.list_experiments(repo_root=REPO, experiments_root=REPO / "config" / "experiments")
    open_rows = [r for r in rows if r.get("record_class") == "open"]
    assert open_rows
    assert all(not r.get("display_verdict") for r in open_rows)


def test_get_experiment_detail() -> None:
    exp_id = "20260910_ma50_ma200_cross"
    row = rd.get_experiment(
        exp_id, repo_root=REPO, experiments_root=REPO / "config" / "experiments"
    )
    assert row is not None
    assert row["id"] == exp_id
    assert row["strategy"] == "ma_cross"
    assert row["has_decision"] is True
    assert "DECISION" in (row.get("decision_text") or "")
    assert row["court"]["scorecard"]["phase3"]


def test_raw_file_rejects_path_escape() -> None:
    assert (
        rd.get_experiment_raw_file(
            "20260910_ma50_ma200_cross",
            "../README.md",
            repo_root=REPO,
            experiments_root=REPO / "config" / "experiments",
        )
        is None
    )


def test_browse_blocks_parent_path(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "ok.txt").write_text("x", encoding="utf-8")
    assert browse_mod.safe_browse_target(root, "..") is None
    assert browse_mod.safe_browse_target(root, "../ok.txt") is None
    assert browse_mod.list_browse(root, "..") is None
    listed = browse_mod.list_browse(root, "")
    assert listed is not None
    assert listed["missing_root"] is False
    names = {e["name"] for e in listed["entries"]}
    assert "ok.txt" in names


def test_browse_missing_root(tmp_path: Path) -> None:
    listed = browse_mod.list_browse(tmp_path / "nope", "")
    assert listed is not None
    assert listed["missing_root"] is True
    assert listed["entries"] == []
