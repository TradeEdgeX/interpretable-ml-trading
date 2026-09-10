"""Unit tests for config-driven quick_scan HTML report ordering."""

from __future__ import annotations

from pathlib import Path

from scripts.research.quick_scan_html import (
    ReportConfig,
    _order_artifact_paths,
    build_report,
    manifest_sections_from_scans,
    resolve_report_config,
    section_order_from_scans,
    subset_label_from_filters,
)


def test_section_order_from_scans() -> None:
    scans = [
        {"out": "quick_scan/a.md"},
        {"out": "results/x/quick_scan/b.json"},
        {"out": "quick_scan/a.md"},
    ]
    assert section_order_from_scans(scans) == ["a.md", "b.json"]


def test_order_artifact_paths_respects_config_then_alpha() -> None:
    paths = [Path("z.md"), Path("a.md"), Path("m.json")]
    ordered = _order_artifact_paths(paths, ["m.json", "a.md"])
    assert [p.name for p in ordered] == ["m.json", "a.md", "z.md"]


def test_resolve_report_config_scan_out(tmp_path: Path) -> None:
    hyp = {
        "topic": "my_exp",
        "quick_layer_scans": [{"out": "quick_scan/first.md"}, {"out": "second.json"}],
        "quick_scan_html": {"title": "Custom Title", "section_order": "scan_out"},
    }
    cfg = resolve_report_config(hypothesis=hyp, html_block=hyp["quick_scan_html"])
    assert cfg.title == "Custom Title"
    assert cfg.section_order == ["first.md", "second.json"]


def test_subset_label_bear() -> None:
    assert "bear" in subset_label_from_filters(["ema_1200_position<=-0.10"])


def test_manifest_sections_from_scans() -> None:
    secs = manifest_sections_from_scans(
        [
            {
                "out": "quick_scan/vol_short.md",
                "mode": "feature-plateau",
                "filter": ["ema_1200_position<=-0.10"],
            }
        ]
    )
    assert secs[0]["file"] == "vol_short.md"
    assert "bear" in secs[0]["subset"]


def test_chart_condition_set_table() -> None:
    md = """# condition_set scan

- base mask n = 100, base_success = 50.000%

| condition | n | succ_in | succ_out | Δpp vs base | |z| |
|---|---:|---:|---:|---:|---:|
| deep_absorb | 80 | 60.000% | 48.000% | +10.00 | 2.10 |
| prod_entry | 5 | 40.000% | 51.000% | -10.00 | 0.50 |
"""
    from scripts.research.quick_scan_html import _md_to_html

    html = _md_to_html(md)
    assert "chart-wrap" in html
    assert "deep_absorb" in html


def test_build_report_includes_extra_files(tmp_path: Path) -> None:
    scan = tmp_path / "quick_scan"
    scan.mkdir()
    (scan / "b.md").write_text(
        "# feature_plateau · x >= ?\n\n- base n = 10, base_success = 50.000%\n\n"
        "| threshold | n_hit | succ_hit | succ_other | |z| |\n"
        "|---:|---:|---:|---:|---:|\n| 0.5 | 5 | 60.000% | 50.000% | 0.5 |\n",
        encoding="utf-8",
    )
    (scan / "extra.md").write_text(
        "# extra\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8"
    )
    html = build_report(scan, title="T", section_order=["b.md"])
    assert "feature_plateau" in html
    assert "extra.md" in html
    assert html.index("b.md") < html.index("extra.md")
