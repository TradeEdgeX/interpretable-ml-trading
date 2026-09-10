"""copy_strategy_package must not clone data/results/docker from strategy dirs."""

from __future__ import annotations

from pathlib import Path

from src.config.strategy_layout import copy_strategy_package


def test_copy_strategy_package_skips_heavy_dirs_and_artifacts(tmp_path: Path) -> None:
    src = tmp_path / "src_pkg"
    src.mkdir()
    (src / "meta.yaml").write_text("strategy: {}\n", encoding="utf-8")
    (src / "features.yaml").write_text("name: demo\n", encoding="utf-8")
    (src / "archetypes").mkdir()
    (src / "archetypes" / "gate.yaml").write_text("hard_gates: []\n", encoding="utf-8")
    (src / "research").mkdir()
    (src / "research" / "calibrate_roll.default.yaml").write_text(
        "strategy_type: bpc\n", encoding="utf-8"
    )
    (src / "data").mkdir()
    (src / "data" / "warmup.zip").write_bytes(b"x" * 1024)
    (src / "results").mkdir()
    (src / "results" / "nested.json").write_text("{}", encoding="utf-8")
    (src / "docker").mkdir()
    (src / "docker" / "torch.whl").write_bytes(b"y" * 1024)

    dst = tmp_path / "dst_pkg"
    copy_strategy_package(src, dst)

    assert (dst / "meta.yaml").is_file()
    assert (dst / "features.yaml").is_file()
    assert (dst / "archetypes" / "gate.yaml").is_file()
    assert not (dst / "data").exists()
    assert not (dst / "results").exists()
    assert not (dst / "docker").exists()
