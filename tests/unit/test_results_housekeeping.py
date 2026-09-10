from __future__ import annotations

from collections import namedtuple
from pathlib import Path

from scripts import results_housekeeping as rh


def test_is_timestamp_dir_name() -> None:
    assert rh.is_timestamp_dir_name("20260512_101450")
    assert rh.is_timestamp_dir_name("20260512_101450_s42")
    assert not rh.is_timestamp_dir_name("train_final_20260512")
    assert not rh.is_timestamp_dir_name("bad_name")


def test_prune_candidates_keeps_latest_timestamp_dirs(tmp_path: Path) -> None:
    root = tmp_path / "results" / "bpc" / "validate_static.constrained" / "bpc"
    (root / "20260510_010101").mkdir(parents=True)
    (root / "20260511_010101").mkdir(parents=True)
    (root / "20260512_010101").mkdir(parents=True)
    got = rh.prune_candidates(root, keep=1)
    assert [p.name for p in got] == ["20260510_010101", "20260511_010101"]


def test_prune_candidates_train_final(tmp_path: Path) -> None:
    root = tmp_path / "results" / "train_final"
    (root / "train_final_20260510_a").mkdir(parents=True)
    (root / "train_final_20260511_a").mkdir(parents=True)
    (root / "train_final_20260512_a").mkdir(parents=True)
    got = rh.prune_candidates(root, keep=2)
    assert [p.name for p in got] == ["train_final_20260510_a"]


def test_preflight_fails_on_low_free_space(monkeypatch, tmp_path: Path) -> None:
    Usage = namedtuple("Usage", "total used free")

    def _fake_disk_usage(_: Path):
        # total=100, used=92, free=8 GiB
        g = 1024**3
        return Usage(100 * g, 92 * g, 8 * g)

    monkeypatch.setattr(rh.shutil, "disk_usage", _fake_disk_usage)
    h = rh.disk_health(tmp_path)
    assert h.used_pct > 90
    assert h.free_gb < 10
