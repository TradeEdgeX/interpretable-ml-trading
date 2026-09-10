from pathlib import Path

from src.time_series_model.live.metrics_exporter import (
    METRICS,
    _disk_monitor_volumes,
    _directory_size_bytes,
)


def test_disk_monitor_volumes_from_env(monkeypatch, tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setenv("MLBOT_DISK_MONITOR_VOLUMES", f"logs:{logs},root:/")
    volumes = dict(_disk_monitor_volumes())
    assert volumes["logs"] == logs
    assert volumes["root"] == Path("/")


def test_directory_size_bytes_counts_files(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 1024)
    (tmp_path / "b.bin").write_bytes(b"y" * 512)
    size = _directory_size_bytes(tmp_path)
    assert size is not None
    assert size >= 1536


def test_update_disk_health_sets_gauges(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "f").write_bytes(b"z" * 100)
    monkeypatch.setenv("MLBOT_DISK_MONITOR_VOLUMES", f"root:{tmp_path},ticks:{data}")
    monkeypatch.setenv("MLBOT_DISK_DIR_SIZE_INTERVAL_SECONDS", "0")
    METRICS.update_disk_health()
    # Prometheus client registers series; values should be readable via collect if installed.
    try:
        from prometheus_client import REGISTRY

        names = {
            s.name
            for m in REGISTRY.collect()
            for s in m.samples
            if s.name.startswith("mlbot_disk_") or s.name.startswith("mlbot_dir_size")
        }
        assert "mlbot_disk_used_percent" in names
        assert "mlbot_dir_size_gb" in names
    except ImportError:
        pass
