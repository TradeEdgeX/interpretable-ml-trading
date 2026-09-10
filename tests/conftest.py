"""
Shared pytest configuration and fixtures.

This conftest.py automatically handles project root setup for all tests.
- Sets PROJECT_ROOT for path calculations
- Adds project root to sys.path (so imports work even if project is not installed)

Note: If you install the project via `pip install -e .`, sys.path setup is optional.
However, we still set it up here to ensure tests work in all environments.
"""

import os

# Headless CI/WSL: avoid matplotlib Qt backend abort before any test imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")
# Cockpit ETF panel: never hit Farside from unit/CMS route tests.
os.environ.setdefault("MLBOT_COCKPIT_ETF_FETCH", "0")

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path if not already there
# This ensures tests work even if project is not installed via pip
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_collection_modifyitems(config, items):
    """
    Auto-mark known heavy tests as `slow` so they can be skipped in fast dev loops.

    Default (see pytest.ini): ``pytest -m "not slow and not integration"``.
    Full suite: ``pytest -m ""`` or ``pytest -o addopts=``.
    """
    for item in items:
        path = str(item.fspath)
        # Heaviest suite: advanced features (DTW/EVT/GARCH/etc.)
        if "tests/features/test_advanced_features.py" in path:
            item.add_marker(pytest.mark.slow)
        if "test_rsi_inf_full_pipeline.py" in path:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.integration)
        for heavy in (
            "test_vpin_adaptive_bucket.py",
            "test_vpin_ofi_chain_diagnostic.py",
            "test_feature_diagnostic.py",
        ):
            if heavy in path:
                item.add_marker(pytest.mark.slow)
                item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def _isolate_live_equity_and_position_tracker_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never load ``live/*/data`` equity anchors into unit tests.

    ``OrderFlowListener`` defaults to ``live/highcap/data/position_tracker``;
    a local `_b_equity_anchors.json` would silently poison G0 / inject equity
    (fund-safety false positives). Each test gets a fresh tmp state dir.
    """
    state_dir = tmp_path / "position_tracker"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLBOT_POSITION_TRACKER_STATE_DIR", str(state_dir))
    monkeypatch.setenv(
        "MLBOT_B_EQUITY_STATE_PATH", str(state_dir / "_b_equity_anchors.json")
    )
    monkeypatch.setenv("MLBOT_LIVE_BASE", str(tmp_path / "live_base"))


@pytest.fixture(autouse=True)
def _disable_exposure_consensus_by_default(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Most tests assume immediate entry/heal; opt in via @pytest.mark.exposure_consensus."""
    if request.node.get_closest_marker("exposure_consensus"):
        return
    monkeypatch.setenv("MLBOT_EXPOSURE_CONSENSUS_ENABLED", "0")


@pytest.fixture(autouse=True)
def _reset_shared_exposure_trackers() -> None:
    from order_management.exchange_exposure_consensus import (
        reset_shared_exposure_trackers,
    )

    reset_shared_exposure_trackers()
    yield
    reset_shared_exposure_trackers()


@pytest.fixture
def sample_data():
    """创建样本数据用于测试"""
    np.random.seed(42)
    n_samples = 500
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4h")

    # 生成价格数据
    price_base = 50000
    returns = np.random.randn(n_samples) * 0.01
    prices = price_base * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_samples) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples),
            "cvd": np.random.randn(n_samples).cumsum() * 1000,
            "taker_buy_ratio": np.random.uniform(0.3, 0.7, n_samples),
        },
        index=dates,
    )

    # 计算基础指标
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["rsi"] = 50 + np.random.randn(n_samples) * 10  # 简化的 RSI

    return df


@pytest.fixture
def feature_loader():
    """创建特征加载器 fixture"""
    from src.features.loader.strategy_feature_loader import StrategyFeatureLoader

    return StrategyFeatureLoader(
        feature_deps_path=str(PROJECT_ROOT / "config" / "feature_dependencies.yaml"),
        cache_dir=str(PROJECT_ROOT / "cache" / "features"),
        use_disk_cache=False,  # 测试时禁用缓存
        use_memory_cache=True,
        max_workers=2,
    )


@pytest.fixture
def strategy_config():
    """加载策略配置 fixture"""
    from src.time_series_model.strategy_config import StrategyConfigLoader

    # sr_reversal 已被删除；默认使用 long-only 策略配置
    strategy_dir = PROJECT_ROOT / "config" / "strategies" / "sr_reversal_long"
    config_loader = StrategyConfigLoader(strategy_dir)
    return config_loader.load()
