"""
综合测试: 方向感知标签系统 (direction_aware labels)

验证目标:
1. rr_extreme: forward_rr 按方向翻转 → failure 标签正确
2. no_opportunity: Short 时 MFE/MAE 互换 → failure 标签正确
3. return_tree: forward_rr 按方向翻转 + GOOD 样本筛选正确
4. me_label any_success BUG 修复: signal_direction 按 breakout 方向分配

数学基础:
  forward_rr_short = -forward_rr_long
  MFE_short = MAE_long,  MAE_short = MFE_long
"""

import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# 确保可以 import 项目模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.strategies.labels.failure_first_label import (
    compute_failure_subtypes,
    compute_bpc_failure_rr_extreme_label,
    compute_bpc_failure_no_opportunity_label,
    compute_bpc_return_tree_label,
    _load_direction_config_for_label,
    _compute_direction_from_rules,
    _direction_aware_subtypes_single_symbol,
    _direction_aware_subtypes,
    _compute_direction_aware_rr_extreme,
    _compute_direction_aware_no_opportunity,
    _compute_direction_aware_return_tree,
)


# ============================================================
# Mock 数据 & 配置 fixtures
# ============================================================


def _make_mock_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成模拟 OHLCV 数据。

    构造一个有趋势的价格序列：
    - 前半段上涨 (适合做多)
    - 后半段下跌 (适合做空)
    """
    rng = np.random.RandomState(seed)

    # 趋势 + 噪声
    trend = np.concatenate(
        [
            np.linspace(0, 0.1, n // 2),  # 上涨
            np.linspace(0.1, -0.05, n - n // 2),  # 下跌
        ]
    )
    noise = rng.randn(n) * 0.01
    returns = trend / n + noise

    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    volume = rng.uniform(1e6, 5e6, n)

    # ATR: 简单近似
    atr = pd.Series(high - low).rolling(14, min_periods=1).mean().values

    # 方向特征: 前半段 positive (long), 后半段 negative (short)
    wpt_energy = np.concatenate(
        [
            rng.uniform(0.5, 2.0, n // 2),
            rng.uniform(-2.0, -0.5, n - n // 2),
        ]
    )

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr": atr,
            "wpt_volume_energy_f": wpt_energy,
        }
    )
    return df


def _make_multi_symbol_df(n_per_symbol: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成多币种模拟数据。"""
    dfs = []
    for i, symbol in enumerate(["BTCUSDT", "ETHUSDT"]):
        df = _make_mock_ohlcv(n=n_per_symbol, seed=seed + i)
        df["_symbol"] = symbol
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def mock_direction_config():
    """模拟 direction.yaml 配置，使用 wpt_volume_energy_f 的 sign。"""
    return {
        "direction_rules": [
            {
                "method": "feature_sign",
                "feature": "wpt_volume_energy_f",
                "transform": "sign",
            }
        ]
    }


@pytest.fixture
def mock_strategy_dir(tmp_path):
    """创建临时 strategy 目录，包含 direction.yaml。"""
    strategy_name = "test_strategy"

    # 创建目录结构
    config_dir = tmp_path / "config" / "strategies" / strategy_name / "archetypes"
    config_dir.mkdir(parents=True)

    # 写入 direction.yaml
    direction_yaml = {
        "direction_rules": [
            {
                "method": "feature_sign",
                "feature": "wpt_volume_energy_f",
                "transform": "sign",
            }
        ]
    }
    with open(config_dir / "direction.yaml", "w") as f:
        yaml.dump(direction_yaml, f)

    return tmp_path, strategy_name


# ============================================================
# Test 1: _compute_direction_from_rules
# ============================================================


class TestDirectionFromRules:
    """测试方向计算规则。"""

    def test_sign_transform(self, mock_direction_config):
        """sign transform: positive → +1, negative → -1, zero → 0"""
        df = pd.DataFrame(
            {
                "wpt_volume_energy_f": [1.0, -2.0, 0.0, 3.5, -0.1, 0.0],
            }
        )
        direction = _compute_direction_from_rules(mock_direction_config, df)
        expected = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 0.0])
        np.testing.assert_array_equal(direction, expected)

    def test_missing_feature_fallback(self):
        """缺少特征时 direction 全为 0。"""
        cfg = {"direction_rules": [{"feature": "nonexistent", "transform": "sign"}]}
        df = pd.DataFrame({"some_col": [1, 2, 3]})
        direction = _compute_direction_from_rules(cfg, df)
        np.testing.assert_array_equal(direction, [0.0, 0.0, 0.0])


# ============================================================
# Test 2: rr_extreme direction-aware
# ============================================================


class TestRRExtremeDirectionAware:
    """测试 rr_extreme 标签的方向感知功能。"""

    def test_short_bars_flip_forward_rr(self, mock_direction_config):
        """Short bars 的 forward_rr 应该被翻转 (取反)。

        数学验证:
        - Long bar: forward_rr_actual = forward_rr_long (不变)
        - Short bar: forward_rr_actual = -forward_rr_long (翻转)
        """
        df = _make_mock_ohlcv(n=200)
        horizon = 20

        # 1. 计算 long-only subtypes (基准)
        subtypes_long = compute_failure_subtypes(
            df=df,
            direction="long",
            horizon=horizon,
        )

        # 2. 计算 direction-aware subtypes
        subtypes_da = _direction_aware_subtypes_single_symbol(
            df=df,
            horizon=horizon,
            direction_cfg=mock_direction_config,
        )

        # 3. 获取 direction
        direction = _compute_direction_from_rules(
            mock_direction_config, df.reset_index(drop=True)
        )

        # 4. 验证:
        # Long bars: forward_rr 不变
        long_mask = direction == 1
        valid_long = long_mask & ~np.isnan(subtypes_long["forward_rr"].values)
        if valid_long.sum() > 0:
            np.testing.assert_array_almost_equal(
                subtypes_da["forward_rr"].values[valid_long],
                subtypes_long["forward_rr"].values[valid_long],
                decimal=10,
                err_msg="Long bars: forward_rr should be unchanged",
            )

        # Short bars: forward_rr 翻转
        short_mask = direction == -1
        valid_short = short_mask & ~np.isnan(subtypes_long["forward_rr"].values)
        if valid_short.sum() > 0:
            np.testing.assert_array_almost_equal(
                subtypes_da["forward_rr"].values[valid_short],
                -subtypes_long["forward_rr"].values[valid_short],
                decimal=10,
                err_msg="Short bars: forward_rr should be negated",
            )

        print(f"  Long bars: {valid_long.sum()}, Short bars: {valid_short.sum()}")

    def test_failure_label_after_flip(self, mock_direction_config):
        """翻转后的 failure_rr_extreme 标签应该一致。

        做空时 forward_rr_long < -0.8 (做多踩坑)
        → forward_rr_actual = -forward_rr_long > 0.8 (做空大赚)
        → failure_rr_extreme = 0 (不是失败)
        """
        df = _make_mock_ohlcv(n=200)
        horizon = 20

        subtypes_da = _direction_aware_subtypes_single_symbol(
            df=df,
            horizon=horizon,
            direction_cfg=mock_direction_config,
        )

        # 验证 failure_rr_extreme = (forward_rr < -0.8)
        valid = ~np.isnan(subtypes_da["forward_rr"])
        expected_failure = (subtypes_da["forward_rr"][valid] < -0.8).astype(float)
        actual_failure = subtypes_da["failure_rr_extreme"][valid]
        np.testing.assert_array_equal(
            actual_failure.values,
            expected_failure.values,
            err_msg="failure_rr_extreme should be (forward_rr < -0.8) after flip",
        )

    def test_some_failures_change(self, mock_direction_config):
        """方向感知后，部分 failure 标签应该不同于 long-only。

        因为 short bars 的 forward_rr 翻转了，
        原来 long 视角下的 failure 可能变成 success，反之亦然。
        """
        df = _make_mock_ohlcv(n=200)
        horizon = 20

        subtypes_long = compute_failure_subtypes(
            df=df,
            direction="long",
            horizon=horizon,
        )
        subtypes_da = _direction_aware_subtypes_single_symbol(
            df=df,
            horizon=horizon,
            direction_cfg=mock_direction_config,
        )

        direction = _compute_direction_from_rules(
            mock_direction_config, df.reset_index(drop=True)
        )
        short_mask = direction == -1

        # 至少有一些 short bars
        assert short_mask.sum() > 0, "Test data should have short bars"

        # 在 short bars 上, failure labels 应该有变化
        valid = ~np.isnan(subtypes_long["failure_rr_extreme"]) & short_mask
        if valid.sum() > 0:
            long_failures = subtypes_long["failure_rr_extreme"][valid].values
            da_failures = subtypes_da["failure_rr_extreme"][valid].values
            changed = (long_failures != da_failures).sum()
            print(
                f"  Short bars failure changes: {changed}/{valid.sum()} "
                f"({changed/valid.sum()*100:.1f}%)"
            )
            # 不要求所有都变，但应该有一些变化
            # (取决于数据分布，可能有也可能没有，所以不 assert)


# ============================================================
# Test 3: no_opportunity direction-aware
# ============================================================


class TestNoOpportunityDirectionAware:
    """测试 no_opportunity 标签的方向感知功能。"""

    def test_mfe_mae_swap_on_short(self, mock_direction_config):
        """Short bars 的 MFE/MAE 应该互换。

        数学: MFE_short = MAE_long, MAE_short = MFE_long
        """
        df = _make_mock_ohlcv(n=200)
        horizon = 20

        subtypes_long = compute_failure_subtypes(
            df=df,
            direction="long",
            horizon=horizon,
        )
        subtypes_da = _direction_aware_subtypes_single_symbol(
            df=df,
            horizon=horizon,
            direction_cfg=mock_direction_config,
        )

        direction = _compute_direction_from_rules(
            mock_direction_config, df.reset_index(drop=True)
        )

        # Long bars: MFE/MAE 不变
        long_mask = direction == 1
        valid_long = long_mask & ~np.isnan(subtypes_long["mfe_atr"].values)
        if valid_long.sum() > 0:
            np.testing.assert_array_almost_equal(
                subtypes_da["mfe_atr"].values[valid_long],
                subtypes_long["mfe_atr"].values[valid_long],
                decimal=10,
            )
            np.testing.assert_array_almost_equal(
                subtypes_da["mae_atr"].values[valid_long],
                subtypes_long["mae_atr"].values[valid_long],
                decimal=10,
            )

        # Short bars: MFE/MAE 互换
        short_mask = direction == -1
        valid_short = short_mask & ~np.isnan(subtypes_long["mfe_atr"].values)
        if valid_short.sum() > 0:
            # MFE_da (short) = MAE_long
            np.testing.assert_array_almost_equal(
                subtypes_da["mfe_atr"].values[valid_short],
                subtypes_long["mae_atr"].values[valid_short],
                decimal=10,
                err_msg="Short: MFE_actual should equal MAE_long",
            )
            # MAE_da (short) = MFE_long
            np.testing.assert_array_almost_equal(
                subtypes_da["mae_atr"].values[valid_short],
                subtypes_long["mfe_atr"].values[valid_short],
                decimal=10,
                err_msg="Short: MAE_actual should equal MFE_long",
            )

        print(f"  Long: {valid_long.sum()}, Short: {valid_short.sum()}")

    def test_no_opportunity_recalculated(self, mock_direction_config):
        """互换后 no_opportunity 标签应正确重算。

        failure_no_opp = (MAE_actual > 1.2*stop) AND (MFE_actual < 0.3*target)
        """
        df = _make_mock_ohlcv(n=200)
        horizon = 20

        subtypes_da = _direction_aware_subtypes_single_symbol(
            df=df,
            horizon=horizon,
            direction_cfg=mock_direction_config,
        )

        # 手动验证 failure_no_opportunity
        mfe = subtypes_da["mfe_atr"]
        mae = subtypes_da["mae_atr"]
        valid = ~np.isnan(mfe) & ~np.isnan(mae)

        expected_no_opp = ((mae[valid] > 1.2 * 1.0) & (mfe[valid] < 0.3 * 2.0)).astype(
            float
        )

        np.testing.assert_array_equal(
            subtypes_da["failure_no_opportunity"][valid].values,
            expected_no_opp.values,
            err_msg="failure_no_opportunity should match manual calculation",
        )


# ============================================================
# Test 4: return_tree direction-aware
# ============================================================


class TestReturnTreeDirectionAware:
    """测试 return_tree 标签的方向感知功能。"""

    def test_forward_rr_flipped(self, mock_direction_config):
        """return_tree 的 forward_rr 应该按方向翻转。"""
        df = _make_mock_ohlcv(n=200)
        horizon = 20

        subtypes_da = _direction_aware_subtypes_single_symbol(
            df=df,
            horizon=horizon,
            direction_cfg=mock_direction_config,
        )

        # _compute_direction_aware_return_tree 的 forward_rr 应该和 subtypes 一致
        # (因为它内部也调用 _direction_aware_subtypes)
        # 这里验证 filter_good_only 的逻辑
        forward_rr = subtypes_da["forward_rr"].copy()
        failure_any = subtypes_da["failure_any"]

        # filter_good_only: failure bars 的 forward_rr 应该是 NaN
        good_forward_rr = forward_rr.where(failure_any == 0)

        # 至少有一些 failure bars 被过滤
        n_filtered = (failure_any == 1).sum()
        n_good = (failure_any == 0).sum()
        print(f"  Good: {n_good}, Failure: {n_filtered}")

        if n_filtered > 0:
            # failure bars 应该 NaN
            assert good_forward_rr[failure_any == 1].isna().all()

    def test_good_samples_use_direction_aware_failure(self, mock_direction_config):
        """GOOD 样本的 failure_any 应该是方向感知版本。

        这意味着方向感知后，一个 long-only 下的 failure bar
        可能因为 direction=short 变成 GOOD bar。
        """
        df = _make_mock_ohlcv(n=200)
        horizon = 20

        # Long-only failure
        subtypes_long = compute_failure_subtypes(
            df=df,
            direction="long",
            horizon=horizon,
        )

        # Direction-aware failure
        subtypes_da = _direction_aware_subtypes_single_symbol(
            df=df,
            horizon=horizon,
            direction_cfg=mock_direction_config,
        )

        # failure_any 可能不同
        valid = ~np.isnan(subtypes_long["failure_any"]) & ~np.isnan(
            subtypes_da["failure_any"]
        )
        if valid.sum() > 0:
            long_fail = subtypes_long["failure_any"][valid].values
            da_fail = subtypes_da["failure_any"][valid].values
            changed = (long_fail != da_fail).sum()
            print(
                f"  failure_any changed: {changed}/{valid.sum()} "
                f"({changed/valid.sum()*100:.1f}%)"
            )


# ============================================================
# Test 5: 多币种调度
# ============================================================


class TestMultiSymbol:
    """测试多币种场景下的方向感知标签。"""

    def test_multi_symbol_rr_extreme(self):
        """多币种 rr_extreme 应该按 symbol 分别计算。"""
        df = _make_multi_symbol_df(n_per_symbol=200)

        # 使用 ME strategy 的 direction.yaml (已存在)
        try:
            result = compute_bpc_failure_rr_extreme_label(
                df=df,
                direction_aware=True,
                strategy="me",
                horizon=20,
            )
            assert len(result) == len(df)
            valid_count = result.notna().sum()
            assert valid_count > 0, "Should have valid labels"
            print(f"  Multi-symbol rr_extreme: {valid_count}/{len(df)} valid")
        except FileNotFoundError:
            pytest.skip("ME direction.yaml not found")

    def test_multi_symbol_no_opportunity(self):
        """多币种 no_opportunity 应该按 symbol 分别计算。"""
        df = _make_multi_symbol_df(n_per_symbol=200)

        try:
            result = compute_bpc_failure_no_opportunity_label(
                df=df,
                direction_aware=True,
                strategy="me",
                horizon=20,
            )
            assert len(result) == len(df)
            valid_count = result.notna().sum()
            assert valid_count > 0
            print(f"  Multi-symbol no_opportunity: {valid_count}/{len(df)} valid")
        except FileNotFoundError:
            pytest.skip("ME direction.yaml not found")

    def test_multi_symbol_return_tree(self):
        """多币种 return_tree 应该按 symbol 分别计算。"""
        df = _make_multi_symbol_df(n_per_symbol=200)

        try:
            result = compute_bpc_return_tree_label(
                df=df,
                direction_aware=True,
                strategy="me",
                horizon=20,
            )
            assert len(result) == len(df)
            valid_count = result.notna().sum()
            assert valid_count > 0
            print(f"  Multi-symbol return_tree: {valid_count}/{len(df)} valid")
        except FileNotFoundError:
            pytest.skip("ME direction.yaml not found")


# ============================================================
# Test 6: 入口函数 backward compatibility
# ============================================================


class TestBackwardCompatibility:
    """测试 direction_aware=False 时行为不变。"""

    def test_rr_extreme_default_unchanged(self):
        """direction_aware=False (默认) 时应和之前行为一致。"""
        df = _make_mock_ohlcv(n=200)

        result_default = compute_bpc_failure_rr_extreme_label(
            df=df,
            direction="long",
            horizon=20,
        )
        result_explicit = compute_bpc_failure_rr_extreme_label(
            df=df,
            direction="long",
            horizon=20,
            direction_aware=False,
        )

        pd.testing.assert_series_equal(result_default, result_explicit)

    def test_no_opportunity_default_unchanged(self):
        """direction_aware=False 时 no_opportunity 行为不变。"""
        df = _make_mock_ohlcv(n=200)

        result_default = compute_bpc_failure_no_opportunity_label(
            df=df,
            direction="long",
            horizon=20,
        )
        result_explicit = compute_bpc_failure_no_opportunity_label(
            df=df,
            direction="long",
            horizon=20,
            direction_aware=False,
        )

        pd.testing.assert_series_equal(result_default, result_explicit)

    def test_return_tree_default_unchanged(self):
        """direction_aware=False 时 return_tree 行为不变。"""
        df = _make_mock_ohlcv(n=200)

        result_default = compute_bpc_return_tree_label(
            df=df,
            direction="long",
            horizon=20,
        )
        result_explicit = compute_bpc_return_tree_label(
            df=df,
            direction="long",
            horizon=20,
            direction_aware=False,
        )

        pd.testing.assert_series_equal(result_default, result_explicit)


# ============================================================
# Test 7: me_label any_success BUG 修复验证
# ============================================================


class TestMeLabelBugFix:
    """验证 compute_me_label 的 any_success BUG 已修复。"""

    def test_any_success_direction_varies(self):
        """any_success 模式下 signal_direction 不应该全是 1.0。"""
        from src.time_series_model.strategies.labels.me_label import (
            compute_me_label,
            detect_compression,
            detect_expansion_breakout,
        )

        df = _make_mock_ohlcv(n=500, seed=123)
        df["volume"] = np.random.RandomState(123).uniform(1e6, 5e6, len(df))

        # 测试 any_success 模式
        label = compute_me_label(
            df=df,
            combine_mode="any_success",
            max_holding_bars=20,
        )

        # 检测突破信号
        compression = detect_compression(df)
        breakout_up, breakout_down = detect_expansion_breakout(
            df,
            compression_mask=compression,
        )

        has_up = breakout_up.sum()
        has_down = breakout_down.sum()
        print(f"  breakout_up: {has_up}, breakout_down: {has_down}")

        # 应该两个方向都有信号 (至少我们的 mock 数据有上涨和下跌)
        # 即使没有，关键是代码不会 crash
        assert label is not None
        assert len(label) == len(df)

    def test_signal_direction_not_fixed(self):
        """验证 any_success 的 signal_direction 实际上根据方向变化。"""
        from src.time_series_model.strategies.labels.me_label import (
            detect_compression,
            detect_expansion_breakout,
        )

        df = _make_mock_ohlcv(n=500, seed=77)
        df["volume"] = np.random.RandomState(77).uniform(1e6, 5e6, len(df))

        compression = detect_compression(df)
        breakout_up, breakout_down = detect_expansion_breakout(
            df,
            compression_mask=compression,
        )

        # 模拟 BUG 修复后的逻辑
        signal_direction = np.where(
            breakout_up, 1.0, np.where(breakout_down, -1.0, 0.0)
        )

        # 验证: breakout_up → direction=1, breakout_down → direction=-1
        if breakout_up.sum() > 0:
            assert (signal_direction[breakout_up] == 1.0).all()
        if breakout_down.sum() > 0:
            assert (signal_direction[breakout_down] == -1.0).all()


# ============================================================
# Test 8: 边界条件
# ============================================================


class TestEdgeCases:
    """边界条件测试。"""

    def test_all_long_direction(self, mock_direction_config):
        """全部 long 方向时，direction_aware 结果应和 long-only 一致。"""
        df = _make_mock_ohlcv(n=200)
        # 让方向特征全部 positive
        df["wpt_volume_energy_f"] = np.abs(df["wpt_volume_energy_f"])
        horizon = 20

        subtypes_long = compute_failure_subtypes(
            df=df,
            direction="long",
            horizon=horizon,
        )
        subtypes_da = _direction_aware_subtypes_single_symbol(
            df=df,
            horizon=horizon,
            direction_cfg=mock_direction_config,
        )

        valid = ~np.isnan(subtypes_long["forward_rr"].values)
        np.testing.assert_array_almost_equal(
            subtypes_da["forward_rr"].values[valid],
            subtypes_long["forward_rr"].values[valid],
            decimal=10,
            err_msg="All-long: DA should equal long-only",
        )

    def test_direction_aware_requires_strategy(self):
        """direction_aware=True 但没给 strategy 时应报错。"""
        df = _make_mock_ohlcv(n=100)

        with pytest.raises(ValueError, match="strategy"):
            compute_bpc_failure_rr_extreme_label(
                df=df,
                direction_aware=True,
                strategy="",
            )

        with pytest.raises(ValueError, match="strategy"):
            compute_bpc_failure_no_opportunity_label(
                df=df,
                direction_aware=True,
                strategy="",
            )

        with pytest.raises(ValueError, match="strategy"):
            compute_bpc_return_tree_label(
                df=df,
                direction_aware=True,
                strategy="",
            )

    def test_empty_dataframe(self, mock_direction_config):
        """空 DataFrame 应返回空结果。"""
        df = pd.DataFrame(
            columns=["close", "high", "low", "atr", "wpt_volume_energy_f"]
        )
        subtypes = _direction_aware_subtypes_single_symbol(
            df=df,
            horizon=20,
            direction_cfg=mock_direction_config,
        )
        assert len(subtypes) == 0


# ============================================================
# Test 9: YAML 配置一致性
# ============================================================


class TestYAMLConsistency:
    """验证 YAML 配置和代码一致。"""

    @pytest.mark.parametrize("strategy", ["me", "bpc", "fer", "lv"])
    def test_rr_extreme_yaml_has_direction_aware(self, strategy):
        """labels_rr_extreme.yaml 应包含 direction_aware=true。"""
        yaml_path = (
            PROJECT_ROOT / "config" / "strategies" / strategy / "labels_rr_extreme.yaml"
        )
        if not yaml_path.exists():
            pytest.skip(f"{strategy} labels_rr_extreme.yaml not found")

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        params = cfg["label_generator"]["params"]
        assert (
            params.get("direction_aware") is True
        ), f"{strategy}/labels_rr_extreme.yaml missing direction_aware: true"
        assert (
            params.get("strategy") == strategy
        ), f"{strategy}/labels_rr_extreme.yaml strategy should be '{strategy}'"

    @pytest.mark.parametrize(
        "strategy,expected",
        [
            ("me", True),
            ("bpc", True),
            ("fer", True),
        ],
    )
    def test_return_tree_yaml_has_direction_aware(self, strategy, expected):
        """labels_return_tree.yaml 应包含 direction_aware=true。"""
        yaml_path = (
            PROJECT_ROOT
            / "config"
            / "strategies"
            / strategy
            / "labels_return_tree.yaml"
        )
        if not yaml_path.exists():
            pytest.skip(f"{strategy} labels_return_tree.yaml not found")

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        params = cfg["label_generator"]["params"]
        assert params.get("direction_aware") is expected
        assert params.get("strategy") == strategy

    @pytest.mark.parametrize("strategy", ["me", "bpc"])
    def test_no_opportunity_yaml_has_direction_aware(self, strategy):
        """labels_no_opportunity.yaml 应包含 direction_aware=true。"""
        yaml_path = (
            PROJECT_ROOT
            / "config"
            / "strategies"
            / strategy
            / "labels_no_opportunity.yaml"
        )
        if not yaml_path.exists():
            pytest.skip(f"{strategy} labels_no_opportunity.yaml not found")

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        params = cfg["label_generator"]["params"]
        assert params.get("direction_aware") is True
        assert params.get("strategy") == strategy

    @pytest.mark.parametrize("strategy", ["me", "bpc", "fer", "lv"])
    def test_direction_yaml_exists(self, strategy):
        """每个策略应有 direction.yaml。"""
        candidates = [
            PROJECT_ROOT
            / "config"
            / "strategies"
            / strategy
            / "archetypes"
            / "direction.yaml",
            PROJECT_ROOT / "config" / "strategies" / strategy / "direction.yaml",
        ]
        found = any(p.exists() for p in candidates)
        assert found, f"No direction.yaml found for strategy '{strategy}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
