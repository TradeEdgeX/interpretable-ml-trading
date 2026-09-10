"""
测试波动率模型配置加载和训练功能
"""

import sys
import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.volatility_model_config import (
    load_volatility_model_config,
    create_vpin_volatility_features,
    get_volatility_model_params,
    get_categorical_features,
    prepare_volatility_model_data,
)

CORE_GARCH_COLS = [
    "garch_volatility",
    "garch_persistence",
    "garch_leverage_gamma",
    "garch_alpha",
    "garch_beta",
]

CORE_EXTENDED_VOL_COLS = [
    "vol_raw_5",
    "vol_raw_10",
    "vol_raw_20",
    "vol_raw_60",
    "vol_atr_norm",
    "vol_atr_ratio_20",
    "vol_atr_change",
    "vol_atr_change_abs",
    "vol_lag_1",
    "vol_lag_2",
    "vol_lag_3",
    "vol_slope_5",
    "vol_slope_10",
    "vol_slope_20",
    "vol_accel",
    "vol_ma_5",
    "vol_ma_10",
    "vol_ma_20",
    "vol_percentile_approx",
    "vol_range_pos_10",
    "vol_mom_3",
]

CORE_VPIN_COLS = ["vpin_volatility_10", "vpin_volatility_20"]


def create_mock_data(n_samples: int = 1000) -> pd.DataFrame:
    """创建模拟数据用于测试"""
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=n_samples, freq="1H")

    # 基础价格数据
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "close": prices,
            "high": prices * (1 + np.abs(np.random.randn(n_samples) * 0.005)),
            "low": prices * (1 - np.abs(np.random.randn(n_samples) * 0.005)),
            "volume": np.random.lognormal(10, 1, n_samples),
        },
        index=dates,
    )

    # 计算ATR
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()

    # 添加GARCH特征
    df["garch_volatility"] = np.abs(np.random.randn(n_samples)) * 0.02
    df["garch_persistence"] = 0.95 + np.random.randn(n_samples) * 0.05
    df["garch_leverage_gamma"] = np.random.randn(n_samples) * 0.1
    df["garch_alpha"] = 0.1 + np.random.randn(n_samples) * 0.05
    df["garch_beta"] = 0.85 + np.random.randn(n_samples) * 0.05

    # 添加扩展波动率特征
    returns_pct = df["close"].pct_change()
    df["vol_raw_5"] = returns_pct.rolling(5, min_periods=1).std()
    df["vol_raw_10"] = returns_pct.rolling(10, min_periods=1).std()
    df["vol_raw_20"] = returns_pct.rolling(20, min_periods=1).std()
    df["vol_raw_60"] = returns_pct.rolling(60, min_periods=1).std()
    df["vol_atr_norm"] = df["atr"] / (df["close"] + 1e-6)
    df["vol_atr_ratio_20"] = df["atr"] / (
        df["atr"].rolling(20, min_periods=1).mean() + 1e-6
    )
    df["vol_atr_change"] = df["atr"].pct_change()
    df["vol_atr_change_abs"] = df["vol_atr_change"].abs()
    df["vol_lag_1"] = df["vol_raw_5"].shift(1)
    df["vol_lag_2"] = df["vol_raw_5"].shift(2)
    df["vol_lag_3"] = df["vol_raw_5"].shift(3)
    df["vol_slope_5"] = np.random.randn(n_samples) * 0.01
    df["vol_slope_10"] = np.random.randn(n_samples) * 0.01
    df["vol_slope_20"] = np.random.randn(n_samples) * 0.01
    df["vol_accel"] = np.random.randn(n_samples) * 0.01
    df["vol_ma_5"] = df["vol_raw_5"].rolling(5, min_periods=1).mean()
    df["vol_ma_10"] = df["vol_raw_10"].rolling(10, min_periods=1).mean()
    df["vol_ma_20"] = df["vol_raw_20"].rolling(20, min_periods=1).mean()
    df["vol_percentile_approx"] = df["vol_raw_20"].rank(pct=True).fillna(0)
    df["vol_range_pos_10"] = np.random.randn(n_samples) * 0.05
    df["vol_mom_3"] = returns_pct.rolling(3, min_periods=1).sum()

    # 添加VPIN特征
    df["vpin"] = np.random.uniform(0, 1, n_samples)
    df["vpin_volatility_10"] = df["vpin"].rolling(10).std()
    df["vpin_volatility_20"] = df["vpin"].rolling(20).std()

    # 添加压缩/ATR衍生特征
    df["bb_width"] = np.random.randn(n_samples) * 0.01
    df["bb_width_normalized"] = df["bb_width"] / (df["bb_width"].abs().max() + 1e-6)
    df["bb_width_ratio"] = np.random.randn(n_samples) * 0.05
    df["compression"] = np.random.randn(n_samples) * 0.5
    df["range_ratio"] = np.random.randn(n_samples) * 0.1
    df["atr_ratio"] = df["atr"] / (df["close"].rolling(5, min_periods=1).mean() + 1e-6)

    # 添加一些不需要的特征（应该被排除）
    df["evt_risk"] = np.random.randn(n_samples)  # 应该被排除
    df["dtw_distance"] = np.random.randn(n_samples)  # 应该被排除
    df["label"] = np.random.randint(0, 2, n_samples)  # 应该被排除
    df["signal"] = np.random.randint(-1, 2, n_samples)  # 应该被排除

    return df.fillna(method="bfill").fillna(method="ffill")


def test_config_loading():
    """测试配置文件加载"""
    print("=" * 70)
    print("测试1: 配置文件加载")
    print("=" * 70)

    try:
        config = load_volatility_model_config()
        print("✅ 配置文件加载成功")
        print(f"   - 配置名称: {config.get('name', 'N/A')}")
        print(f"   - 模型类型: {config.get('trainer', {}).get('model_type', 'N/A')}")
        groups = config.get("volatility_features", {}).get("groups", [])
        print(f"   - 特征分组数: {len(groups)}")
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        raise


def test_vpin_feature_engineering(df):
    """测试VPIN特征工程"""
    print("\n" + "=" * 70)
    print("测试3: VPIN特征工程")
    print("=" * 70)

    try:
        config = load_volatility_model_config()
        df_enhanced = create_vpin_volatility_features(df, config)

        # 检查是否创建了衍生特征
        expected_features = ["vpin_vol_ratio", "vpin_vol_zscore", "vpin_spike"]
        created_features = [f for f in expected_features if f in df_enhanced.columns]

        print(f"✅ VPIN特征工程成功")
        print(f"   - 创建的衍生特征: {created_features}")

        if len(created_features) > 0:
            print(f"   - 特征统计:")
            for feat in created_features:
                if feat in df_enhanced.columns:
                    print(
                        f"     * {feat}: mean={df_enhanced[feat].mean():.4f}, std={df_enhanced[feat].std():.4f}"
                    )

        return df_enhanced
    except Exception as e:
        print(f"❌ VPIN特征工程失败: {e}")
        raise


def test_prepare_volatility_model_data(df):
    """测试准备波动率模型数据"""
    print("\n" + "=" * 70)
    print("测试4: 准备波动率模型数据")
    print("=" * 70)

    try:
        config = load_volatility_model_config()
        X_vol, vol_features, categorical_features = prepare_volatility_model_data(
            df, config
        )

        print(f"✅ 数据准备成功")
        print(f"   - 数据形状: {X_vol.shape}")
        print(f"   - 特征数: {len(vol_features)}")
        print(
            f"   - 分类特征: {categorical_features if categorical_features else 'None'}"
        )

        # 核心特征应全部存在
        for expected_col in CORE_GARCH_COLS + CORE_VPIN_COLS:
            assert expected_col in vol_features, f"缺少核心特征 {expected_col}"

        missing_core = [
            col for col in CORE_EXTENDED_VOL_COLS if col not in vol_features
        ]
        if missing_core:
            raise AssertionError(f"缺少扩展波动率特征: {missing_core}")

        return X_vol, vol_features, categorical_features
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        raise


def test_model_training(X_vol, y_vol):
    """测试模型训练"""
    print("\n" + "=" * 70)
    print("测试5: 模型训练")
    print("=" * 70)

    try:
        from src.time_series_model.strategies.models.lightgbm_model import (
            LightGBMTrainer,
        )

        config = load_volatility_model_config()
        trainer_config = config.get("trainer", {})
        use_gpu = trainer_config.get("use_gpu", False)  # 测试时使用CPU
        model_params = get_volatility_model_params(config)
        n_splits = trainer_config.get("n_splits", 3)  # 测试时使用较少的splits

        # 获取分类特征
        categorical_features = get_categorical_features(X_vol, config)

        print(f"   - 使用GPU: {use_gpu}")
        print(f"   - 交叉验证折数: {n_splits}")
        print(f"   - 特征数: {X_vol.shape[1]}")
        print(f"   - 样本数: {X_vol.shape[0]}")
        if categorical_features:
            print(f"   - 分类特征: {categorical_features}")

        # 分割训练/测试集
        split_idx = int(len(X_vol) * 0.8)
        X_train = X_vol.iloc[:split_idx]
        X_test = X_vol.iloc[split_idx:]
        y_train = y_vol.iloc[:split_idx]
        y_test = y_vol.iloc[split_idx:]

        # 创建模型
        model = LightGBMTrainer(model_type="regression", use_gpu=use_gpu)

        # 初始化分类特征属性（如果不存在）
        if not hasattr(model, "_categorical_features"):
            model._categorical_features = None

        # 设置模型参数
        if model_params:
            model.params = model_params

        # 设置分类特征
        if categorical_features:
            model._categorical_features = categorical_features

        # 训练模型
        print("   - 开始训练...")
        metrics, _ = model.train(
            X_train,
            y_train,
            n_splits=n_splits,
            use_time_series_cv=True,
            groups=None,
            auto_tune_params=False,
        )

        print(f"✅ 模型训练成功")
        print(f"   - 训练指标:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"     * {key}: {value:.6f}")
            else:
                print(f"     * {key}: {value}")

        # 测试预测
        print("   - 测试预测...")
        predictions = model.predict(X_test)
        print(f"   ✅ 预测成功")
        print(f"     * 预测形状: {predictions.shape}")
        print(f"     * 预测均值: {predictions.mean():.6f}")
        print(f"     * 预测标准差: {predictions.std():.6f}")

        # 检查特征列表是否存储
        if hasattr(model, "_volatility_features"):
            print(f"   ✅ 特征列表已存储: {len(model._volatility_features)} 个特征")

        return model, metrics
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("波动率模型配置和训练测试")
    print("=" * 70 + "\n")

    try:
        # 测试1: 加载配置
        config = test_config_loading()

        # 创建模拟数据
        print("\n" + "=" * 70)
        print("创建模拟数据...")
        print("=" * 70)
        df = create_mock_data(n_samples=500)
        print(f"✅ 创建模拟数据成功: {df.shape}")

        # 测试2: VPIN特征工程
        df_enhanced = test_vpin_feature_engineering(df)

        # 测试3: 准备数据
        X_vol, vol_features, categorical_features = test_prepare_volatility_model_data(
            df_enhanced
        )

        # 创建波动率标签（模拟）
        print("\n" + "=" * 70)
        print("创建波动率标签...")
        print("=" * 70)
        y_vol = df_enhanced["close"].pct_change().rolling(10).std().shift(-10)
        y_vol = y_vol.fillna(y_vol.mean())
        print(f"✅ 创建波动率标签成功: {y_vol.shape}")
        print(f"   - 标签均值: {y_vol.mean():.6f}")
        print(f"   - 标签标准差: {y_vol.std():.6f}")

        # 测试5: 模型训练
        model, metrics = test_model_training(X_vol, y_vol)

        # 总结
        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)
        print("\n测试总结:")
        print(f"  ✅ 配置文件加载正常")
        print(f"  ✅ VPIN特征工程正常")
        print(f"  ✅ 数据准备正常")
        print(f"  ✅ 模型训练和预测正常")
        print("\n")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ 测试失败")
        print("=" * 70)
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
