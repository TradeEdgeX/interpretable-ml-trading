"""
四种策略特征加载测试

专门测试四种策略（SR Reversal, SR Breakout, Compression Breakout, Trend Following）
的特征是否能正确加载
"""

from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到路径

from src.features.loader import StrategyFeatureLoader


def create_test_data(n_samples=200):
    """创建测试数据"""
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "open": np.random.randn(n_samples).cumsum() + 100,
            "high": np.random.randn(n_samples).cumsum() + 101,
            "low": np.random.randn(n_samples).cumsum() + 99,
            "close": np.random.randn(n_samples).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, n_samples),
            "cvd": np.random.randn(n_samples).cumsum(),
            "taker_buy_ratio": np.random.uniform(0.3, 0.7, n_samples),
        }
    )

    # 确保价格合理
    df["close"] = df["close"].abs() + 50
    df["high"] = df["close"] + np.abs(np.random.randn(n_samples))
    df["low"] = df["close"] - np.abs(np.random.randn(n_samples))
    df["open"] = df["close"] + np.random.randn(n_samples) * 0.5

    return df


def test_strategy_features():
    """测试四种策略的特征加载"""
    print("=" * 70)
    print("四种策略特征加载测试")
    print("=" * 70)
    print()

    # 创建测试数据
    df = create_test_data(200)
    print(f"测试数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"数据列: {list(df.columns)}")
    print()

    # 初始化加载器（不再使用 strategy_features.yaml）
    try:
        loader = StrategyFeatureLoader(
            feature_deps_path="config/feature_dependencies.yaml",
            strategy_config_path=None,  # 不再使用 strategy_features.yaml
            cache_dir="cache/features",
            use_disk_cache=True,
            use_memory_cache=True,
            max_workers=2,
            parallel_backend="process",
        )
        print("✅ 特征加载器初始化成功")
        print()
    except Exception as e:
        print(f"❌ 特征加载器初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 测试四种策略
    strategies = [
        "sr_reversal",
        "sr_breakout",
        "compression_breakout",
        "trend_following",
    ]

    results = {}

    for strategy in strategies:
        print("=" * 70)
        print(f"测试策略: {strategy.upper()}")
        print("=" * 70)

        try:
            # 从目录管理方式读取特征配置
            from src.time_series_model.strategy_config import StrategyConfigLoader
            import yaml

            strategy_dir = Path(f"config/strategies/{strategy}")
            if strategy_dir.exists():
                config_loader = StrategyConfigLoader(strategy_dir)
                strategy_config = config_loader.load()
                requested_features = strategy_config.features.requested_features
            else:
                # Fallback: 直接从 features.yaml 读取
                features_path = strategy_dir / "features.yaml"
                if features_path.exists():
                    with open(features_path, "r", encoding="utf-8") as f:
                        features_data = yaml.safe_load(f)
                    requested_features = features_data.get("feature_pipeline", {}).get(
                        "requested_features", []
                    )
                else:
                    requested_features = ["atr", "rsi"]

            print(f"请求的特征: {requested_features}")
            print()

            # 加载特征
            print("开始加载特征...")
            result_df = loader.load_features_from_requested(
                df.copy(), requested_features, fit=True
            )

            # 分析结果
            original_cols = len(df.columns)
            new_cols = [c for c in result_df.columns if c not in df.columns]
            total_cols = len(result_df.columns)

            print()
            print(f"✅ 特征加载成功")
            print(f"   原始列数: {original_cols}")
            print(f"   新增列数: {len(new_cols)}")
            print(f"   总列数: {total_cols}")

            if new_cols:
                print(f"   新增列示例 (前10个): {new_cols[:10]}")

            # 检查请求的特征是否在结果中
            print()
            print("检查请求的特征:")
            for feature in requested_features:
                # 检查特征配置
                if feature in loader.feature_deps.get("features", {}):
                    feature_info = loader.feature_deps["features"][feature]
                    output_cols = feature_info.get("output_columns", [feature])

                    found_cols = [c for c in output_cols if c in result_df.columns]
                    if found_cols:
                        print(f"   ✅ {feature}: {found_cols}")
                    else:
                        print(f"   ⚠️  {feature}: 未找到输出列 {output_cols}")
                else:
                    print(f"   ⚠️  {feature}: 未在特征依赖配置中找到")

            results[strategy] = {
                "success": True,
                "original_cols": original_cols,
                "new_cols_count": len(new_cols),
                "total_cols": total_cols,
                "new_cols": new_cols,
            }

        except Exception as e:
            print(f"❌ 特征加载失败: {e}")
            import traceback

            traceback.print_exc()
            results[strategy] = {
                "success": False,
                "error": str(e),
            }

        print()

    # 总结
    print("=" * 70)
    print("测试总结")
    print("=" * 70)
    print()

    success_count = sum(1 for r in results.values() if r.get("success", False))
    total_count = len(results)

    print(f"成功: {success_count}/{total_count}")
    print()

    for strategy, result in results.items():
        if result.get("success"):
            print(f"✅ {strategy}:")
            print(f"   原始列数: {result['original_cols']}")
            print(f"   新增列数: {result['new_cols_count']}")
            print(f"   总列数: {result['total_cols']}")
        else:
            print(f"❌ {strategy}: {result.get('error', 'Unknown error')}")
        print()

    print("=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_strategy_features()
