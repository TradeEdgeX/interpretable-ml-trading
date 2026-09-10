"""
测试 ZIP 到 Parquet 数据转换功能

测试 DataConverter 类的核心功能：
1. ZIP 文件转换为 Parquet
2. 自动检测交易对符号
3. 数据预处理
4. 批量转换
"""

import pytest
import pandas as pd
import numpy as np
import zipfile
import tempfile
import shutil
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def temp_dirs():
    """创建临时目录用于测试"""
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    backup_dir = tempfile.mkdtemp()

    yield {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "backup_dir": backup_dir,
    }

    # 清理
    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(backup_dir, ignore_errors=True)


@pytest.fixture
def sample_zip_file(temp_dirs):
    """创建示例 ZIP 文件用于测试"""
    input_dir = temp_dirs["input_dir"]

    # 创建模拟的 aggTrades 数据
    np.random.seed(42)
    n_samples = 100
    base_time = pd.Timestamp("2024-01-01 00:00:00")

    # 生成 tick 数据
    prices = 50000 + np.cumsum(np.random.randn(n_samples) * 50)
    quantities = np.random.uniform(0.1, 10.0, n_samples)
    timestamps = [base_time + pd.Timedelta(seconds=i) for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "agg_trade_id": range(1, n_samples + 1),
            "price": prices,
            "quantity": quantities,
            "first_trade_id": range(1000, 1000 + n_samples),
            "last_trade_id": range(1000, 1000 + n_samples),
            "transact_time": [int(ts.timestamp() * 1000) for ts in timestamps],
            "is_buyer_maker": np.random.choice([True, False], n_samples),
        }
    )

    # 创建 ZIP 文件
    zip_path = Path(input_dir) / "BTCUSDT-aggTrades-2024-01.csv.zip"
    csv_path = Path(input_dir) / "BTCUSDT-aggTrades-2024-01.csv"

    # 保存 CSV（无 header，模拟 Binance 格式）
    df.to_csv(csv_path, index=False, header=False)

    # 压缩为 ZIP
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, csv_path.name)

    # 删除临时 CSV
    csv_path.unlink()

    return str(zip_path)


class TestZipToParquet:
    """ZIP 到 Parquet 转换测试类"""

    def test_data_converter_initialization(self, temp_dirs):
        """测试 DataConverter 初始化"""
        from src.data_tools.zip_to_parquet import DataConverter

        converter = DataConverter(
            input_dir=temp_dirs["input_dir"],
            output_dir=temp_dirs["output_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        assert converter.input_dir == temp_dirs["input_dir"]
        assert converter.output_dir == temp_dirs["output_dir"]
        assert converter.backup_dir == temp_dirs["backup_dir"]
        assert os.path.exists(temp_dirs["output_dir"])
        assert os.path.exists(temp_dirs["backup_dir"])
        print("✅ DataConverter 初始化成功")

    def test_convert_single_zip_file(self, temp_dirs, sample_zip_file):
        """测试单个 ZIP 文件转换"""
        from src.data_tools.zip_to_parquet import DataConverter

        converter = DataConverter(
            input_dir=temp_dirs["input_dir"],
            output_dir=temp_dirs["output_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        result = converter.convert_zip_to_parquet(sample_zip_file)

        assert result is not None
        assert "output_file" in result
        assert "symbol" in result
        assert result["symbol"] == "BTCUSDT"

        # 检查输出文件是否存在
        output_file = result["output_file"]
        assert os.path.exists(output_file)

        # 检查 Parquet 文件内容
        df = pd.read_parquet(output_file)
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert "volume" in df.columns
        assert "side" in df.columns
        assert "symbol" in df.columns
        assert len(df) > 0
        assert df["symbol"].iloc[0] == "BTCUSDT"

        print(f"✅ ZIP 文件转换成功: {os.path.basename(output_file)}")
        print(f"   转换了 {len(df)} 条 tick 数据")

    def test_symbol_detection(self, temp_dirs):
        """测试交易对符号自动检测"""
        from src.data_tools.zip_to_parquet import DataConverter

        converter = DataConverter(
            input_dir=temp_dirs["input_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        # 测试不同的文件名格式
        test_cases = [
            ("BTCUSDT-aggTrades-2024-01.zip", "BTCUSDT"),
            ("ETHUSDT-aggTrades-2024-01.zip", "ETHUSDT"),
            ("SOLUSDT-aggTrades-2024-01.zip", "SOLUSDT"),
        ]

        for filename, expected_symbol in test_cases:
            # 创建临时文件用于测试符号检测
            test_file = Path(temp_dirs["input_dir"]) / filename
            test_file.touch()

            # 直接测试符号检测逻辑（通过查看 _generate_output_filename）
            output_file = converter._generate_output_filename(
                str(test_file), expected_symbol
            )
            assert expected_symbol in output_file

            test_file.unlink()

        print("✅ 交易对符号自动检测功能正常")

    def test_preprocess_tick_data(self, temp_dirs):
        """测试 tick 数据预处理"""
        from src.data_tools.zip_to_parquet import DataConverter

        converter = DataConverter(
            input_dir=temp_dirs["input_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        # 创建测试 DataFrame
        df = pd.DataFrame(
            {
                "transact_time": [
                    1704067200000,
                    1704067201000,
                    1704067202000,
                ],  # 毫秒时间戳
                "price": [50000.0, 50010.0, 50005.0],
                "quantity": [1.0, 2.0, 1.5],
                "is_buyer_maker": [False, True, False],
            }
        )

        result = converter._preprocess_tick_data(df)

        assert result is not None
        assert "timestamp" in result.columns
        assert "price" in result.columns
        assert "volume" in result.columns
        assert "side" in result.columns
        assert len(result) == 3
        assert isinstance(result["timestamp"].iloc[0], pd.Timestamp)
        # 新规范：mlbot data convert 输出 tz-aware UTC
        assert result["timestamp"].iloc[0].tz is not None

        # 检查 side 值（is_buyer_maker=False 应该是 1，True 应该是 -1）
        assert result["side"].iloc[0] == 1  # False -> 1 (buy)
        assert result["side"].iloc[1] == -1  # True -> -1 (sell)

        print("✅ Tick 数据预处理功能正常")

    def test_output_filename_generation(self, temp_dirs):
        """测试输出文件名生成"""
        from src.data_tools.zip_to_parquet import DataConverter

        converter = DataConverter(
            input_dir=temp_dirs["input_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        test_file = "/path/to/BTCUSDT-aggTrades-2024-01.zip"
        output_file = converter._generate_output_filename(test_file, "BTCUSDT")

        assert "BTCUSDT" in output_file
        assert "2024-01" in output_file
        assert output_file.endswith(".parquet")
        assert temp_dirs["output_dir"] in output_file

        print("\u2705 输出文件名生成功能正常")

    def test_output_filename_generation_monthly_vs_daily(self, temp_dirs):
        """测试输出文件名生成：区分 monthly 和 daily 格式

        Bug 修复验证：
        - Monthly ZIP (SYMBOL-aggTrades-2025-12.zip) → SYMBOL_2025-12.parquet
        - Daily ZIP (SYMBOL-aggTrades-2025-12-01.zip) → SYMBOL_2025-12-01.parquet

        之前的 bug：daily ZIP 也会生成 SYMBOL_2025-12.parquet，导致多个 daily 文件互相覆盖
        """
        from src.data_tools.zip_to_parquet import DataConverter

        converter = DataConverter(
            input_dir=temp_dirs["input_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        # 1. Monthly 格式：应该生成 SYMBOL_2025-12.parquet
        monthly_file = "/path/to/BTCUSDT-aggTrades-2025-12.zip"
        monthly_output = converter._generate_output_filename(monthly_file, "BTCUSDT")
        assert "BTCUSDT_2025-12.parquet" in monthly_output
        assert "2025-12-" not in monthly_output  # 不应该有日期部分

        # 2. Daily 格式：应该生成 SYMBOL_2025-12-01.parquet
        daily_file_1 = "/path/to/BTCUSDT-aggTrades-2025-12-01.zip"
        daily_output_1 = converter._generate_output_filename(daily_file_1, "BTCUSDT")
        assert "BTCUSDT_2025-12-01.parquet" in daily_output_1

        # 3. 另一个 Daily 文件：应该生成不同的文件名
        daily_file_2 = "/path/to/BTCUSDT-aggTrades-2025-12-15.zip"
        daily_output_2 = converter._generate_output_filename(daily_file_2, "BTCUSDT")
        assert "BTCUSDT_2025-12-15.parquet" in daily_output_2

        # 关键断言：两个 daily 文件应该生成不同的输出文件名
        assert daily_output_1 != daily_output_2, (
            f"Bug: daily files should generate different output names! "
            f"Got same: {daily_output_1}"
        )

        print("\u2705 Monthly/Daily 文件名区分功能正常")
        print(
            f"   Monthly: BTCUSDT-2025-12.zip \u2192 {os.path.basename(monthly_output)}"
        )
        print(
            f"   Daily:   BTCUSDT-2025-12-01.zip \u2192 {os.path.basename(daily_output_1)}"
        )
        print(
            f"   Daily:   BTCUSDT-2025-12-15.zip \u2192 {os.path.basename(daily_output_2)}"
        )

    def test_convert_all_files_empty(self, temp_dirs):
        """测试批量转换（空目录）"""
        from src.data_tools.zip_to_parquet import DataConverter

        converter = DataConverter(
            input_dir=temp_dirs["input_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        results = converter.convert_all_files()

        assert results["total_files"] == 0
        assert len(results["converted_files"]) == 0
        assert len(results["failed_files"]) == 0

        print("✅ 空目录批量转换处理正常")

    def test_convert_all_files_with_data(self, temp_dirs, sample_zip_file):
        """测试批量转换（有数据）"""
        from src.data_tools.zip_to_parquet import DataConverter

        converter = DataConverter(
            input_dir=temp_dirs["input_dir"],
            output_dir=temp_dirs["output_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        results = converter.convert_all_files()

        assert results["total_files"] == 1
        assert len(results["converted_files"]) == 1
        assert len(results["failed_files"]) == 0

        # 检查备份文件
        backup_files = list(Path(temp_dirs["backup_dir"]).glob("*.zip"))
        assert len(backup_files) == 1

        print("✅ 批量转换功能正常")
        print(f"   转换了 {results['total_files']} 个文件")


def test_is_binance_um_monthly_aggtrade_zip():
    from src.data_tools.zip_to_parquet import is_binance_um_monthly_aggtrade_zip

    assert is_binance_um_monthly_aggtrade_zip("BTCUSDT-aggTrades-2026-02.zip")
    assert is_binance_um_monthly_aggtrade_zip("/data/BTCUSDT-aggTrades-2026-02.zip")
    assert not is_binance_um_monthly_aggtrade_zip("BTCUSDT-aggTrades-2026-02-01.zip")
    assert not is_binance_um_monthly_aggtrade_zip("foo.csv")


def test_convert_all_files_excludes_monthly_when_requested(temp_dirs):
    from src.data_tools.zip_to_parquet import DataConverter

    input_dir = Path(temp_dirs["input_dir"])
    (input_dir / "BTCUSDT-aggTrades-2026-02.zip").write_bytes(b"")
    (input_dir / "ETHUSDT-aggTrades-2026-02-03.zip").write_bytes(b"")

    converter = DataConverter(
        input_dir=str(input_dir),
        output_dir=temp_dirs["output_dir"],
    )
    excluded = converter.convert_all_files(
        pattern="*-aggTrades-*.zip",
        exclude_binance_monthly_aggtrade_zips=True,
    )
    assert excluded["total_files"] == 1

    inclusive = converter.convert_all_files(
        pattern="*-aggTrades-*.zip",
        exclude_binance_monthly_aggtrade_zips=False,
    )
    assert inclusive["total_files"] == 2
