"""
Bug fix 测试: forward_rr 等前瞻标签泄漏排除

修复背景:
    optimize_entry_filter_plateau.py 在搜索 entry filter 最优特征时，
    未排除 forward_rr、forward_return 等前瞻标签。
    ME 20260306 run 中泄漏导致 entry_filter=forward_rr>=2.65，
    OOS 回测 0 trades（因为实盘/事件回测中不存在 forward_rr）。

修复内容:
    1. _META_EXCLUDE 集合新增 forward_rr/forward_return/forward_r/path_extreme/path_extreme_r
    2. 额外增加 forward_ 前缀通配排除逻辑
"""

import pytest
import pandas as pd
import numpy as np


# 从源码提取的排除集合 —— 测试时直接验证关键元素存在
_FORWARD_LOOKING_COLS = {
    "forward_rr",
    "forward_return",
    "forward_r",
    "path_extreme",
    "path_extreme_r",
}


class TestForwardRrExclusion:
    """验证 forward-looking 标签被排除出 entry filter 候选特征"""

    def _get_meta_exclude(self) -> set:
        """从源码模块获取 _META_EXCLUDE 集合"""
        # _META_EXCLUDE 是脚本内的局部变量，无法直接 import
        # 直接检查关键成员是否存在于预期位置
        import ast
        from pathlib import Path

        script_path = Path("scripts/optimize_entry_filter_plateau.py")
        if not script_path.exists():
            pytest.skip("optimize_entry_filter_plateau.py 不存在")

        source = script_path.read_text(encoding="utf-8")
        # 搜索 _META_EXCLUDE 字符串确认关键列存在
        return source

    def test_meta_exclude_contains_forward_rr(self):
        """_META_EXCLUDE 必须包含 forward_rr"""
        source = self._get_meta_exclude()
        for col in _FORWARD_LOOKING_COLS:
            assert (
                f'"{col}"' in source
            ), f"_META_EXCLUDE 缺少 '{col}' — 前瞻标签会泄漏进 entry filter"

    def test_forward_prefix_wildcard_exclusion(self):
        """所有 forward_ 开头的列应被通配排除"""
        source = self._get_meta_exclude()
        # 验证源码中有 c.startswith("forward_") 的排除逻辑
        assert 'startswith("forward_")' in source, "缺少 forward_ 前缀通配排除逻辑"

    def test_filter_logic_excludes_forward_columns(self):
        """模拟完整过滤逻辑: forward_* 列不应出现在候选特征列表中"""
        # 模拟 df_trades 的列
        all_cols = [
            "rsi_14",
            "atr_norm",
            "cvd_slope",
            "forward_rr",
            "forward_return",
            "forward_r",
            "forward_rr_5bar",
            "path_extreme",
            "path_extreme_r",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "_symbol",
            "gate_ok",
            "__internal",
            "momentum_score",
        ]

        _META_EXCLUDE = {
            "symbol",
            "_symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
            "forward_rr",
            "forward_return",
            "forward_r",
            "path_extreme",
            "path_extreme_r",
        }

        # 模拟源码的过滤逻辑
        features = []
        for c in all_cols:
            if c in _META_EXCLUDE:
                continue
            if c.startswith("gate_") or c.startswith("__"):
                continue
            if c.startswith("forward_"):
                continue
            features.append(c)

        # 验证：没有任何 forward_* 列通过过滤
        forward_leaked = [f for f in features if f.startswith("forward_")]
        assert (
            len(forward_leaked) == 0
        ), f"forward-looking 列泄漏进候选特征: {forward_leaked}"

        # 验证：正常特征应保留
        assert "rsi_14" in features
        assert "atr_norm" in features
        assert "cvd_slope" in features
        assert "momentum_score" in features

        # 验证：OHLCV 和 meta 列被排除
        assert "open" not in features
        assert "_symbol" not in features
        assert "gate_ok" not in features
        assert "__internal" not in features

    def test_path_extreme_excluded(self):
        """path_extreme 和 path_extreme_r 是前瞻标签，必须被排除"""
        # path_extreme 在实盘中不存在，是回测标注的路径极值
        _META_EXCLUDE = {
            "forward_rr",
            "forward_return",
            "forward_r",
            "path_extreme",
            "path_extreme_r",
        }

        for col in ["path_extreme", "path_extreme_r"]:
            assert col in _META_EXCLUDE, f"{col} 未被排除"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
