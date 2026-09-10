"""Regime layer — A/B/C 共用慢变量数据空间约束 + 多空掩码。

主要模块:
    - ``threshold_calibrator``  : Tier-0 季度 plateau 校准核心（纯函数 + IO 分离）。

运行时入口在 ``src/time_series_model/archetype/loader.py`` 的 ``RegimeConfig``。
"""
