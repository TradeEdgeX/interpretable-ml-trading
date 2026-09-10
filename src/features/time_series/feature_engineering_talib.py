"""Enhanced feature engineering module using TA-Lib indicators.

基于TA-Lib库的增强特征工程模块，提供：
1. 使用TA-Lib替换自实现的技术指标
2. 添加更多传统技术指标（趋势、动量、波动率、成交量等）
3. 特征归一化（StandardScaler/MinMaxScaler/RobustScaler）
4. Scaler保存和加载功能

TA-Lib提供158+种技术指标，比自实现更准确和高效。
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import warnings

# 忽略TA-Lib的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)


class TalibFeatureEngineer:
    """Enhanced feature engineer using TA-Lib indicators."""

    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize the TA-Lib feature engineer.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scalers = {}  # Store scalers for each timeframe
        self.feature_stats = {}  # Store feature statistics

        # Choose scaler
        if scaler_type == "standard":
            self.scaler_class = StandardScaler
        elif scaler_type == "minmax":
            self.scaler_class = MinMaxScaler
        elif scaler_type == "robust":
            self.scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    def add_trend_indicators(
        self, data: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """添加趋势类指标，如果指定了required_features，只计算需要的特征."""
        df = data.copy()

        # 简单移动平均线 (SMA)
        for period in [5, 10, 20, 50, 100, 200]:
            col = f"sma_{period}"
            if col not in df.columns:
                if not required_features or col in required_features:
                    df[col] = talib.SMA(df["close"].values, timeperiod=period)

        # 指数移动平均线 (EMA)
        for period in [5, 10, 20, 50, 100]:
            col = f"ema_{period}"
            if col not in df.columns:
                if not required_features or col in required_features:
                    df[col] = talib.EMA(df["close"].values, timeperiod=period)

        # 加权移动平均线 (WMA)
        for period in [10, 20, 50]:
            col = f"wma_{period}"
            if col not in df.columns:
                if not required_features or col in required_features:
                    df[col] = talib.WMA(df["close"].values, timeperiod=period)

        # 三角移动平均线 (TEMA)
        for period in [10, 20, 30]:
            col = f"tema_{period}"
            if col not in df.columns:
                if not required_features or col in required_features:
                    df[col] = talib.TEMA(df["close"].values, timeperiod=period)

        # 考夫曼自适应移动平均线 (KAMA)
        for period in [10, 20, 30]:
            col = f"kama_{period}"
            if col not in df.columns:
                if not required_features or col in required_features:
                    df[col] = talib.KAMA(df["close"].values, timeperiod=period)

        # 抛物线SAR
        if "sar" not in df.columns:
            if not required_features or "sar" in required_features:
                df["sar"] = talib.SAR(df["high"].values, df["low"].values)
        if "sar_ext" not in df.columns:
            if not required_features or "sar_ext" in required_features:
                df["sar_ext"] = talib.SAREXT(df["high"].values, df["low"].values)

        # 平均方向指数 (ADX)
        if "adx" not in df.columns:
            if not required_features or "adx" in required_features:
                df["adx"] = talib.ADX(
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                    timeperiod=14,
                )
        if "adxr" not in df.columns:
            if not required_features or "adxr" in required_features:
                df["adxr"] = talib.ADXR(
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                    timeperiod=14,
                )

        # 正负方向指标
        if "plus_di" not in df.columns:
            if not required_features or "plus_di" in required_features:
                df["plus_di"] = talib.PLUS_DI(
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                    timeperiod=14,
                )
        if "minus_di" not in df.columns:
            if not required_features or "minus_di" in required_features:
                df["minus_di"] = talib.MINUS_DI(
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                    timeperiod=14,
                )

        # 阿隆指标
        need_aroon = not required_features or any(
            f in required_features for f in ["aroon_up", "aroon_down", "aroon_osc"]
        )
        if need_aroon:
            if "aroon_up" not in df.columns or "aroon_down" not in df.columns:
                aroon_up, aroon_down = talib.AROON(
                    df["high"].values, df["low"].values, timeperiod=14
                )
                if "aroon_up" not in df.columns:
                    if not required_features or "aroon_up" in required_features:
                        df["aroon_up"] = aroon_up
                if "aroon_down" not in df.columns:
                    if not required_features or "aroon_down" in required_features:
                        df["aroon_down"] = aroon_down
            if "aroon_osc" not in df.columns:
                if not required_features or "aroon_osc" in required_features:
                    df["aroon_osc"] = talib.AROONOSC(
                        df["high"].values, df["low"].values, timeperiod=14
                    )

        return df

    def add_momentum_indicators(
        self, data: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """添加动量类指标."""
        df = data.copy()

        # RSI (多个周期)
        for period in [7, 14, 21]:
            col = f"rsi_{period}"
            if col not in df.columns:
                rsi = talib.RSI(df["close"].values, timeperiod=period)
                rsi = pd.Series(rsi, index=df.index).replace([np.inf, -np.inf], np.nan)
                df[col] = rsi

        # 随机指标 (Stochastic)
        df["stoch_k"], df["stoch_d"] = talib.STOCH(
            df["high"].values, df["low"].values, df["close"].values
        )
        df["stochf_k"], df["stochf_d"] = talib.STOCHF(
            df["high"].values, df["low"].values, df["close"].values
        )
        df["stochrsi_k"], df["stochrsi_d"] = talib.STOCHRSI(
            df["close"].values, timeperiod=14
        )

        # 威廉指标 (Williams %R)
        df["willr"] = talib.WILLR(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=14
        )

        # 动量指标
        for period in [5, 10, 14]:
            df[f"mom_{period}"] = talib.MOM(df["close"].values, timeperiod=period)

        # 变化率 (ROC)
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = talib.ROC(df["close"].values, timeperiod=period)

        # 商品通道指数 (CCI)
        df["cci"] = talib.CCI(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=14
        )

        # 终极指标 (Ultimate Oscillator)
        df["ultosc"] = talib.ULTOSC(
            df["high"].values, df["low"].values, df["close"].values
        )

        # 真实强度指数 (TSI) - 使用自定义实现，因为TA-Lib没有TSI
        def compute_tsi(series, period1=14, period2=25):
            """计算真实强度指数 (TSI)."""
            momentum = series.diff()
            smoothed_momentum = (
                momentum.ewm(span=period1).mean().ewm(span=period2).mean()
            )
            smoothed_abs_momentum = (
                momentum.abs().ewm(span=period1).mean().ewm(span=period2).mean()
            )
            tsi = 100 * smoothed_momentum / smoothed_abs_momentum
            return tsi

        df["tsi"] = compute_tsi(df["close"])

        return df

    def add_volatility_indicators(
        self, data: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """添加波动率类指标."""
        df = data.copy()

        # 布林带
        if {"bb_upper", "bb_middle", "bb_lower"}.difference(df.columns):
            upper, middle, lower = talib.BBANDS(
                df["close"].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0,
            )
            if "bb_upper" not in df.columns:
                df["bb_upper"] = upper
            if "bb_middle" not in df.columns:
                df["bb_middle"] = middle
            if "bb_lower" not in df.columns:
                df["bb_lower"] = lower

        # 布林带宽度和位置
        if "bb_width" not in df.columns:
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        if "bb_position" not in df.columns:
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            )

        # 平均真实波幅 (ATR)
        for period in [7, 14, 21]:
            col = f"atr_{period}"
            if col not in df.columns:
                df[col] = talib.ATR(
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                    timeperiod=period,
                )

        # 真实波幅 (TRANGE)
        # 使用 shift(1) 确保时间对齐，避免使用未来信息
        if "trange" not in df.columns:
            trange_vals = talib.TRANGE(
                df["high"].values, df["low"].values, df["close"].values
            )
            df["trange"] = pd.Series(trange_vals, index=df.index).shift(1)

        # 平均方向指数 (ADX) - 也用于波动率
        if "natr" not in df.columns:
            df["natr"] = talib.NATR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )

        # 历史波动率
        returns = df["close"].pct_change()
        for period in [10, 20, 30]:
            col = f"volatility_{period}"
            if col not in df.columns:
                df[col] = returns.rolling(window=period).std()

        return df

    def add_volume_indicators(
        self, data: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """添加成交量类指标."""
        df = data.copy()

        # 成交量移动平均
        for period in [5, 10, 20]:
            col = f"volume_sma_{period}"
            if col not in df.columns:
                df[col] = talib.SMA(df["volume"].values, timeperiod=period)

        # 成交量比率
        if "volume_ratio" not in df.columns:
            df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # 平衡成交量 (OBV)
        df["obv"] = talib.OBV(df["close"].values, df["volume"].values)

        # 成交量加权平均价格 (VWAP) - 简化版
        df["vwap"] = (df["close"] * df["volume"]).rolling(window=20).sum() / df[
            "volume"
        ].rolling(window=20).sum()

        # 累积/派发线 (A/D Line)
        df["ad"] = talib.AD(
            df["high"].values, df["low"].values, df["close"].values, df["volume"].values
        )

        # 柴金资金流量 (CMF)
        df["cmf"] = talib.ADOSC(
            df["high"].values, df["low"].values, df["close"].values, df["volume"].values
        )

        # 成交量价格趋势 (VPT) - 使用自定义实现，因为TA-Lib没有VPT
        def compute_vpt(close, volume):
            """计算成交量价格趋势 (VPT)."""
            price_change = close.pct_change()
            vpt = (price_change * volume).cumsum()
            return vpt

        df["vpt"] = compute_vpt(df["close"], df["volume"])

        return df

    def add_pattern_indicators(
        self, data: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """添加形态识别指标，如果指定了required_features，只计算需要的特征."""
        df = data.copy()

        # 定义所有可用的蜡烛图形态
        pattern_functions = {
            "cdl_doji": talib.CDLDOJI,
            "cdl_hammer": talib.CDLHAMMER,
            "cdl_hanging_man": talib.CDLHANGINGMAN,
            "cdl_engulfing": talib.CDLENGULFING,
            "cdl_harami": talib.CDLHARAMI,
            "cdl_doji_star": talib.CDLDOJISTAR,
            "cdl_shooting_star": talib.CDLSHOOTINGSTAR,
            "cdl_3blackcrows": talib.CDL3BLACKCROWS,
            "cdl_3whitesoldiers": talib.CDL3WHITESOLDIERS,
            "cdl_3inside": talib.CDL3INSIDE,
            "cdl_3outside": talib.CDL3OUTSIDE,
            "cdl_3linestrike": talib.CDL3LINESTRIKE,
            "cdl_abandonedbaby": talib.CDLABANDONEDBABY,
            "cdl_advanceblock": talib.CDLADVANCEBLOCK,
            "cdl_belthold": talib.CDLBELTHOLD,
            "cdl_breakaway": talib.CDLBREAKAWAY,
            "cdl_closingmarubozu": talib.CDLCLOSINGMARUBOZU,
            "cdl_concealbabyswall": talib.CDLCONCEALBABYSWALL,
            "cdl_counterattack": talib.CDLCOUNTERATTACK,
            "cdl_darkcloudcover": talib.CDLDARKCLOUDCOVER,
            "cdl_dragonflydoji": talib.CDLDRAGONFLYDOJI,
            "cdl_eveningdojistar": talib.CDLEVENINGDOJISTAR,
            "cdl_eveningstar": talib.CDLEVENINGSTAR,
            "cdl_gapsidesidewhite": talib.CDLGAPSIDESIDEWHITE,
            "cdl_gravestonedoji": talib.CDLGRAVESTONEDOJI,
            "cdl_identical3crows": talib.CDLIDENTICAL3CROWS,
            "cdl_inneck": talib.CDLINNECK,
            "cdl_invertedhammer": talib.CDLINVERTEDHAMMER,
            "cdl_kicking": talib.CDLKICKING,
            "cdl_kickingbylength": talib.CDLKICKINGBYLENGTH,
            "cdl_ladderbottom": talib.CDLLADDERBOTTOM,
            "cdl_longleggeddoji": talib.CDLLONGLEGGEDDOJI,
            "cdl_longline": talib.CDLLONGLINE,
            "cdl_marubozu": talib.CDLMARUBOZU,
            "cdl_matchinglow": talib.CDLMATCHINGLOW,
            "cdl_mathold": talib.CDLMATHOLD,
            "cdl_morningdojistar": talib.CDLMORNINGDOJISTAR,
            "cdl_morningstar": talib.CDLMORNINGSTAR,
            "cdl_onneck": talib.CDLONNECK,
            "cdl_piercing": talib.CDLPIERCING,
            "cdl_rickshawman": talib.CDLRICKSHAWMAN,
            "cdl_risefall3methods": talib.CDLRISEFALL3METHODS,
            "cdl_separatinglines": talib.CDLSEPARATINGLINES,
            "cdl_shortline": talib.CDLSHORTLINE,
            "cdl_spinningtop": talib.CDLSPINNINGTOP,
            "cdl_stalledpattern": talib.CDLSTALLEDPATTERN,
            "cdl_sticksandwich": talib.CDLSTICKSANDWICH,
            "cdl_takuri": talib.CDLTAKURI,
            "cdl_tasukigap": talib.CDLTASUKIGAP,
            "cdl_thrusting": talib.CDLTHRUSTING,
            "cdl_tristar": talib.CDLTRISTAR,
            "cdl_unique3river": talib.CDLUNIQUE3RIVER,
            "cdl_upsidegap2crows": talib.CDLUPSIDEGAP2CROWS,
            "cdl_xsidegap3methods": talib.CDLXSIDEGAP3METHODS,
        }

        # 如果指定了required_features，只计算需要的特征
        if required_features:
            # 找出所有需要的cdl_*特征
            needed_patterns = [
                p for p in pattern_functions.keys() if p in required_features
            ]
            for pattern_name in needed_patterns:
                if pattern_name not in df.columns:
                    try:
                        df[pattern_name] = pattern_functions[pattern_name](
                            df["open"].values,
                            df["high"].values,
                            df["low"].values,
                            df["close"].values,
                        )
                    except Exception as e:
                        print(f"Warning: Error computing {pattern_name}: {e}")
        else:
            # 计算所有特征
            for pattern_name, pattern_func in pattern_functions.items():
                if pattern_name not in df.columns:
                    try:
                        df[pattern_name] = pattern_func(
                            df["open"].values,
                            df["high"].values,
                            df["low"].values,
                            df["close"].values,
                        )
                    except Exception as e:
                        print(f"Warning: Error computing {pattern_name}: {e}")

        return df

    def add_math_indicators(
        self, data: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """添加数学运算指标."""
        df = data.copy()

        # 数学运算
        df["add"] = talib.ADD(df["high"].values, df["low"].values)
        df["div"] = talib.DIV(df["high"].values, df["low"].values)
        df["max"] = talib.MAX(df["close"].values, timeperiod=14)
        df["min"] = talib.MIN(df["close"].values, timeperiod=14)
        df["maxindex"] = talib.MAXINDEX(df["close"].values, timeperiod=14)
        df["minindex"] = talib.MININDEX(df["close"].values, timeperiod=14)

        # 统计指标
        # 使用 shift(1) 确保时间对齐，避免使用未来信息
        # stddev 和 var 是滚动窗口统计，本身只使用历史数据，但为了安全起见添加 shift(1)
        df["stddev"] = talib.STDDEV(df["close"].values, timeperiod=14).shift(1)
        df["var"] = talib.VAR(df["close"].values, timeperiod=14).shift(1)

        return df

    def add_macd_variants(
        self, data: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """添加MACD变体指标."""
        df = data.copy()

        # 标准MACD
        if {"macd", "macd_signal", "macd_hist"}.difference(df.columns):
            macd, signal, hist = talib.MACD(df["close"].values)
            if "macd" not in df.columns:
                df["macd"] = macd
            if "macd_signal" not in df.columns:
                df["macd_signal"] = signal
            if "macd_hist" not in df.columns:
                df["macd_hist"] = hist

        # MACD扩展
        df["macd_ext"], df["macd_ext_signal"], df["macd_ext_hist"] = talib.MACDEXT(
            df["close"].values, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # MACD固定
        df["macd_fix"], df["macd_fix_signal"], df["macd_fix_hist"] = talib.MACDFIX(
            df["close"].values, signalperiod=9
        )

        return df

    def add_technical_indicators(
        self, data: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """添加TA-Lib技术指标，如果指定了required_features，只计算需要的特征."""
        if data.empty:
            return data

        df = data.copy()

        # 确保数据是数值类型并转换为double
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

        # 移除包含NaN的行
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        if df.empty:
            return df

        try:
            # 如果指定了required_features，分析需要哪些类别的特征
            need_trend = need_momentum = need_volatility = need_volume = (
                need_pattern
            ) = need_math = need_macd = need_derived = True
            if required_features:
                # 分析需要的特征类别
                trend_features = {
                    "sma_",
                    "ema_",
                    "wma_",
                    "tema_",
                    "kama_",
                    "dema_",
                    "trima_",
                    "adx",
                    "aroon",
                    "cci",
                    "dx",
                    "minus_di",
                    "plus_di",
                    "minus_dm",
                    "plus_dm",
                }
                momentum_features = {
                    "rsi_",
                    "stoch",
                    "willr",
                    "mom_",
                    "roc",
                    "ppo",
                    "trix",
                    "ultosc",
                    "cmo",
                    "cci",
                }
                volatility_features = {"atr", "natr", "trange", "bb_", "bollinger"}
                volume_features = {"ad", "adosc", "obv", "adx"}
                pattern_features = {"cdl_"}
                math_features = {"math_", "max_", "min_", "sum_", "stddev", "var"}
                macd_features = {"macd", "macd_signal", "macd_hist"}
                derived_features = {
                    "price_change",
                    "high_low_ratio",
                    "close_open_ratio",
                    "sma_",
                    "_ratio",
                    "rsi_",
                    "_normalized",
                    "_diff",
                    "macd_normalized",
                    "macd_signal_ratio",
                }

                need_trend = any(
                    any(f.startswith(p) or p in f for p in trend_features)
                    for f in required_features
                )
                need_momentum = any(
                    any(f.startswith(p) or p in f for p in momentum_features)
                    for f in required_features
                )
                need_volatility = any(
                    any(f.startswith(p) or p in f for p in volatility_features)
                    for f in required_features
                )
                need_volume = any(
                    any(f.startswith(p) or p in f for p in volume_features)
                    for f in required_features
                )
                need_pattern = any(
                    any(f.startswith(p) or p in f for p in pattern_features)
                    for f in required_features
                )
                need_math = any(
                    any(f.startswith(p) or p in f for p in math_features)
                    for f in required_features
                )
                need_macd = any(
                    any(f.startswith(p) or p in f for p in macd_features)
                    for f in required_features
                )
                need_derived = any(
                    any(f.startswith(p) or p in f for p in derived_features)
                    for f in required_features
                )

            # 添加各类指标，每个类别独立处理以避免单个失败影响整体
            if need_trend:
                try:
                    df = self.add_trend_indicators(df, required_features)
                except Exception as e:
                    print(f"Warning: Error in trend indicators: {e}")

            if need_momentum:
                try:
                    df = self.add_momentum_indicators(df, required_features)
                except Exception as e:
                    print(f"Warning: Error in momentum indicators: {e}")

            if need_volatility:
                try:
                    df = self.add_volatility_indicators(df, required_features)
                except Exception as e:
                    print(f"Warning: Error in volatility indicators: {e}")

            if need_volume:
                try:
                    df = self.add_volume_indicators(df, required_features)
                except Exception as e:
                    print(f"Warning: Error in volume indicators: {e}")

            if need_pattern:
                try:
                    df = self.add_pattern_indicators(df, required_features)
                except Exception as e:
                    print(f"Warning: Error in pattern indicators: {e}")

            if need_math:
                try:
                    df = self.add_math_indicators(df, required_features)
                except Exception as e:
                    print(f"Warning: Error in math indicators: {e}")

            if need_macd:
                try:
                    df = self.add_macd_variants(df, required_features)
                except Exception as e:
                    print(f"Warning: Error in MACD indicators: {e}")

            # 添加价格衍生特征（只在需要时）
            if need_derived:
                if not required_features or any(
                    f in required_features
                    for f in ["price_change", "high_low_ratio", "close_open_ratio"]
                ):
                    if not required_features or "price_change" in required_features:
                        df["price_change"] = df["close"].pct_change()
                    if not required_features or "high_low_ratio" in required_features:
                        df["high_low_ratio"] = df["high"] / df["low"]
                    if not required_features or "close_open_ratio" in required_features:
                        df["close_open_ratio"] = df["close"] / df["open"]

                # 添加移动平均比率（只在需要时）
                if not required_features or any(
                    f in required_features
                    for f in ["sma_5_20_ratio", "sma_10_50_ratio", "ema_5_20_ratio"]
                ):
                    if not required_features or "sma_5_20_ratio" in required_features:
                        if "sma_5" in df.columns and "sma_20" in df.columns:
                            df["sma_5_20_ratio"] = df["sma_5"] / df["sma_20"]
                    if not required_features or "sma_10_50_ratio" in required_features:
                        if "sma_10" in df.columns and "sma_50" in df.columns:
                            df["sma_10_50_ratio"] = df["sma_10"] / df["sma_50"]
                    if not required_features or "ema_5_20_ratio" in required_features:
                        if "ema_5" in df.columns and "ema_20" in df.columns:
                            df["ema_5_20_ratio"] = df["ema_5"] / df["ema_20"]

                # 添加RSI衍生特征（只在需要时）
                if not required_features or any(
                    f in required_features
                    for f in ["rsi_14_normalized", "rsi_7_14_diff"]
                ):
                    if (
                        not required_features
                        or "rsi_14_normalized" in required_features
                    ):
                        if "rsi_14" in df.columns:
                            df["rsi_14_normalized"] = (df["rsi_14"] - 50) / 50
                    if not required_features or "rsi_7_14_diff" in required_features:
                        if "rsi_7" in df.columns and "rsi_14" in df.columns:
                            df["rsi_7_14_diff"] = df["rsi_7"] - df["rsi_14"]

                # 添加MACD衍生特征（只在需要时）
                if not required_features or any(
                    f in required_features
                    for f in ["macd_normalized", "macd_signal_ratio"]
                ):
                    if not required_features or "macd_normalized" in required_features:
                        if "macd" in df.columns and "close" in df.columns:
                            df["macd_normalized"] = df["macd"] / df["close"]
                    if (
                        not required_features
                        or "macd_signal_ratio" in required_features
                    ):
                        if "macd_signal" in df.columns and "macd" in df.columns:
                            df["macd_signal_ratio"] = df["macd_signal"] / df["macd"]

        except Exception as e:
            print(f"Warning: Error computing TA-Lib indicators: {e}")
            # 如果出错，返回原始数据
            return data

        # 填充NaN值
        feature_cols = [
            col
            for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        for col in feature_cols:
            df[col] = df[col].fillna(0)

        return df

    def normalize_features(
        self, data: pd.DataFrame, timeframe: str, fit: bool = True
    ) -> pd.DataFrame:
        """
        Improved feature normalization with feature grouping and outlier handling.

        Args:
            data: DataFrame with features
            timeframe: Timeframe identifier
            fit: Whether to fit the scaler (True for training, False for prediction)

        Returns:
            DataFrame with normalized features
        """
        df = data.copy()

        # Get feature columns (exclude OHLCV)
        feature_cols = [
            col
            for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        if not feature_cols:
            return df

        # 1. 预处理: 移除常数特征和异常值
        valid_features = []
        for col in feature_cols:
            values = df[col].dropna()
            if len(values) > 0:
                # 检查是否为常数特征
                if values.std() < 1e-10:
                    print(f"Warning: Removing constant feature: {col}")
                    continue

                # 检查异常值比例
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outliers = values[(values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)]
                    outlier_ratio = len(outliers) / len(values)
                    if outlier_ratio > 0.5:  # 超过50%的异常值，跳过该特征
                        print(
                            f"Warning: Skipping feature {col} due to too many outliers ({outlier_ratio:.1%})"
                        )
                        continue

                valid_features.append(col)

        if not valid_features:
            print("Warning: No valid features found after preprocessing")
            return df

        # 2. 特征分组 - 根据特征类型选择不同的归一化策略
        price_features = [
            col
            for col in valid_features
            if any(x in col for x in ["sma", "ema", "wma", "tema", "kama", "sar"])
        ]
        ratio_features = [
            col
            for col in valid_features
            if "ratio" in col or "position" in col or "normalized" in col
        ]
        volatility_features = [
            col
            for col in valid_features
            if any(x in col for x in ["atr", "natr", "trange", "bb_", "volatility"])
        ]
        volume_features = [
            col
            for col in valid_features
            if any(x in col for x in ["volume", "obv", "ad", "vpt", "cmf"])
        ]
        momentum_features = [
            col
            for col in valid_features
            if any(
                x in col
                for x in ["rsi", "stoch", "willr", "mom", "roc", "cci", "ultosc", "tsi"]
            )
        ]
        index_features = [
            col
            for col in valid_features
            if any(x in col for x in ["maxindex", "minindex", "max", "min"])
        ]
        other_features = [
            col
            for col in valid_features
            if col
            not in price_features
            + ratio_features
            + volatility_features
            + volume_features
            + momentum_features
            + index_features
        ]

        # 3. 分组归一化
        feature_groups = {
            "price": (price_features, self.scaler_class()),
            "ratio": (ratio_features, MinMaxScaler()),
            "volatility": (volatility_features, RobustScaler()),
            "volume": (volume_features, self.scaler_class()),
            "momentum": (momentum_features, MinMaxScaler()),
            "index": (index_features, MinMaxScaler()),
            "other": (other_features, self.scaler_class()),
        }

        # 初始化group_scalers
        if not hasattr(self, "group_scalers"):
            self.group_scalers = {}

        for group_name, (group_features, scaler) in feature_groups.items():
            if not group_features:
                continue

            # 准备数据
            X = df[group_features].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # 对于成交量特征，先进行对数变换
            if group_name == "volume":
                X = np.log1p(np.abs(X)) * np.sign(X)  # log1p处理负值

            if fit:
                X_scaled = scaler.fit_transform(X)
                # 保存scaler
                self.group_scalers[f"{timeframe}_{group_name}"] = scaler
            else:
                # 使用已保存的scaler
                if f"{timeframe}_{group_name}" not in self.group_scalers:
                    raise ValueError(f"No scaler found for {timeframe}_{group_name}")
                scaler = self.group_scalers[f"{timeframe}_{group_name}"]
                X_scaled = scaler.transform(X)

            # 更新DataFrame
            for i, col in enumerate(group_features):
                df[col] = X_scaled[:, i]

        # 4. 存储特征统计信息
        if fit:
            all_features = []
            for group_features, _ in feature_groups.values():
                all_features.extend(group_features)

            if all_features:
                X_all = df[all_features].values
                self.feature_stats[timeframe] = {
                    "mean": np.mean(X_all, axis=0),
                    "std": np.std(X_all, axis=0),
                    "min": np.min(X_all, axis=0),
                    "max": np.max(X_all, axis=0),
                    "feature_groups": {
                        name: features
                        for name, (features, _) in feature_groups.items()
                        if features
                    },
                }

        return df

    def engineer_features(
        self, multi_tf_data: Dict[str, pd.DataFrame], fit: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for multi-timeframe data with normalization.

        Args:
            multi_tf_data: Dictionary mapping timeframe to DataFrame
            fit: Whether to fit scalers (True for training, False for prediction)

        Returns:
            Dictionary with engineered and normalized features for each timeframe
        """
        engineered_data = {}

        for timeframe, data in multi_tf_data.items():
            print(f"Engineering TA-Lib features for {timeframe}: {data.shape}")

            # Add technical indicators
            df_with_indicators = self.add_technical_indicators(data)
            print(
                f"Added TA-Lib indicators for {timeframe}: {df_with_indicators.shape}"
            )

            # Normalize features
            df_normalized = self.normalize_features(
                df_with_indicators, timeframe, fit=fit
            )
            print(f"Normalized features for {timeframe}: {df_normalized.shape}")

            engineered_data[timeframe] = df_normalized

        return engineered_data

    def save_scalers(self, filepath: str):
        """Save fitted scalers to file."""
        scaler_data = {
            "scalers": getattr(self, "scalers", {}),
            "group_scalers": getattr(self, "group_scalers", {}),
            "feature_stats": self.feature_stats,
            "scaler_type": self.scaler_type,
            "removed_features": getattr(self, "removed_features", {}),
        }

        with open(filepath, "wb") as f:
            pickle.dump(scaler_data, f)

        print(f"TA-Lib scalers saved to {filepath}")

    def load_scalers(self, filepath: str):
        """Load fitted scalers from file."""
        with open(filepath, "rb") as f:
            scaler_data = pickle.load(f)

        self.scalers = scaler_data.get("scalers", {})
        self.group_scalers = scaler_data.get("group_scalers", {})
        self.feature_stats = scaler_data.get("feature_stats", {})
        self.scaler_type = scaler_data.get("scaler_type", "standard")
        self.removed_features = scaler_data.get("removed_features", {})

        print(f"TA-Lib scalers loaded from {filepath}")

    def get_feature_importance_info(self, timeframe: str) -> Dict:
        """Get feature statistics for analysis."""
        if timeframe not in self.feature_stats:
            return {}

        stats = self.feature_stats[timeframe]
        return {
            "mean": stats["mean"].tolist(),
            "std": stats["std"].tolist(),
            "min": stats["min"].tolist(),
            "max": stats["max"].tolist(),
            "scaler_type": self.scaler_type,
        }

    def get_available_indicators(self) -> List[str]:
        """Get list of all available TA-Lib indicators."""
        return talib.get_functions()

    def get_feature_count(self, data: pd.DataFrame) -> int:
        """Get the number of features after adding indicators."""
        df_with_indicators = self.add_technical_indicators(data)
        feature_cols = [
            col
            for col in df_with_indicators.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        return len(feature_cols)
