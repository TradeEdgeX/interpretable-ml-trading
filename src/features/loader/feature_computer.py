"""
特征计算器（顺序 + 缓存）

当前实现**强制顺序执行**特征计算，并依赖：
- 内存缓存（同一 DataFrame 签名内复用）
- 磁盘缓存 / monthly 缓存（跨运行复用，支持按月增量）

说明：
- 文件名已从 `parallel_computer.py` 重命名为 `feature_computer.py`（旧并行实现已移除，现为顺序执行）。
- 为了减少复杂度与大 DataFrame 序列化/复制风险，已移除多进程/多线程执行路径。
"""

import multiprocessing as mp
from functools import lru_cache
import hashlib
import os
import pickle
import os
import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any
import pandas as pd
import numpy as np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.features.registry import get_compute_func


def analyze_dependency_levels(
    features: Dict, requested_features: List[str]
) -> Dict[int, List[str]]:
    """
    分析特征依赖层级

    Args:
        features: 特征配置字典
        requested_features: 请求的特征列表

    Returns:
        levels: {level: [feature_names]}
    """
    # 1. 收集所有需要的特征（包括依赖）
    all_needed = set(requested_features)
    queue = list(requested_features)

    while queue:
        feat_name = queue.pop(0)
        if feat_name not in features:
            continue
        deps = features[feat_name].get("dependencies", [])
        for dep in deps:
            if dep not in all_needed:
                all_needed.add(dep)
                queue.append(dep)

    # 2. 计算层级（使用 memo + recursion stack，避免把“已处理集合”误当成循环检测）
    levels: Dict[int, List[str]] = {}
    memo: Dict[str, int] = {}
    visiting: set[str] = set()

    def get_level(feat_name: str) -> int:
        if feat_name in memo:
            return memo[feat_name]
        if feat_name in visiting:
            raise ValueError(f"Circular dependency detected at: {feat_name}")
        visiting.add(feat_name)
        if feat_name not in features:
            lvl = 0
        else:
            deps = features[feat_name].get("dependencies", []) or []
            if not deps:
                lvl = 0
            else:
                lvl = max(get_level(dep) for dep in deps) + 1
        visiting.remove(feat_name)
        memo[feat_name] = lvl
        return lvl

    for feat_name in all_needed:
        level = get_level(feat_name)
        levels.setdefault(level, []).append(feat_name)

    return levels


def _build_call_args(
    feature_info: Dict, df: pd.DataFrame, feature_name: str = "unknown"
) -> Tuple[tuple, dict]:
    """
    构建特征计算函数的调用参数

    Args:
        feature_info: 特征配置信息
        df: 输入 DataFrame
        feature_name: 特征名称

    Returns:
        (args, kwargs): 调用参数元组和关键字参数字典
    """
    compute_params = feature_info.get("compute_params", {}) or {}
    column_mappings = feature_info.get("column_mappings", {}) or {}

    # 获取 compute_func 以检查函数签名
    from src.features.registry import get_compute_func
    compute_func_name = feature_info.get("compute_func", feature_name)
    func_sig = None
    try:
        compute_func = get_compute_func(compute_func_name)
        import inspect
        func_sig = inspect.signature(compute_func)
        first_param = list(func_sig.parameters.values())[0] if func_sig.parameters else None
        # 如果第一个参数是 'df' 且没有显式配置 positional_params，自动添加
        auto_df_positional = (
            first_param is not None
            and first_param.name == "df"
            and first_param.kind != inspect.Parameter.KEYWORD_ONLY
            and "positional_params" not in feature_info
        )
    except Exception:
        auto_df_positional = False
        compute_func = None

    # 构建位置参数（按顺序）
    args = []
    positional_params = feature_info.get("positional_params", [])
    if auto_df_positional:
        # 自动检测：如果函数第一个参数是 df，且没有显式配置，自动添加
        args.append(df)
    else:
        for param_name in positional_params:
            if param_name == "df":
                args.append(df)
            elif param_name in df.columns:
                args.append(df[param_name])
            elif param_name in compute_params:
                args.append(compute_params[param_name])
            else:
                raise KeyError(
                    f"Column '{param_name}' required for parameter '{param_name}' not found in DataFrame"
                )

    # 构建关键字参数（来自 compute_params）
    # 排除非函数参数的配置项：
    # - ticks_dir/lookback_minutes: only used to auto-generate ticks_loader_json
    # - normalized/output_normalization/output_normalization_map: repo normalization contract metadata
    # - node_cache_version: caching metadata (not compute function argument)
    excluded_params = {
        "ticks_dir",
        "lookback_minutes",
        "normalized",
        "output_normalization",
        "output_normalization_map",
        "node_cache_version",
    }
    kwargs: Dict[str, Any] = {}
    for param_name, param_value in compute_params.items():
        if param_name in excluded_params:
            # 跳过这些配置项，它们不是函数参数
            continue
        if param_name not in feature_info.get("positional_params", []):
            # 对于 compute_params，我们保持保守策略：大多数字符串参数是枚举/配置，
            # 而不是列名（如 price_col="close"），真正的列映射由 column_mappings 处理。
            # 只有当参数名本身就是列名时，才从 DataFrame 提取对应列。
            if isinstance(param_value, str) and param_name in df.columns:
                kwargs[param_name] = df[param_name]
            else:
                kwargs[param_name] = param_value

    # 自动生成 ticks_loader_json（如果函数需要 ticks_loader_json/ticks，且未显式提供）
    try:
        needs_ticks_loader = False
        if func_sig is not None:
            needs_ticks_loader = (
                "ticks_loader_json" in func_sig.parameters
                or "ticks" in func_sig.parameters
            )
        if needs_ticks_loader:
            tlj = compute_params.get("ticks_loader_json")
            if tlj is None and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                # 尝试自动生成：调用 list_tick_files 获取 tick 文件列表，再序列化
                from src.data_tools.tick_loader import (
                    list_tick_files,
                    serialize_tick_loader_params,
                )

                ticks_dir = compute_params.get("ticks_dir", "data/parquet_data")
                lookback_minutes = compute_params.get("lookback_minutes", 60)
                start_ts = str(df.index.min())
                end_ts = str(df.index.max())

                symbol = None
                if "_symbol" in df.columns:
                    sym_series = df["_symbol"].dropna()
                    if len(sym_series) > 0:
                        symbol = str(sym_series.iloc[0])
                elif "symbol" in df.columns:
                    sym_series = df["symbol"].dropna()
                    if len(sym_series) > 0:
                        symbol = str(sym_series.iloc[0])

                try:
                    tick_files = list_tick_files(
                        symbol=symbol or "",
                        start_ts=start_ts,
                        end_ts=end_ts,
                        ticks_dir=ticks_dir,
                        lookback_minutes=lookback_minutes,
                    )
                    if tick_files:
                        tlj_obj = {
                            "symbol": symbol or "",
                            "tick_files": tick_files,
                            "start_ts": start_ts,
                            "end_ts": end_ts,
                            "lookback_minutes": lookback_minutes,
                        }
                        tlj = serialize_tick_loader_params(tlj_obj)
                except Exception:
                    tlj = None

            if tlj is not None:
                kwargs["ticks_loader_json"] = tlj
    except Exception:
        pass

    # 处理 column_mappings：将 DataFrame 指定列注入到函数参数
    for param_name, source in column_mappings.items():
        if isinstance(source, str):
            col_name = source
            if col_name not in df.columns:
                raise KeyError(
                    f"Column '{col_name}' required for parameter '{param_name}' not "
                    f"found in DataFrame when building call args for feature "
                    f"'{feature_name}'"
                )
            kwargs[param_name] = df[col_name]
        elif isinstance(source, list):
            missing = [col for col in source if col not in df.columns]
            if missing:
                raise KeyError(
                    f"Columns {missing} required for parameter '{param_name}' not "
                    f"found in DataFrame when building call args for feature "
                    f"'{feature_name}'"
                )
            # 对于多列映射，按列名子 DataFrame 传入
            kwargs[param_name] = df[source]
        else:
            raise ValueError(
                f"Unsupported column mapping type for parameter '{param_name}': "
                f"{type(source)}"
            )

    # Auto-wire required_columns into kwargs when there are no explicit column_mappings
    # and the node is not passing the full df. This is primarily used by selector
    # nodes like `select_columns_from_series` which accept **series_kwargs.
    if not column_mappings and not feature_info.get("pass_full_df", False):
        required_columns = feature_info.get("required_columns", []) or []
        for col in required_columns:
            if col in df.columns and col not in kwargs:
                kwargs[col] = df[col]

    # Keep args as a list for easier inspection in tests; callers can still use *args.
    return (args, kwargs)


def _get_monthly_cache_key(
    feature_name: str,
    month_key: str,
    cache_version: str,
    fit: bool,
    compute_params: Dict,
    *,
    data_id: str = "unknown",
) -> str:
    """
    生成月度缓存键

    Args:
        feature_name: 特征名称
        month_key: 月份键（如 "2024-01"）
        df_sig: DataFrame 签名（用于区分不同的数据集）
        cache_version: 缓存版本
        fit: 是否拟合
        compute_params: 计算参数

    Returns:
        cache_key: 缓存键字符串
    """
    # Build a stable param signature.
    # IMPORTANT: strip volatile runtime params (e.g. ticks_loader_json) that would
    # destroy cache hit-rate across processes/runs for the same month.
    volatile_keys = {
        "ticks_loader_json",
        "ticks_dir",
        "lookback_minutes",
    }
    stable_items = []
    try:
        for k, v in sorted((compute_params or {}).items()):
            if k in volatile_keys:
                continue
            stable_items.append((k, v))
    except Exception:
        stable_items = []
    params_sig = hashlib.md5(str(stable_items).encode()).hexdigest()[:8]

    # 构建缓存键
    key_parts = [
        feature_name,
        month_key,
        data_id,
        cache_version,
        "fit" if fit else "predict",
        params_sig,
    ]
    cache_key = "_".join(key_parts)
    return cache_key


def _save_monthly_cache(
    cache_dir: Path, cache_key: str, result: pd.DataFrame | pd.Series
) -> None:
    """保存月度缓存"""
    cache_file = cache_dir / f"{cache_key}.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)


def _load_monthly_cache(cache_dir: Path, cache_key: str) -> Optional[pd.DataFrame | pd.Series]:
    """加载月度缓存"""
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            # Corrupted cache (often caused by an interrupted write). Treat as cache-miss.
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass
            return None
    return None


def _compute_single_feature_worker_monthly(
    feature_name: str,
    feature_info: Dict,
    df_bytes: bytes,
    fit: bool,
    monthly_cache_dir: Optional[str],
    ticks_loader_json: Optional[str] = None,
) -> Tuple[str, bytes]:
    """
    Monthly parallel worker (opt-in).

    This is intentionally scoped to *per-month slices* to keep payloads small and avoid
    cross-process contention. It is only used when FEATURE_MONTHLY_WORKERS>1.
    """
    df = pickle.loads(df_bytes)
    compute_func_name = (feature_info.get("compute_func") or "").strip()
    if not compute_func_name:
        raise ValueError(f"feature_info.compute_func missing for {feature_name}")
    compute_func = get_compute_func(compute_func_name)

    # Build call args/kwargs (auto-wires ticks_loader_json when needed).
    call_args, call_kwargs = _build_call_args(feature_info, df, feature_name)

    # Best-effort: allow caller to override ticks_loader_json (rare; mostly for debugging).
    if ticks_loader_json is not None:
        call_kwargs["ticks_loader_json"] = ticks_loader_json

    # Filter kwargs by signature unless compute_func accepts **kwargs
    try:
        import inspect

        sig = inspect.signature(compute_func)
        params = sig.parameters
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if not accepts_var_kw and call_kwargs:
            allowed = set(params.keys())
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in allowed}
    except Exception:
        pass

    month_result = compute_func(*call_args, **call_kwargs)

    # Strictly keep only output_columns (when present)
    output_cols = feature_info.get("output_columns", [feature_name]) or [feature_name]
    if isinstance(month_result, tuple):
        if len(month_result) == len(output_cols):
            month_result = pd.DataFrame(
                {col: series for col, series in zip(output_cols, month_result)}
            )
        else:
            month_result = pd.DataFrame(
                {f"{feature_name}_{i}": series for i, series in enumerate(month_result)}
            )
    if isinstance(month_result, pd.DataFrame):
        if month_result.columns.duplicated().any():
            month_result = month_result.loc[:, ~month_result.columns.duplicated()]
        existing = [c for c in output_cols if c in month_result.columns]
        if existing:
            month_result = month_result[existing]
        else:
            month_result = pd.DataFrame(index=month_result.index)
    elif isinstance(month_result, pd.Series):
        series_name = month_result.name if month_result.name else feature_name
        if series_name not in output_cols:
            month_result = pd.DataFrame(index=month_result.index)

    return feature_name, pickle.dumps(month_result, protocol=pickle.HIGHEST_PROTOCOL)


class FeatureComputer:
    """
    特征计算器（顺序 + 缓存）

    当前实现**强制顺序执行**特征计算，并依赖：
    - 内存缓存（同一 DataFrame 签名内复用）
    - 磁盘缓存 / monthly 缓存（跨运行复用，支持按月增量）

    说明：
    - 为了减少复杂度与大 DataFrame 序列化/复制风险，已移除多进程/多线程执行路径。
    - 性能主要依赖磁盘/月度缓存，内存缓存作为补充加速。
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_disk_cache: bool = True,
        use_memory_cache: bool = True,
        max_workers: Optional[int] = None,
        parallel_backend: str = "process",  # deprecated (sequential-only)
        use_monthly_cache: bool = True,  # 是否使用按月缓存
        enable_parallel: bool = False,  # deprecated (sequential-only)
        monthly_warmup_months: Optional[int] = None,
        verbose: bool = True,  # 研究=True（打印详细），实盘=False（只打印异常）
    ):
        """
        Args:
            cache_dir: 磁盘缓存目录
            use_disk_cache: 是否使用磁盘缓存
            use_memory_cache: 是否使用内存缓存
            max_workers: 最大并行数（None 表示使用 CPU 核心数）
            parallel_backend: 并行后端（process/thread）
            use_monthly_cache: 是否使用按月缓存（增量计算）
            verbose: 是否打印详细日志（研究True，实盘False）
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_disk_cache = use_disk_cache
        self.use_memory_cache = use_memory_cache
        self.use_monthly_cache = use_monthly_cache
        self.verbose = verbose
        try:
            env_warmup = int(os.getenv("FEATURE_MONTHLY_WARMUP_MONTHS", "3"))
        except Exception:
            env_warmup = 3
        if monthly_warmup_months is None:
            monthly_warmup_months = env_warmup
        self.monthly_warmup_months = max(0, int(monthly_warmup_months))

        # Opt-in: monthly parallelism (still feature-level sequential).
        # This is a compromise: it keeps caching stable and payloads small (per-month slices),
        # while allowing expensive tick/sequence features to compute faster on multi-core machines.
        try:
            env_workers = int(os.getenv("FEATURE_MONTHLY_WORKERS", "1"))
        except Exception:
            env_workers = 1
        self.monthly_workers = int(max_workers) if max_workers is not None else env_workers
        if self.monthly_workers < 1:
            self.monthly_workers = 1
        self.monthly_backend = (
            os.getenv("FEATURE_MONTHLY_BACKEND", "").strip().lower()
            or str(parallel_backend or "").strip().lower()
            or "process"
        )
        if self.monthly_backend not in {"process", "thread"}:
            self.monthly_backend = "process"

        # Simplified design: always run sequentially at the feature level.
        # Performance is achieved via disk/monthly caches rather than process/thread pools.
        self.enable_parallel = False
        self.max_workers = 1
        self.parallel_backend = "sequential"
        self.executor = None
        if self.verbose:
            self._print_memory_info()
            print("   🔧 Feature-level parallelism disabled. Running sequentially.")
            if self.monthly_workers > 1 and self.use_monthly_cache and self.cache_dir:
                print(
                    f"   ⚡ Monthly parallelism enabled: workers={self.monthly_workers}, backend={self.monthly_backend}"
                )

        # 内存缓存：使用 (df_signature, feature_name) 作为键，而不是全局清空
        # 这样即使 DataFrame 切换，相同 DataFrame 签名的特征结果仍可复用
        self.memory_cache = {}  # key: (df_sig_tuple, feature_name), value: cached result

        # Debug stats (last run). Callers can drain via drain_debug_stats().
        # - index_mismatch: feature_name -> {"extra": int, "missing": int}
        # - cache_hits: {"memory": int, "monthly": int, "memory_features": List[str], "monthly_features": List[str]}
        self._debug_stats: Dict[str, Any] = {
            "index_mismatch": {},
            "cache_hits": {
                "memory": 0,
                "monthly": 0,
                "memory_features": [],
                "monthly_features": [],
            },
            "perf": {
                "per_feature": {},  # feature_name -> {"seconds": float, "source": str}
                "slow_features": [],  # list of {"feature": str, "seconds": float, "source": str}
            },
        }

        # 缓存版本控制（用于失效旧缓存）
        # 当特征计算逻辑改变时，更新此版本号以失效旧缓存
        self.cache_version = "v6"

        # 月度缓存目录
        if self.use_monthly_cache and self.cache_dir:
            self.monthly_cache_dir = self.cache_dir / "monthly"
            self.monthly_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.monthly_cache_dir = None

        print(f"   🔧 Workers: {self.max_workers}, Backend: {self.parallel_backend}")

    def _print_memory_info(self) -> None:
        """打印内存信息"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            system_mem = psutil.virtual_memory()
            print(
                f"   💾 Memory: {system_mem.available / 1024**3:.1f}GB available, "
                f"{mem_info.rss / 1024**3:.1f}GB used, "
                f"{system_mem.total / 1024**3:.1f}GB total "
                f"({system_mem.percent:.1f}% used)"
            )

    def drain_debug_stats(self) -> Dict[str, Any]:
        """Return and reset debug stats from the last compute_features_parallel run."""
        stats = self._debug_stats.copy()
        self._debug_stats = {
            "index_mismatch": {},
            "cache_hits": {
                "memory": 0,
                "monthly": 0,
                "memory_features": [],
                "monthly_features": [],
            },
            "perf": {"per_feature": {}, "slow_features": []},
        }
        return stats

    def _get_monthly_cache_key(
        self,
        feature_name: str,
        month_key: str,
        compute_params: Dict,
        feature_info: Optional[Dict] = None,
        df_sig: str = "unknown",
    ) -> str:
        """
        Back-compat wrapper used by tests. Includes monthly warmup in key.
        """
        params = dict(compute_params or {})
        if self.monthly_warmup_months:
            params["__monthly_warmup_months"] = int(self.monthly_warmup_months)
        return _get_monthly_cache_key(
            feature_name=feature_name,
            month_key=month_key,
            cache_version=self.cache_version,
            fit=True,
            compute_params=params,
            data_id=str(df_sig),
        )

    def _split_df_by_month(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Back-compat helper used by tests."""
        if not isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
            raise ValueError("Expected Datetime-like index for monthly split.")
        out: Dict[str, pd.DataFrame] = {}
        for month_key, df_month in df.groupby(pd.Grouper(freq="M")):
            if df_month.empty:
                continue
            out[month_key.strftime("%Y-%m")] = df_month
        return out

    def _save_monthly_cache(self, cache_key: str, data: pd.DataFrame | pd.Series) -> None:
        if not self.monthly_cache_dir:
            return
        _save_monthly_cache(self.monthly_cache_dir, cache_key, data)

    def _load_monthly_cache(self, cache_key: str) -> Optional[pd.DataFrame | pd.Series]:
        if not self.monthly_cache_dir:
            return None
        return _load_monthly_cache(self.monthly_cache_dir, cache_key)

    def _try_monthly_cache(
        self,
        feature_name: str,
        df: pd.DataFrame,
        compute_params: Dict,
        feature_info: Dict,
    ) -> Optional[Dict[str, pd.DataFrame | pd.Series]]:
        """
        Back-compat helper used by tests. Return cached month results if all months exist.
        """
        monthly_dfs = self._split_df_by_month(df)
        results: Dict[str, pd.DataFrame | pd.Series] = {}
        for month_key, month_df in monthly_dfs.items():
            cache_key = self._get_monthly_cache_key(
                feature_name, month_key, compute_params, feature_info, df_sig=self._get_df_signature(month_df)
            )
            cached = self._load_monthly_cache(cache_key)
            if cached is None:
                return None
            results[month_key] = cached
        return results

    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """计算 DataFrame 的哈希值（用于缓存键）"""
        h = hashlib.md5()
        h.update(str(df.index).encode())
        h.update(str(df.shape).encode())
        h.update(str(df.columns.tolist()).encode())
        # 采样一些数据值以确保不同数据集的哈希不同
        if len(df) > 0:
            sample_size = min(100, len(df))
            sample_indices = df.index[:: max(1, len(df) // sample_size)][:sample_size]
            sample_data = df.loc[sample_indices]
            h.update(str(sample_data.values).encode())
        return h.hexdigest()

    def _get_df_signature(self, df: pd.DataFrame) -> str:
        """
        生成 DataFrame 的轻量级签名（用于月度缓存键）

        这个签名应该足够轻量，但又能区分不同的数据集。
        包括：索引范围、长度、列哈希、采样数据值。

        Args:
            df: 输入 DataFrame

        Returns:
            df_sig: DataFrame 签名字符串
        """
        h = hashlib.md5()
        # 索引范围
        if len(df) > 0:
            h.update(str(df.index.min()).encode())
            h.update(str(df.index.max()).encode())
        # 长度
        h.update(str(len(df)).encode())
        # 列哈希
        h.update(str(sorted(df.columns)).encode())
        # 采样数据值（只采样 close 和 volume，如果存在）
        if len(df) > 0:
            sample_size = min(50, len(df))
            sample_indices = df.index[:: max(1, len(df) // sample_size)][:sample_size]
            for col in ["close", "volume"]:
                if col in df.columns:
                    sample_values = df.loc[sample_indices, col].values
                    h.update(str(sample_values).encode())
        return h.hexdigest()

    def _align_to_base_index(
        self, feature_name: str, result: pd.DataFrame | pd.Series, base_index: pd.Index
    ) -> pd.DataFrame | pd.Series:
        """
        将特征结果对齐到基础索引

        Args:
            feature_name: 特征名称（用于调试）
            result: 特征计算结果
            base_index: 基础索引

        Returns:
            aligned_result: 对齐后的结果
        """
        # ------------------------------------------------------------------
        # Index timezone normalization (critical for cache correctness)
        #
        # Cached monthly/disk results may carry tz-naive DatetimeIndex while the
        # current base dataframe index can be tz-aware (UTC) depending on the data loader.
        # A plain reindex() would then yield ALL-NaN (no timestamp matches).
        #
        # Policy:
        # - base tz-aware + result tz-naive  => localize result to base tz (assume same clock, typically UTC)
        # - base tz-naive + result tz-aware  => drop tz from result
        # ------------------------------------------------------------------
        try:
            if isinstance(base_index, pd.DatetimeIndex) and isinstance(
                result, (pd.Series, pd.DataFrame)
            ) and isinstance(result.index, pd.DatetimeIndex):
                base_tz = base_index.tz
                res_tz = result.index.tz
                if base_tz is not None and res_tz is None:
                    _tmp = result.copy()
                    _tmp.index = _tmp.index.tz_localize(base_tz)
                    result = _tmp
                elif base_tz is None and res_tz is not None:
                    _tmp = result.copy()
                    _tmp.index = _tmp.index.tz_localize(None)
                    result = _tmp
        except Exception:
            pass

        if isinstance(result, pd.Series):
            if result.index.has_duplicates:
                # Monthly warmup windows can overlap across months; keep the latest.
                result = result[~result.index.duplicated(keep="last")]
            if not result.index.equals(base_index):
                # 记录索引不匹配统计
                extra = len(result.index.difference(base_index))
                missing = len(base_index.difference(result.index))
                if feature_name not in self._debug_stats["index_mismatch"]:
                    self._debug_stats["index_mismatch"][feature_name] = {
                        "extra": 0,
                        "missing": 0,
                    }
                self._debug_stats["index_mismatch"][feature_name]["extra"] += extra
                self._debug_stats["index_mismatch"][feature_name]["missing"] += missing

                # 对齐到基础索引
                result = result.reindex(base_index)
            return result
        elif isinstance(result, pd.DataFrame):
            if result.index.has_duplicates:
                # Monthly warmup windows can overlap across months; keep the latest.
                result = result[~result.index.duplicated(keep="last")]
            if not result.index.equals(base_index):
                # 记录索引不匹配统计
                extra = len(result.index.difference(base_index))
                missing = len(base_index.difference(result.index))
                if feature_name not in self._debug_stats["index_mismatch"]:
                    self._debug_stats["index_mismatch"][feature_name] = {
                        "extra": 0,
                        "missing": 0,
                    }
                self._debug_stats["index_mismatch"][feature_name]["extra"] += extra
                self._debug_stats["index_mismatch"][feature_name]["missing"] += missing

                # 对齐到基础索引
                result = result.reindex(base_index)
            return result
        else:
            return result

    def _validate_cache_quality(
        self,
        cached_result: pd.DataFrame | pd.Series,
        feature_name: str,
        cache_type: str = "unknown",
    ) -> dict:
        """
        验证缓存数据质量

        Args:
            cached_result: 缓存的结果
            feature_name: 特征名称
            cache_type: 缓存类型（memory/disk/monthly）
        """
        import numpy as np
        import pandas as pd

        warnings: list[str] = []

        # Default payload (safe / deterministic).
        result: dict = {
            "feature_name": feature_name,
            "cache_type": cache_type,
            "total_values": 0,
            "nan_pct": 0.0,
            "inf_pct": 0.0,
            "has_issues": False,
            "warnings": warnings,
        }

        if not isinstance(cached_result, (pd.Series, pd.DataFrame)):
            warnings.append("non_pandas_input")
            return result

        if cached_result.empty:
            warnings.append("empty")
            result["has_issues"] = True
            return result

        # Only validate numeric values. Non-numeric cols are ignored (intended).
        if isinstance(cached_result, pd.Series):
            if not pd.api.types.is_numeric_dtype(cached_result):
                return result
            values = pd.to_numeric(cached_result, errors="coerce").to_numpy()
        else:
            numeric_df = cached_result.select_dtypes(include=[np.number])
            if numeric_df.shape[1] == 0:
                return result
            values = numeric_df.to_numpy()

        total = int(values.size)
        result["total_values"] = total
        if total == 0:
            return result

        nan_count = int(np.isnan(values).sum())
        inf_count = int(np.isinf(values).sum())
        result["nan_pct"] = float(nan_count) / float(total) * 100.0
        result["inf_pct"] = float(inf_count) / float(total) * 100.0

        # Thresholds (kept consistent with integration test expectations)
        nan_threshold_pct = 50.0
        inf_threshold_pct = 10.0
        if result["nan_pct"] > nan_threshold_pct:
            result["has_issues"] = True
            warnings.append(f"high_nan_pct>{nan_threshold_pct}")
        if result["inf_pct"] > inf_threshold_pct:
            result["has_issues"] = True
            warnings.append(f"high_inf_pct>{inf_threshold_pct}")

        return result

    def _compute_and_cache_monthly(
        self,
        feature_name: str,
        df: pd.DataFrame,
        compute_params: Dict,
        feature_info: Dict,
        compute_func: Callable,
    ) -> pd.DataFrame | pd.Series:
        """
        按月计算并缓存特征（支持增量计算）

        Args:
            feature_name: 特征名称
            df: 输入 DataFrame
            compute_params: 计算参数
            feature_info: 特征配置信息
            compute_func: 计算函数

        Returns:
            result: 特征计算结果
        """
        if (
            not self.use_monthly_cache
            or self.monthly_cache_dir is None
            or not isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex))
        ):
            # 如果不使用月度缓存，直接计算
            call_args, call_kwargs = _build_call_args(feature_info, df, feature_name)
            # Filter kwargs by signature unless compute_func accepts **kwargs
            try:
                import inspect

                sig = inspect.signature(compute_func)
                params = sig.parameters
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                if not accepts_var_kw and call_kwargs:
                    allowed = set(params.keys())
                    call_kwargs = {k: v for k, v in call_kwargs.items() if k in allowed}
            except Exception:
                pass

            result = compute_func(*call_args, **call_kwargs)

            # Even without monthly cache, ensure we only materialize output_columns.
            output_cols = feature_info.get("output_columns", [feature_name])
            if not output_cols:
                output_cols = [feature_name]

            if isinstance(result, tuple):
                # Convert tuple of Series to DataFrame using output_columns (best-effort)
                if len(result) == len(output_cols):
                    result = pd.DataFrame(
                        {col: series for col, series in zip(output_cols, result)}
                    )

            if isinstance(result, pd.DataFrame):
                if result.columns.duplicated().any():
                    result = result.loc[:, ~result.columns.duplicated()]
                # Trim wide results strictly to requested output columns (when present)
                existing = [c for c in output_cols if c in result.columns]
                if existing:
                    result = result[existing]

            return result

        # 按月分组
        monthly_groups = df.groupby(pd.Grouper(freq="M"))
        monthly_results: Dict[str, pd.DataFrame | pd.Series] = {}

        # 统计缓存命中情况
        cached_months = 0
        computed_months = 0
        queued_months: List[tuple[str, str, bytes]] = []

        warmup_months = int(self.monthly_warmup_months or 0)
        for month_key, df_month in monthly_groups:
            if df_month.empty:
                continue

            month_str = month_key.strftime("%Y-%m")
            month_start = df_month.index.min()
            month_end = df_month.index.max()
            if warmup_months > 0:
                start_ts = pd.Timestamp(month_start) - pd.DateOffset(months=warmup_months)
                df_window = df.loc[(df.index >= start_ts) & (df.index <= month_end)]
            else:
                df_window = df_month
            compute_params_key = dict(compute_params or {})
            if warmup_months > 0:
                compute_params_key["__monthly_warmup_months"] = warmup_months

            # 生成月度缓存键（include warmup + month signature）
            df_sig = self._get_df_signature(df_month)
            monthly_cache_key = _get_monthly_cache_key(
                feature_name=feature_name,
                month_key=month_str,
                cache_version=self.cache_version,
                fit=True,  # 月度缓存不区分 fit/predict
                compute_params=compute_params_key,
                data_id=str(df_sig),
            )

            # 尝试加载缓存
            cached_result = _load_monthly_cache(
                self.monthly_cache_dir, monthly_cache_key
            )

            if cached_result is not None:
                # 验证缓存质量
                self._validate_cache_quality(
                    cached_result, feature_name, cache_type="monthly"
                )
                # Handle cached tuple results (e.g., old cache format for macd_f)
                output_cols = feature_info.get("output_columns", [feature_name])
                if not output_cols:
                    output_cols = [feature_name]
                if isinstance(cached_result, tuple) and len(cached_result) == len(output_cols):
                    # Convert cached tuple to DataFrame (same as new computation)
                    cached_result = pd.DataFrame(
                        {col: series for col, series in zip(output_cols, cached_result)}
                    )
                monthly_results[month_str] = cached_result
                cached_months += 1
                # 记录月度缓存命中（每个特征只记录一次）
                if feature_name not in self._debug_stats["cache_hits"]["monthly_features"]:
                    self._debug_stats["cache_hits"]["monthly_features"].append(feature_name)
            else:
                # Monthly parallel path (opt-in): queue cache-miss months for parallel compute.
                if int(getattr(self, "monthly_workers", 1)) > 1:
                    queued_months.append(
                        (
                            month_str,
                            monthly_cache_key,
                            pickle.dumps(df_window, protocol=pickle.HIGHEST_PROTOCOL),
                        )
                    )
                    continue

                # 计算特征
                print(f"       🔸 Computing {feature_name} for {month_str}...")
                call_args, call_kwargs = _build_call_args(
                    feature_info, df_window, feature_name
                )
                # Some configs may include non-compute metadata keys inside compute_params.
                # To avoid runtime failures, filter kwargs by the compute_func signature
                # unless it explicitly accepts **kwargs.
                try:
                    import inspect

                    sig = inspect.signature(compute_func)
                    params = sig.parameters
                    accepts_var_kw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                    )
                    if not accepts_var_kw and call_kwargs:
                        allowed = set(params.keys())
                        call_kwargs = {k: v for k, v in call_kwargs.items() if k in allowed}
                except Exception:
                    # Defensive: if signature introspection fails, run as-is.
                    pass

                # If no args were built but compute_func expects a DataFrame, pass df_window.
                if not call_args:
                    try:
                        import inspect

                        sig = inspect.signature(compute_func)
                        params = list(sig.parameters.keys())
                        if params and params[0] in {"df", "data", "frame"}:
                            call_args = [df_window]
                    except Exception:
                        pass
                month_result = compute_func(*call_args, **call_kwargs)
                
                # Debug: check return type for macd_f
                if feature_name == "macd_f":
                    print(f"       🔍 DEBUG macd_f: compute_func returned type: {type(month_result)}")
                    if isinstance(month_result, tuple):
                        print(f"       🔍 DEBUG macd_f: tuple length: {len(month_result)}")

                # Trim to current month index after warmup computation
                if isinstance(month_result, (pd.Series, pd.DataFrame)):
                    try:
                        month_result = month_result.reindex(df_month.index)
                    except Exception:
                        pass

                # 处理计算结果的重复列名
                if (
                    isinstance(month_result, pd.DataFrame)
                    and month_result.columns.duplicated().any()
                ):
                    month_result = month_result.loc[
                        :, ~month_result.columns.duplicated()
                    ]

                # 根本性解决方案：只提取 output_columns 中定义的列
                output_cols = feature_info.get("output_columns", [feature_name])
                if not output_cols:
                    output_cols = [feature_name]

                # Handle tuple return values (e.g., compute_macd returns Tuple[Series, Series, Series])
                if isinstance(month_result, tuple):
                    # Convert tuple of Series to DataFrame using output_columns
                    if len(month_result) == len(output_cols):
                        month_result = pd.DataFrame(
                            {col: series for col, series in zip(output_cols, month_result)}
                        )
                        # Debug: verify conversion worked
                        if feature_name == "macd_f":
                            print(f"       🔍 DEBUG macd_f: Converted tuple to DataFrame with columns: {list(month_result.columns)}")
                    else:
                        # Fallback: use default names if output_cols length doesn't match
                        month_result = pd.DataFrame(
                            {f"{feature_name}_{i}": series for i, series in enumerate(month_result)}
                        )
                        if feature_name == "macd_f":
                            print(f"       🔍 DEBUG macd_f: Fallback conversion, columns: {list(month_result.columns)}, output_cols: {output_cols}")
                elif isinstance(month_result, pd.DataFrame):
                    # 只保留 output_columns 中定义的列
                    result_cols = [
                        col for col in output_cols if col in month_result.columns
                    ]
                    if result_cols:
                        month_result = month_result[result_cols].copy()
                    else:
                        month_result = pd.DataFrame(index=month_result.index)
                elif isinstance(month_result, pd.Series):
                    series_name = (
                        month_result.name if month_result.name else feature_name
                    )
                    if series_name not in output_cols:
                        # 如果不在 output_columns 中，返回空 DataFrame
                        month_result = pd.DataFrame(index=month_result.index)

                # 保存缓存（只保存 output_columns）
                _save_monthly_cache(self.monthly_cache_dir, monthly_cache_key, month_result)

                monthly_results[month_str] = month_result
                computed_months += 1

            # 打印每个月份的结果信息
            if isinstance(monthly_results[month_str], pd.DataFrame):
                month_result = monthly_results[month_str]
                print(
                    f"       📊 Month {month_str}: {len(month_result)} rows, {len(month_result.columns)} columns: {list(month_result.columns)[:5]}..."
                )
                if month_result.columns.duplicated().any():
                    dup_cols = month_result.columns[
                        month_result.columns.duplicated()
                    ].tolist()
                    print(
                        f"       ⚠️  Month {month_str} has duplicate columns: {dup_cols}"
                    )
            elif isinstance(monthly_results[month_str], pd.Series):
                month_result = monthly_results[month_str]
                print(
                    f"       📊 Month {month_str}: {len(month_result)} rows, Series name: {month_result.name}"
                )

        # Compute queued cache-miss months in parallel (if enabled)
        if queued_months and int(getattr(self, "monthly_workers", 1)) > 1:
            backend = str(getattr(self, "monthly_backend", "process") or "process").lower()
            max_workers = int(getattr(self, "monthly_workers", 1))
            Exec = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
            print(
                f"       ⚡ Computing {len(queued_months)} month(s) in parallel "
                f"(workers={max_workers}, backend={backend})..."
            )
            futures = {}
            with Exec(max_workers=max_workers) as ex:
                for mstr, cache_key, df_b in queued_months:
                    fut = ex.submit(
                        _compute_single_feature_worker_monthly,
                        feature_name,
                        feature_info,
                        df_b,
                        True,
                        str(self.monthly_cache_dir) if self.monthly_cache_dir else None,
                        None,
                    )
                    futures[fut] = (mstr, cache_key)
                for fut in as_completed(futures):
                    mstr, cache_key = futures[fut]
                    _, res_bytes = fut.result()
                    month_result = pickle.loads(res_bytes)
                    _save_monthly_cache(self.monthly_cache_dir, cache_key, month_result)
                    monthly_results[mstr] = month_result
                    computed_months += 1

        # 获取特征的输出列（根本性解决方案：只返回 output_columns 中定义的列）
        output_cols = feature_info.get("output_columns", [feature_name])
        if not output_cols:
            output_cols = [feature_name]

        # 合并所有月份的结果
        print(f"       🔄 Merging {len(monthly_results)} monthly results...")
        try:
            def _tz_normalize_index(x):
                """
                Normalize DatetimeIndex to UTC tz-naive to avoid pandas comparisons failing
                when some monthly chunks are tz-aware and others are tz-naive.
                """
                try:
                    if hasattr(x, "index") and isinstance(x.index, pd.DatetimeIndex):
                        if x.index.tz is not None:
                            y = x.copy()
                            y.index = y.index.tz_convert("UTC").tz_localize(None)
                            return y
                except Exception:
                    return x
                return x

            # 处理不同的返回类型
            if isinstance(list(monthly_results.values())[0], tuple):
                # 如果是tuple，需要分别合并每个元素
                combined_results = []
                for i in range(len(output_cols)):
                    combined_series = pd.concat(
                        [_tz_normalize_index(r[i]) for r in monthly_results.values()],
                        axis=0,
                    ).sort_index()
                    combined_results.append(combined_series)
                result_df = pd.DataFrame(
                    {col: series for col, series in zip(output_cols, combined_results)}
                )
            elif isinstance(list(monthly_results.values())[0], pd.DataFrame):
                # 根本性解决方案：只保留 output_columns 中定义的列
                all_columns = set(output_cols)  # 只使用 output_columns，不包含其他列

                # 确保所有 DataFrame 都有相同的列（缺失的列填充 NaN）
                aligned_results = []
                for month_key, month_result in monthly_results.items():
                    month_result = _tz_normalize_index(month_result)
                    # 处理重复列名：如果有重复列，保留第一个
                    if (
                        isinstance(month_result, pd.DataFrame)
                        and month_result.columns.duplicated().any()
                    ):
                        dup_cols = month_result.columns[
                            month_result.columns.duplicated()
                        ].tolist()
                        print(
                            f"       ⚠️  Month {month_key} has duplicate columns before dedup: {dup_cols}"
                        )
                        month_result = month_result.loc[
                            :, ~month_result.columns.duplicated()
                        ]
                        print(
                            f"       ✅ Month {month_key} after dedup: {len(month_result.columns)} columns"
                        )
                        # 更新字典中的值
                        monthly_results[month_key] = month_result

                    # 只提取 output_columns 中定义的列
                    if isinstance(month_result, pd.DataFrame):
                        result_cols = [
                            col for col in output_cols if col in month_result.columns
                        ]
                        if result_cols:
                            # 确保列顺序一致
                            month_result = month_result[result_cols].copy()
                        else:
                            # Debug: if no columns match, check what columns we have
                            if feature_name == "macd_f":
                                print(f"       🔍 DEBUG macd_f merge: month {month_key} has columns {list(month_result.columns)}, expected {output_cols}")
                            month_result = pd.DataFrame(index=month_result.index)
                    aligned_results.append(month_result)

                # 合并所有月份的结果
                result_df = pd.concat(aligned_results, axis=0).sort_index()

                # 确保只保留 output_columns 中定义的列
                result_cols = [
                    col for col in output_cols if col in result_df.columns
                ]
                if result_cols:
                    result_df = result_df[result_cols].copy()
                    # Debug: verify MACD columns are present after merge
                    if feature_name == "macd_f":
                        print(f"       🔍 DEBUG macd_f: After merge, result_df has columns: {list(result_df.columns)}")
                else:
                    # Debug: if no columns match, check what we have
                    if feature_name == "macd_f":
                        print(f"       🔍 DEBUG macd_f: No matching columns! result_df has: {list(result_df.columns)}, expected: {output_cols}")
                    result_df = pd.DataFrame(index=result_df.index)
            elif isinstance(list(monthly_results.values())[0], pd.Series):
                # 合并 Series
                combined_series = pd.concat(
                    [_tz_normalize_index(s) for s in list(monthly_results.values())],
                    axis=0,
                ).sort_index()
                # 如果 output_columns 中只有一个列，且 Series name 匹配，则返回 Series
                if len(output_cols) == 1 and combined_series.name == output_cols[0]:
                    result_df = combined_series
                else:
                    # 否则转换为 DataFrame
                    result_df = pd.DataFrame({output_cols[0]: combined_series})
            else:
                raise ValueError(
                    f"Unexpected result type: {type(list(monthly_results.values())[0])}"
                )
        except Exception as e:
            print(f"       ❌ Error merging monthly results: {e}")
            raise

        # 打印缓存统计
        if cached_months > 0 or computed_months > 0:
            if self.verbose:
                print(
                    f"       💾 [monthly] Used {cached_months} cached months, computed {computed_months} new months"
                )
            # 更新月度缓存命中总数（按月份数累加）
            self._debug_stats["cache_hits"]["monthly"] += cached_months

        return result_df

    def compute_features_parallel(
        self,
        df: pd.DataFrame,
        features: Dict,
        requested_features: List[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        计算特征（顺序执行 + 缓存）

        Args:
            df: 输入 DataFrame
            features: 特征配置字典
            requested_features: 请求的特征列表
            fit: 是否拟合

        Returns:
            result_df: 包含计算特征的 DataFrame
        """
        result_df = df.copy()
        if result_df.empty:
            return result_df
        base_index = result_df.index
        df_hash = self._get_df_hash(result_df)

        # Memory cache key: use (df_signature, feature_name) instead of global clearing.
        # This allows cross-variant reuse when the same DataFrame signature is encountered.
        # We still track the current signature for logging/debugging purposes.
        current_df_sig = None
        if self.use_memory_cache:
            try:
                current_df_sig = (
                    df_hash,
                    len(result_df),
                    str(result_df.index.min()) if len(result_df) else "EMPTY",
                    str(result_df.index.max()) if len(result_df) else "EMPTY",
                )
                # Store current signature for potential future use (e.g., cache size limits)
                self._current_df_sig = current_df_sig
            except Exception:
                # If signature calculation fails, disable memory cache for this call
                current_df_sig = None

        # 1.5. 确保所有请求特征的 required_columns 都在 DataFrame 中
        # 收集所有需要的 required_columns
        all_required_columns = set()
        for feature_name in requested_features:
            if feature_name in features:
                feature_info = features[feature_name]
                required_columns = feature_info.get("required_columns", [])
                all_required_columns.update(required_columns)

        # 检查缺失的 required_columns
        missing_required = [
            col for col in all_required_columns if col not in result_df.columns
        ]
        if missing_required:
            # 检查哪些是基础数据列（可能在原始df中），哪些是特征输出列（会通过依赖关系计算）
            base_data_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            base_missing = [c for c in missing_required if c in base_data_cols]
            feature_missing = [c for c in missing_required if c not in base_data_cols]

            if base_missing:
                raise ValueError(
                    f"Missing required base data columns: {base_missing}. "
                    f"Please ensure these columns exist in the input DataFrame."
                )
            # feature_missing 会在依赖计算时自动解决，这里只警告
            # Suppress warning if there are too many missing columns (likely all will be computed via dependencies)
            # Only warn if there are few missing columns (might indicate a real issue)
            # Also suppress if missing columns are all from optional_blocks (expected to be missing sometimes)
            if feature_missing:
                # Check if missing columns are mostly from optional blocks (vpin, trade_cluster, etc.)
                optional_block_patterns = ['vpin', 'trade_cluster', 'volume_profile', 'vp_']
                is_mostly_optional = sum(
                    1 for col in feature_missing[:20] 
                    if any(pattern in col.lower() for pattern in optional_block_patterns)
                ) >= len(feature_missing[:20]) * 0.7  # 70% are optional block features
                
                if not is_mostly_optional and len(feature_missing) <= 20:
                    if self.verbose:
                        print(
                            f"   ⚠️  Missing feature columns (will be computed via dependencies): {feature_missing[:10]}..."
                        )
                # If mostly optional blocks or too many missing, suppress warning (expected behavior)
            elif feature_missing and len(feature_missing) > 20:
                # Too many missing columns, likely all will be computed - suppress verbose warning
                pass

        # 2. 分析依赖层级
        levels = analyze_dependency_levels(features, requested_features)

        # 3. 按层级顺序计算特征
        computed_features = set()
        total_start_time = time.perf_counter()
        for level in sorted(levels.keys()):
            level_features = levels[level]
            level_start_time = time.perf_counter()
            if self.verbose:
                print(
                    f"   ▶️ Level {level}: {len(level_features)} features "
                    f"(Mem: {psutil.Process(os.getpid()).memory_info().rss / 1024**3:.2f}GB)"
                )

            # 在同一层内做一个简单的拓扑顺序：优先计算依赖已完成的特征
            pending = list(level_features)
            safeguard = len(pending) * 2  # 防止意外死循环
            while pending and safeguard > 0:
                feature_name = pending.pop(0)
                safeguard -= 1

                if feature_name in computed_features:
                    continue

                # 如果依赖未满足，放到队尾，等待依赖先计算
                deps = features.get(feature_name, {}).get("dependencies", [])
                if any(dep not in computed_features for dep in deps):
                    pending.append(feature_name)
                    continue

                feature_info = features[feature_name]
                # 优先使用 feature_dependencies.yaml 中显式配置的 compute_func，
                # 否则退回到以 feature_name 作为注册名（兼容旧配置）。
                compute_func_name = feature_info.get("compute_func", feature_name)
                compute_func = get_compute_func(compute_func_name)

                if compute_func is None:
                    print(
                        f"     ⚠️  Skipping {feature_name}: compute function "
                        f"'{compute_func_name}' not found in registry"
                    )
                    continue

                # 检查内存缓存：使用 (df_signature, feature_name) 作为键
                cache_key = None
                cached_result = None
                t0 = time.perf_counter()
                if self.use_memory_cache and current_df_sig is not None:
                    cache_key = (current_df_sig, feature_name)
                    if cache_key in self.memory_cache:
                        print(
                            f"     💾 [memory] Using memory cache for {feature_name} "
                            f"(df_sig={current_df_sig[0][:8]}...)"
                        )
                        # 记录内存缓存命中
                        self._debug_stats["cache_hits"]["memory"] += 1
                        if feature_name not in self._debug_stats["cache_hits"]["memory_features"]:
                            self._debug_stats["cache_hits"]["memory_features"].append(feature_name)
                        cached_result = self.memory_cache[cache_key]

                        # SAFETY: cached results must match current index; otherwise it will
                        # explode the index (train/test/volatility mix) and can cause huge memory use.
                        try:
                            if isinstance(cached_result, (pd.Series, pd.DataFrame)):
                                if not cached_result.index.equals(result_df.index):
                                    print(
                                        f"     ⚠️  Memory cache index mismatch for {feature_name} "
                                        f"(cached={len(cached_result)}, current={len(result_df)}), recomputing."
                                    )
                                    cached_result = None
                        except Exception:
                            # Be conservative: if we cannot validate, don't use cache
                            cached_result = None

                        if cached_result is not None:
                            cached_result = self._align_to_base_index(
                                feature_name, cached_result, base_index
                            )
                            # 验证cache数据质量
                            self._validate_cache_quality(
                                cached_result, feature_name, cache_type="memory"
                            )
                            # 合并结果（支持 Series 和 DataFrame）
                            if isinstance(cached_result, pd.DataFrame):
                                new_cols = [
                                    c
                                    for c in cached_result.columns
                                    if c not in result_df.columns
                                ]
                                existing_cols = [
                                    c
                                    for c in cached_result.columns
                                    if c in result_df.columns
                                ]

                                # 处理已存在的列：直接丢弃新列（重名的应该是一样的，不需要合并）
                                if existing_cols:
                                    # 直接跳过，不合并（节省内存和时间）
                                    pass

                                # 添加新列
                                if new_cols:
                                    # Ensure both DataFrames are aligned to base_index before concat
                                    # This prevents index expansion when indices don't match
                                    cached_aligned = cached_result[new_cols].reindex(
                                        base_index
                                    )
                                    result_df = pd.concat(
                                        [result_df, cached_aligned], axis=1
                                    )
                            elif isinstance(cached_result, pd.Series):
                                if cached_result.name not in result_df.columns:
                                    cached_aligned = cached_result.reindex(base_index)
                                    result_df[cached_result.name] = cached_aligned

                            computed_features.add(feature_name)
                            dt = float(time.perf_counter() - t0)
                            self._debug_stats["perf"]["per_feature"][feature_name] = {
                                "seconds": dt,
                                "source": "memory",
                            }
                            continue

                # 如果没有缓存，计算特征
                compute_params = feature_info.get("compute_params", {})
                print(f"     ▶️ {feature_name}: start (level {level})") if self.verbose else None

                # 检查是否使用月度缓存
                if self.use_monthly_cache and self.monthly_cache_dir:
                    # 使用月度缓存计算
                    print(f"     🔸 Running {feature_name} sequentially (monthly)") if self.verbose else None
                    # Always use monthly cache path (even for small DataFrames).
                    # Rationale: the goal of monthly caching is cross-run reuse; bypassing it
                    # destroys cache hit-rate in multi-seed / multi-step workflows.
                    combined_result = self._compute_and_cache_monthly(
                        feature_name,
                        result_df,
                        compute_params,
                        feature_info,
                        compute_func,
                    )

                    # 验证合并后的cache数据质量
                    self._validate_cache_quality(
                        combined_result, feature_name, cache_type="monthly"
                    )
                    combined_result = self._align_to_base_index(
                        feature_name, combined_result, base_index
                    )
                    # Store in memory cache with (df_signature, feature_name) key
                    if self.use_memory_cache and current_df_sig is not None:
                        cache_key = (current_df_sig, feature_name)
                        self.memory_cache[cache_key] = combined_result
                    # 合并到result_df
                    if isinstance(combined_result, pd.DataFrame):
                        # 处理重复列名：如果有重复列，保留第一个
                        if combined_result.columns.duplicated().any():
                            combined_result = combined_result.loc[
                                :, ~combined_result.columns.duplicated()
                            ]

                        new_cols = [
                            c
                            for c in combined_result.columns
                            if c not in result_df.columns
                        ]
                        existing_cols = [
                            c
                            for c in combined_result.columns
                            if c in result_df.columns
                        ]

                        # 处理已存在的列：直接丢弃新列（重名的应该是一样的，不需要合并）
                        if existing_cols:
                            # 直接跳过，不合并（节省内存和时间）
                            pass

                        # 添加新列
                        if new_cols:
                            # Ensure both DataFrames are aligned to base_index before concat
                            # This prevents index expansion when indices don't match
                            combined_aligned = combined_result[new_cols].reindex(
                                base_index
                            )
                            result_df = pd.concat([result_df, combined_aligned], axis=1)
                    elif isinstance(combined_result, pd.Series):
                        if combined_result.name not in result_df.columns:
                            combined_aligned = combined_result.reindex(base_index)
                            result_df[combined_result.name] = combined_aligned
                    dt = float(time.perf_counter() - t0)
                    self._debug_stats["perf"]["per_feature"][feature_name] = {
                        "seconds": dt,
                        "source": "monthly",
                    }
                else:
                    # 不使用月度缓存，直接计算
                    call_args, call_kwargs = _build_call_args(
                        feature_info, result_df, feature_name
                    )
                    # Filter kwargs by signature unless compute_func accepts **kwargs
                    try:
                        import inspect

                        sig = inspect.signature(compute_func)
                        params = sig.parameters
                        accepts_var_kw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                        )
                        if not accepts_var_kw and call_kwargs:
                            allowed = set(params.keys())
                            call_kwargs = {
                                k: v for k, v in call_kwargs.items() if k in allowed
                            }
                    except Exception:
                        pass
                    computed_result = compute_func(*call_args, **call_kwargs)

                    # Handle tuple return values (e.g., compute_macd returns Tuple[Series, Series, Series])
                    output_cols = feature_info.get("output_columns", [feature_name])
                    if not output_cols:
                        output_cols = [feature_name]
                    if isinstance(computed_result, tuple):
                        if len(computed_result) == len(output_cols):
                            computed_result = pd.DataFrame(
                                {col: series for col, series in zip(output_cols, computed_result)}
                            )
                        else:
                            # Fallback: use default names if output_cols length doesn't match
                            computed_result = pd.DataFrame(
                                {f"{feature_name}_{i}": series for i, series in enumerate(computed_result)}
                            )

                    # 对齐到基础索引
                    computed_result = self._align_to_base_index(
                        feature_name, computed_result, base_index
                    )

                    # 验证新计算的特征质量
                    self._validate_cache_quality(
                        computed_result,
                        feature_name,
                        cache_type="computed",
                    )

                    # 保存内存缓存
                    if self.use_memory_cache and current_df_sig is not None:
                        cache_key = (current_df_sig, feature_name)
                        self.memory_cache[cache_key] = computed_result

                    # 合并到result_df
                    if isinstance(computed_result, pd.DataFrame):
                        new_cols = [
                            c
                            for c in computed_result.columns
                            if c not in result_df.columns
                        ]
                        if new_cols:
                            computed_aligned = computed_result[new_cols].reindex(
                                base_index
                            )
                            result_df = pd.concat([result_df, computed_aligned], axis=1)
                    elif isinstance(computed_result, pd.Series):
                        if computed_result.name not in result_df.columns:
                            computed_aligned = computed_result.reindex(base_index)
                            result_df[computed_result.name] = computed_aligned
                    dt = float(time.perf_counter() - t0)
                    self._debug_stats["perf"]["per_feature"][feature_name] = {
                        "seconds": dt,
                        "source": "computed",
                    }

                print(f"     ✅ Computed {feature_name}") if self.verbose else None
                # Simple "alarm" for very slow features (usually tick-heavy or cache-miss).
                try:
                    dt_val = float(self._debug_stats["perf"]["per_feature"][feature_name]["seconds"])
                    if dt_val >= 30.0:
                        self._debug_stats["perf"]["slow_features"].append(
                            {
                                "feature": feature_name,
                                "seconds": dt_val,
                                "source": self._debug_stats["perf"]["per_feature"][feature_name]["source"],
                            }
                        )
                        print(
                            f"     ⚠️  SLOW feature: {feature_name} took {dt_val:.1f}s "
                            f"(source={self._debug_stats['perf']['per_feature'][feature_name]['source']})"
                        )
                except Exception:
                    pass
                computed_features.add(feature_name)

            if self.verbose:
                print(
                    f"   ✅ Level {level} done: {len([f for f in level_features if f in computed_features])} features "
                    f"in {time.perf_counter() - level_start_time:.1f}s "
                    f"(Mem: {psutil.Process(os.getpid()).memory_info().rss / 1024**3:.2f}GB)"
                )

        # 无论 verbose 与否，打印一行最终摘要
        total_time = time.perf_counter() - total_start_time
        slow = [s for s in self._debug_stats["perf"]["slow_features"]]
        slow_msg = f" (⚠️ {len(slow)} slow)" if slow else ""
        print(
            f"   📊 Features computed: {len(computed_features)}/{len(requested_features)} "
            f"in {total_time:.1f}s{slow_msg}"
        )

        # 4. 清理内存
        if self.use_memory_cache:
            # 可选：限制内存缓存大小（避免内存爆炸）
            # 这里暂时不限制，因为键是基于 (df_signature, feature_name) 的，通常不会太多
            pass

        return result_df

    def clear_cache(self, memory: bool = True, disk: bool = False) -> None:
        """
        清除缓存

        Args:
            memory: 是否清除内存缓存
            disk: 是否清除磁盘缓存
        """
        if memory:
            self.memory_cache.clear()
            print("   🗑️  Memory cache cleared")
        if disk and self.cache_dir:
            import shutil

            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.monthly_cache_dir:
                self.monthly_cache_dir.mkdir(parents=True, exist_ok=True)
            print("   🗑️  Disk cache cleared")


__all__ = ["FeatureComputer", "analyze_dependency_levels"]
