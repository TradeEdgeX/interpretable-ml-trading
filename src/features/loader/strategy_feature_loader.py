"""
基于纯配置文件的特征加载器（支持并行计算和缓存）
"""

import time
from typing import List, Dict, Optional

import yaml
import pandas as pd
from pathlib import Path

from src.features.loader.feature_computer import FeatureComputer


class StrategyFeatureLoader:
    """
    基于纯配置文件的特征加载器（支持并行计算和缓存）

    特点：
    1. 特征依赖关系在 YAML 中定义
    2. 策略特征集在 YAML 中定义
    3. 自动解析依赖，按需计算
    4. 支持并行计算和缓存
    """

    def __init__(
        self,
        feature_deps_path: str = "config/feature_dependencies.yaml",
        strategy_config_path: Optional[str] = None,
        cache_dir: Optional[str] = "cache/features",
        use_disk_cache: bool = True,
        use_memory_cache: bool = True,
        use_monthly_cache: bool = True,
        monthly_warmup_months: Optional[int] = None,
        max_workers: Optional[int] = None,
        parallel_backend: str = "process",
        normalization_contract_mode: str = "warn",  # "warn" | "error"
        verbose: bool = True,  # 研究=True（详细日志），实盘=False（只打印异常+摘要）
    ):
        """
        初始化特征加载器

        Args:
            feature_deps_path: 特征依赖配置文件路径
            strategy_config_path: 策略配置文件路径（可选，如果文件不存在则不加载）
            cache_dir: 磁盘缓存目录
            use_disk_cache: 是否使用磁盘缓存
            use_memory_cache: 是否使用内存缓存
            max_workers: 最大并行数
            parallel_backend: 并行后端（process/thread）
        """
        self.feature_deps = self._load_yaml(feature_deps_path)

        # Normalization contract (config-level): prevent drift between "declared normalized"
        # and actually specified normalization methods.
        #
        # Default is warn for ergonomics; CI/tests should enforce via mode="error".
        try:
            from src.features.normalization.feature_contract import (
                validate_feature_dependencies_normalization,
            )

            validate_feature_dependencies_normalization(
                self.feature_deps, mode=normalization_contract_mode
            )
        except Exception as e:
            if normalization_contract_mode == "error":
                raise
            print(f"   ⚠️  Normalization contract warning: {e}")
        # 可选加载策略配置（用于向后兼容）
        if strategy_config_path is not None:
            self.strategy_config = self._load_yaml_optional(strategy_config_path)
        else:
            self.strategy_config = {}

        # 创建并行计算器
        self.computer = FeatureComputer(
            cache_dir=cache_dir,
            use_disk_cache=use_disk_cache,
            use_memory_cache=use_memory_cache,
            use_monthly_cache=use_monthly_cache,
            monthly_warmup_months=monthly_warmup_months,
            max_workers=max_workers,
            parallel_backend=parallel_backend,
            verbose=verbose,
        )

    def _load_yaml(self, path: str) -> Dict:
        """加载 YAML 配置文件（必需）"""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # NOTE:
        # Some workflows (feature-group-search background runs) may overlap with interactive edits
        # to YAML configs. If a file is being written, a reader may briefly observe an incomplete
        # YAML and get a transient yaml.scanner.ScannerError.
        #
        # We retry a few times to make the loader resilient to those transient states.
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                with open(path_obj, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data is None:
                    raise ValueError(f"YAML file is empty or invalid (parsed None): {path}")
                return data
            except (OSError, yaml.YAMLError, ValueError) as e:
                last_err = e
                # small backoff: 50ms, 100ms, 150ms, ...
                time.sleep(0.05 * (attempt + 1))
        raise RuntimeError(f"Failed to load YAML after retries: {path}") from last_err

    def _load_yaml_optional(self, path: str) -> Dict:
        """加载 YAML 配置文件（可选，文件不存在时返回空字典）"""
        path_obj = Path(path)
        if not path_obj.exists():
            return {}

        try:
            return self._load_yaml(path) or {}
        except Exception:
            # Optional config should never bring down the whole pipeline.
            return {}

    def resolve_dependencies(self, requested_features: List[str]) -> List[str]:
        """
        解析特征依赖关系，返回计算顺序（拓扑排序）

        Args:
            requested_features: 请求的特征列表（可以是特征计算函数名或特征输出列名）
                - 特征计算函数名必须带 _f 后缀（例如：bb_width_f）
                - 特征输出列名不需要 _f 后缀（例如：bb_upper）

        Returns:
            computation_order: 计算顺序（特征计算函数名，带 _f 后缀）

        Raises:
            ValueError: 如果请求的特征计算函数名不带 _f 后缀
        """
        features = self.feature_deps.get("features", {})

        # 1. 验证并收集所有需要的特征（包括依赖）
        all_needed = set()
        queue = []

        for item in requested_features:
            # 首先检查是否是旧名称（不带 _f 后缀但带 _f 的版本存在）
            # 如果存在带 _f 的版本，说明这是旧的特征计算函数名，必须报错
            potential_new_name = f"{item}_f"
            if potential_new_name in features:
                # If `item` is an output column of `{item}_f`, treat it as an output-column request.
                # This keeps the API user-friendly (tests and some callers request output columns).
                output_cols = features.get(potential_new_name, {}).get("output_columns", [])
                if item in output_cols:
                    if potential_new_name not in all_needed:
                        all_needed.add(potential_new_name)
                        queue.append(potential_new_name)
                    continue

                # Otherwise, this is likely an old compute-function name and should error.
                raise ValueError(
                    f"Feature compute function name must end with '_f' suffix. "
                    f"Got: '{item}'. Did you mean '{potential_new_name}'?"
                )

            # 检查是否是特征计算函数名（必须带 _f 后缀）
            if item in features:
                if not item.endswith("_f"):
                    raise ValueError(
                        f"Feature compute function name must end with '_f' suffix. "
                        f"Got: '{item}'. Did you mean '{item}_f'?"
                    )
                all_needed.add(item)
                queue.append(item)
            else:
                # 可能是输出列名，尝试找到对应的特征
                found = False
                for feat_name, feat_info in features.items():
                    output_cols = feat_info.get("output_columns", [])
                    if item in output_cols:
                        if feat_name not in all_needed:
                            all_needed.add(feat_name)
                            queue.append(feat_name)
                        found = True
                        break

                if not found:
                    # 如果既不是特征计算函数名也不是输出列名，保留它（可能是原始列或其他）
                    # 但不会添加到 all_needed 中
                    pass

        # 2. 收集所有依赖的特征
        while queue:
            feature = queue.pop(0)
            if feature in features:
                deps = features[feature].get("dependencies", [])
                for dep in deps:
                    # 验证依赖特征名必须带 _f 后缀
                    if dep not in features:
                        raise ValueError(
                            f"Dependency '{dep}' not found in feature definitions. "
                            f"Feature compute function names must end with '_f' suffix."
                        )
                    if not dep.endswith("_f"):
                        raise ValueError(
                            f"Dependency feature compute function name must end with '_f' suffix. "
                            f"Got: '{dep}'. Did you mean '{dep}_f'?"
                        )
                    if dep not in all_needed:
                        all_needed.add(dep)
                        queue.append(dep)

        # 3. 构建依赖图
        graph = {f: [] for f in all_needed}
        in_degree = {f: 0 for f in all_needed}

        for feature in all_needed:
            if feature in features:
                deps = features[feature].get("dependencies", [])
                for dep in deps:
                    if dep in all_needed:
                        graph[dep].append(feature)
                        in_degree[feature] += 1

        # 3. 拓扑排序
        queue = [f for f in all_needed if in_degree[f] == 0]
        result = []

        while queue:
            feature = queue.pop(0)
            result.append(feature)
            for neighbor in graph[feature]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 4. 检查循环依赖
        if len(result) != len(all_needed):
            raise ValueError("Circular dependency detected!")

        return result

    def load_strategy_features(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        fit: bool = True,
        requested_features_override: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        为指定策略加载特征（支持并行计算和缓存）

        注意：此方法已废弃，推荐使用 load_features_from_requested() 配合目录管理方式。
        此方法仅用于向后兼容，需要 strategy_features.yaml 文件存在。

        Args:
            df: 输入 DataFrame
            strategy_name: 策略名称（sr_reversal, sr_breakout, compression_breakout, trend_following）
            fit: 是否拟合（研究阶段=True，实盘阶段=False）
            requested_features_override: 可选的特征列表覆盖

        Returns:
            df_with_features: 包含计算特征的 DataFrame
        """
        if not self.strategy_config or "strategies" not in self.strategy_config:
            raise ValueError(
                f"Strategy config not loaded. "
                f"Please use load_features_from_requested() with directory-based configs instead."
            )

        if strategy_name not in self.strategy_config.get("strategies", {}):
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available strategies: {list(self.strategy_config.get('strategies', {}).keys())}"
            )

        strategy_config = self.strategy_config["strategies"][strategy_name]
        result_df = df.copy()

        requested_features = (
            requested_features_override
            if requested_features_override is not None
            else strategy_config.get("requested_features", [])
        )
        if not requested_features:
            print(f"   ⚠️  No requested features for strategy {strategy_name}")
            return result_df

        features = self.feature_deps.get("features", {})

        result_df = self.computer.compute_features_parallel(
            result_df,
            features,
            requested_features,
            fit=fit,
        )

        # Attach debug stats for downstream diagnostics (e.g. feature-induced index drift)
        try:
            result_df.attrs["feature_debug_stats"] = self.computer.drain_debug_stats()
        except Exception:
            pass

        # 3. 只返回请求的特征列（以及它们的输出列）
        # Build requested output columns for the final returned frame.
        #
        # Important:
        # - `requested_features` may contain feature nodes (e.g. `atr_ratio_f`) OR
        #   specific output column names (e.g. `trade_cluster_absorption_scene_score`)
        #   when semantic singletons are enabled.
        # - We keep only the explicitly requested outputs, not intermediate dependency outputs.
        output_col_to_feature: Dict[str, str] = {}
        for feat_name, feat_info in features.items():
            for col in feat_info.get("output_columns", [feat_name]):
                output_col_to_feature[col] = feat_name

        output_cols: List[str] = []
        for name in requested_features or []:
            if name in features:
                feature_info = features[name]
                output_cols.extend(feature_info.get("output_columns", [name]) or [name])
            elif name in output_col_to_feature:
                # Singleton output column requested by name
                output_cols.append(name)
            elif name in df.columns:
                # Allow passing through existing columns by name (rare; mostly for debugging)
                output_cols.append(name)

        # 保留原始列和计算的特征列
        all_cols = list(df.columns) + [c for c in output_cols if c in result_df.columns]
        return result_df[all_cols]

    def load_features_from_requested(
        self,
        df: pd.DataFrame,
        requested_features: Optional[List[str]],
        fit: bool = True,
        *,
        feature_store_dir: Optional[str] = None,
        feature_store_layer: Optional[str] = None,
        feature_store_symbol: Optional[str] = None,
        feature_store_timeframe: Optional[str] = None,
        feature_store_strict: bool = False,
    ) -> pd.DataFrame:
        """
        直接根据请求的特征列表加载特征。

        Args:
            df: 输入 DataFrame
            requested_features: 特征列表
            fit: 是否拟合（研究阶段=True，实盘阶段=False）
            feature_store_strict: 为 True 时，只要提供了四个 FeatureStore 参数就**必须**从
                FeatureStore 命中全部请求输出列；否则抛错，且绝不回退到本地计算。
        """
        result_df = df.copy()
        requested_features = requested_features or []
        if not requested_features:
            return result_df

        if feature_store_strict:
            if not (
                feature_store_dir
                and feature_store_layer
                and feature_store_symbol
                and feature_store_timeframe
            ):
                raise ValueError(
                    "feature_store_strict=True requires feature_store_dir, "
                    "feature_store_layer, feature_store_symbol, feature_store_timeframe"
                )
            if not isinstance(result_df.index, pd.DatetimeIndex) or len(result_df) == 0:
                raise ValueError(
                    "feature_store_strict=True requires a non-empty DataFrame with DatetimeIndex"
                )

        # Optional (Plan B): load from FeatureStore if available.
        # This is best-effort and opt-in; fallback to compute if anything is missing.
        store = None
        spec = None
        if (
            feature_store_dir
            and feature_store_layer
            and feature_store_symbol
            and feature_store_timeframe
            and isinstance(result_df.index, pd.DatetimeIndex)
            and len(result_df) > 0
        ):
            try:
                from src.feature_store.feature_store import (
                    FeatureStore,
                    FeatureStoreSpec,
                )

                store = FeatureStore(feature_store_dir)
                spec = FeatureStoreSpec(
                    layer=feature_store_layer,
                    symbol=feature_store_symbol,
                    timeframe=feature_store_timeframe,
                )
                # Version gate: if FeatureComputer cache_version changed, treat existing FeatureStore partitions as stale.
                expected_cache_version = getattr(self.computer, "cache_version", None)
                months = pd.period_range(
                    start=result_df.index.min(), end=result_df.index.max(), freq="M"
                )
                stale = False
                if expected_cache_version is not None:
                    for p in months:
                        month = f"{p.year:04d}-{p.month:02d}"
                        if not store.has_month(spec, month):
                            continue
                        try:
                            meta = store.read_month_meta(spec, month)
                            md = meta.get("metadata", {}) or {}
                            stored_version = md.get("feature_cache_version")
                            # Tolerate missing version (old builds); only flag
                            # stale when an explicit *different* version is found.
                            if stored_version is not None and stored_version != expected_cache_version:
                                stale = True
                                break
                        except Exception:
                            stale = True
                            break

                if stale:
                    raise ValueError("stale feature store month partition(s)")
                df_store = store.read_range(
                    spec, result_df.index.min(), result_df.index.max()
                )
                if not df_store.empty:
                    # Timezone alignment: FeatureStore may be tz-naive while raw data is UTC
                    result_tz = getattr(result_df.index, 'tz', None)
                    store_tz = getattr(df_store.index, 'tz', None)
                    if store_tz is None and result_tz is not None:
                        # FeatureStore is tz-naive, localize to match result_df
                        df_store.index = df_store.index.tz_localize(result_tz)
                    elif store_tz is not None and result_tz is None:
                        # Result is tz-naive, strip tz from store
                        df_store.index = df_store.index.tz_localize(None)
                    elif store_tz is not None and result_tz is not None and store_tz != result_tz:
                        # Both have tz but different, convert store to result's tz
                        df_store.index = df_store.index.tz_convert(result_tz)
                    df_store = df_store.reindex(result_df.index)
                    features_cfg = self.feature_deps.get("features", {})
                    declared_output_cols = {
                        col
                        for feat_info in features_cfg.values()
                        for col in feat_info.get("output_columns", [])
                    }
                    output_cols: List[str] = []
                    for feature_name in requested_features:
                        if feature_name in features_cfg:
                            output_cols.extend(
                                features_cfg[feature_name].get(
                                    "output_columns", [feature_name]
                                )
                            )
                        elif feature_name in declared_output_cols:
                            output_cols.append(feature_name)

                    # Deduplicate output_cols (two features may declare the same output column)
                    _seen_oc: set = set()
                    _dedup_oc: List[str] = []
                    for _c in output_cols:
                        if _c not in _seen_oc:
                            _seen_oc.add(_c)
                            _dedup_oc.append(_c)
                    output_cols = _dedup_oc

                    if not output_cols:
                        if feature_store_strict:
                            raise ValueError(
                                "FeatureStore strict read: requested_features did not resolve to any "
                                "output columns declared in feature_dependencies.yaml"
                            )
                    # if store already has all needed outputs, return joined frame
                    elif all(c in df_store.columns for c in output_cols):
                        # Only concat columns NOT already present in result_df
                        # to avoid duplicate column names (e.g. cvd_change_features_f
                        # declares output_columns that overlap with raw data columns).
                        new_cols = [c for c in output_cols if c not in result_df.columns]
                        if new_cols:
                            feature_subset = df_store[new_cols]
                            merged = pd.concat([result_df, feature_subset], axis=1)
                        else:
                            merged = result_df
                        print(
                            f"   ✅ FeatureStore hit: {feature_store_symbol}/{feature_store_timeframe} "
                            f"({len(df_store)} rows, {len(output_cols)} feature cols)"
                        )
                        return merged
                    else:
                        _missing = [c for c in output_cols if c not in df_store.columns]
                        if feature_store_strict:
                            raise KeyError(
                                f"FeatureStore strict read: missing columns {_missing} "
                                f"for {feature_store_symbol}/{feature_store_timeframe} "
                                f"(layer={feature_store_layer}); resolved output_cols={len(output_cols)}"
                            )
                        print(
                            f"   ⚠️  FeatureStore partial: {feature_store_symbol}/{feature_store_timeframe} "
                            f"missing {len(_missing)} cols: {_missing[:5]}{'...' if len(_missing)>5 else ''}. "
                            f"Falling back to compute."
                        )
                else:
                    if feature_store_strict:
                        raise FileNotFoundError(
                            f"FeatureStore strict read: no rows in range for "
                            f"{feature_store_symbol}/{feature_store_timeframe} (layer={feature_store_layer})"
                        )
                    print(
                        f"   ⚠️  FeatureStore empty: {feature_store_symbol}/{feature_store_timeframe} "
                        f"(layer={feature_store_layer}). Falling back to compute."
                    )
            except Exception as _fs_err:
                if feature_store_strict:
                    raise
                import sys as _sys
                print(
                    f"⚠️  FeatureStore read failed for "
                    f"{feature_store_symbol}/{feature_store_timeframe} "
                    f"(layer={feature_store_layer}): "
                    f"{type(_fs_err).__name__}: {_fs_err}. "
                    f"Falling back to compute.",
                    file=_sys.stderr,
                    flush=True,
                )

        if feature_store_strict:
            raise RuntimeError(
                "feature_store_strict=True but FeatureStore path did not return merged features"
            )

        features = self.feature_deps.get("features", {})

        # 将 output column 名（如 dtw_shooting_star_dist_w15）自动映射回对应的特征函数
        # 例如：如果某列在某个 feature 的 output_columns 中，则用该 feature 名替换。
        output_col_to_feature: Dict[str, str] = {}
        for feat_name, feat_info in features.items():
            for col in feat_info.get("output_columns", [feat_name]):
                output_col_to_feature[col] = feat_name

        actual_requested: List[str] = []
        for name in requested_features:
            if name in features:
                actual_requested.append(name)
            elif name in output_col_to_feature:
                parent = output_col_to_feature[name]
                if parent not in actual_requested:
                    actual_requested.append(parent)
            else:
                # Unknown name: only keep it if it's already a column in the input.
                # Otherwise, skip it (treat as invalid feature id) to avoid KeyError downstream.
                if name in result_df.columns or name in df.columns:
                    actual_requested.append(name)
                else:
                    print(f"   ⚠️  Unknown feature '{name}' (not in deps and not in df columns); skipping.")

        # 去重但保持顺序
        seen = set()
        dedup_requested: List[str] = []
        for name in actual_requested:
            if name in seen:
                continue
            seen.add(name)
            dedup_requested.append(name)
        actual_requested = dedup_requested

        # 确保所有请求特征的 required_columns 都在 DataFrame 中
        # 收集所有需要的 required_columns
        all_required_columns = set()
        for feature_name in actual_requested:
            if feature_name in features:
                feature_info = features[feature_name]
                required_columns = feature_info.get("required_columns", [])
                all_required_columns.update(required_columns)

        # 检查缺失的 required_columns 并尝试从原始 df 中获取
        missing_required = [
            col for col in all_required_columns if col not in result_df.columns
        ]
        if missing_required:
            for col in missing_required:
                if col in df.columns:
                    # 确保索引对齐
                    try:
                        result_df[col] = df[col].reindex(result_df.index)
                    except Exception:
                        # 如果 reindex 失败，尝试直接赋值（假设索引相同）
                        if len(df) == len(result_df):
                            result_df[col] = df[col].values
                        else:
                            # 如果长度不匹配，尝试使用 loc
                            common_idx = result_df.index.intersection(df.index)
                            if len(common_idx) > 0:
                                result_df.loc[common_idx, col] = df.loc[common_idx, col]

        # Store original indices to filter out any new indices introduced during feature computation
        original_indices = set(df.index)

        result_df = self.computer.compute_features_parallel(
            result_df,
            features,
            actual_requested,
            fit=fit,
        )

        # Attach debug stats for downstream diagnostics & performance monitoring.
        # This includes cache hit info and per-feature timings.
        try:
            result_df.attrs["feature_debug_stats"] = self.computer.drain_debug_stats()
        except Exception:
            pass

        # Filter out any indices that were not in the original input DataFrame
        # This prevents feature computation from introducing overlapping indices
        new_indices = set(result_df.index) - original_indices
        if new_indices:
            print(
                f"     ⚠️  Feature computation introduced {len(new_indices)} new indices, filtering them out"
            )
            if len(new_indices) <= 10:
                print(
                    f"        Examples of new indices: {sorted(list(new_indices))[:5]}"
                )
            # Filter to original indices, but preserve all columns
            result_df = result_df.loc[result_df.index.isin(original_indices)]

        # 验证：确保输出索引与输入索引一致
        if not result_df.index.equals(df.index):
            # 尝试重新对齐索引
            # Use merge/join instead of reindex to preserve values when indices overlap
            # First, try to align by intersection
            common_idx = result_df.index.intersection(df.index)
            if len(common_idx) > 0:
                # Keep the aligned subset
                result_df_aligned = result_df.loc[common_idx].copy()
                # Reindex to full original index (will introduce NaN for missing rows, but preserve existing values)
                result_df = result_df_aligned.reindex(df.index)
            else:
                # No common indices, use reindex (will create all NaN, but at least structure is correct)
                result_df = result_df.reindex(df.index)
            print(f"     ℹ️  Reindexed output to match input index (may introduce NaN for missing rows)")

        # 验证：检查数据类型
        # 允许的 object 类型列（这些列本来就是字符串类型，不需要警告）
        allowed_object_columns = {
            "_symbol",  # 交易对标识符（带下划线）
            "symbol",  # 交易对标识符（不带下划线）
            "trend_direction",  # UP/DOWN label from trend_confidence_f (bus stores ±1)
            "box_regime_label",  # box 结构分类标签: small/mid/big/none
            "dtw_best_match_w15",  # DTW 特征：最佳匹配模式（可能是 "none"）
            "dtw_best_match_w20",
            "dtw_best_match_w25",
        }
        # 允许以这些前缀开头的列
        allowed_object_prefixes = [
            "dtw_best_match_",  # DTW 匹配模式列
            "_symbol",  # 符号相关列
        ]

        for col in result_df.columns:
            try:
                # 检查 col 是否是单个列（Series）还是多个列（DataFrame）
                col_data = result_df[col]
                if isinstance(col_data, pd.DataFrame):
                    # 如果是 DataFrame，跳过（可能是多列特征）
                    continue
                elif isinstance(col_data, pd.Series):
                    # 如果是 Series，检查 dtype
                    if col_data.dtype == "object":
                        # 检查是否在允许列表中
                        if col in allowed_object_columns:
                            continue  # 允许的列，跳过警告

                        # 检查是否匹配允许的前缀
                        is_allowed = False
                        for prefix in allowed_object_prefixes:
                            if col.startswith(prefix):
                                is_allowed = True
                                break

                        if is_allowed:
                            continue  # 允许的列，跳过警告

                        # 检查是否有意外的 object 类型（可能是字符串或其他类型）
                        sample_values = col_data.dropna().head(5)
                        if len(sample_values) > 0:
                            first_val = sample_values.iloc[0]
                            if not isinstance(
                                first_val, (int, float, bool, type(None))
                            ):
                                print(
                                    f"     ⚠️  Warning: Column '{col}' has unexpected dtype 'object' with sample value: {first_val}"
                                )
            except (KeyError, AttributeError, TypeError) as e:
                # 如果无法访问列，跳过
                continue

        # Build requested output columns for the final returned frame.
        #
        # `requested_features` may contain:
        # - feature nodes (e.g. `atr_ratio_f`), OR
        # - specific output column names (e.g. `trade_cluster_absorption_scene_score`)
        #   when semantic singleton expansion is enabled.
        #
        # We keep only explicitly requested outputs (plus original input columns),
        # and do NOT automatically include intermediate dependency outputs.
        output_cols: List[str] = []
        for name in requested_features or []:
            if name in features:
                feature_info = features[name]
                output_cols.extend(feature_info.get("output_columns", [name]) or [name])
            elif name in output_col_to_feature:
                output_cols.append(name)
            elif name in df.columns:
                output_cols.append(name)

        # Auto materialize FeatureStore (wide table) when FeatureStore args were provided:
        # - try read -> if missing -> compute (using FeatureComputer caches) -> write monthly partitions
        if (
            store is not None
            and spec is not None
            and output_cols
            and isinstance(result_df.index, pd.DatetimeIndex)
            and len(result_df) > 0
        ):
            try:
                # Base columns to persist (only keep what exists)
                base_cols = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "_symbol",
                    "symbol",
                    "datetime",
                    "timestamp",
                    "date",
                ]
                base_cols = [c for c in base_cols if c in result_df.columns]
                feat_cols = [c for c in output_cols if c in result_df.columns]

                if base_cols and feat_cols:
                    df_sorted = result_df.sort_index()
                    for period, df_month in df_sorted.groupby(pd.Grouper(freq="M")):
                        if df_month.empty:
                            continue
                        month_str = period.strftime("%Y-%m")
                        overwrite = False
                        if store.has_month(spec, month_str):
                            # If existing partition is missing any required feature cols, overwrite to fill.
                            try:
                                _ = store.read_month(spec, month_str, columns=feat_cols)
                            except Exception:
                                overwrite = True
                            # If existing partition was built with a different FeatureComputer cache_version, overwrite.
                            try:
                                meta = store.read_month_meta(spec, month_str)
                                md = meta.get("metadata", {}) or {}
                                if md.get("feature_cache_version") != getattr(
                                    self.computer, "cache_version", None
                                ):
                                    overwrite = True
                            except Exception:
                                overwrite = True
                        store.write_month(
                            spec,
                            month_str,
                            df_month,
                            base_columns=base_cols,
                            feature_columns=feat_cols,
                            overwrite=overwrite,
                            # When we need to fill missing columns, merge into existing month partition
                            # instead of clobbering previously materialized columns.
                            merge_existing=True,
                            metadata={
                                "auto_materialized": True,
                                "feature_cache_version": getattr(
                                    self.computer, "cache_version", None
                                ),
                            },
                        )
            except Exception:
                # Never break the caller if FeatureStore materialization fails.
                pass

        # Avoid huge allocations from pandas block copies when slicing columns.
        # Also protect against duplicate column names leaking in from upstream merges.
        if result_df.columns.duplicated().any():
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        # Build a stable, de-duplicated column list (preserve order).
        all_cols = list(dict.fromkeys(list(df.columns) + output_cols))
        all_cols = [c for c in all_cols if c in result_df.columns]

        # In most cases result_df already contains exactly these columns; avoid an
        # unnecessary copy that can blow up memory.
        if len(all_cols) == len(result_df.columns) and set(all_cols) == set(
            result_df.columns
        ):
            return result_df

        return result_df.loc[:, all_cols]

    def get_strategy_features(self, strategy_name: str) -> List[str]:
        """
        获取策略的特征列表（包括依赖）

        注意：此方法已废弃，推荐使用目录管理方式。
        此方法仅用于向后兼容，需要 strategy_features.yaml 文件存在。

        Args:
            strategy_name: 策略名称

        Returns:
            feature_list: 特征列表（包括依赖）
        """
        if not self.strategy_config or "strategies" not in self.strategy_config:
            raise ValueError(
                f"Strategy config not loaded. "
                f"Please use directory-based configs instead."
            )

        if strategy_name not in self.strategy_config.get("strategies", {}):
            raise ValueError(f"Unknown strategy: {strategy_name}")

        requested_features = self.strategy_config["strategies"][strategy_name].get(
            "requested_features", []
        )
        return self.resolve_dependencies(requested_features)

    def clear_cache(self, memory: bool = True, disk: bool = False):
        """清除缓存"""
        self.computer.clear_cache(memory=memory, disk=disk)
