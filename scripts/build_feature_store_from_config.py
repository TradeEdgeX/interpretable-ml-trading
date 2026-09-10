#!/usr/bin/env python3
"""
Model-agnostic FeatureStore builder.

This is intended to sit *above* tree/nn:
- build once (monthly partitions + warmup)
- tree and nn both read from the same FeatureStore dataset (layer)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.data_tools.universe_config import load_universe_config  # noqa: E402
from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec  # noqa: E402
from src.feature_store.layer_health import (  # noqa: E402
    format_skew_report,
    scan_layer_columns,
    summarize_column_skew,
)
from src.feature_store.layer_naming import resolve_layer_name  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.time_series_model.strategy_config import StrategyConfigLoader  # noqa: E402


def _load_feature_store_build_config(
    cfg_dir: Path, *, features_yaml: Path | None = None
):
    """Load enough strategy config for FeatureStore builds.

    Tree strategies have the full ``features.yaml`` + ``labels.yaml`` +
    ``model.yaml`` package and can use ``StrategyConfigLoader``. Multi-leg
    strategies are feature-only at this layer, so they only need
    ``features.yaml`` and optional ``meta.yaml`` to materialize FeatureStore.
    """
    try:
        loader = StrategyConfigLoader(
            cfg_dir,
            features_override=features_yaml,
        )
        return loader.load()
    except FileNotFoundError as exc:
        features_path = features_yaml or (cfg_dir / "features.yaml")
        if not features_path.exists():
            raise
        missing_text = str(exc)
        if "labels.yaml" not in missing_text and "model.yaml" not in missing_text:
            raise
        features_raw = yaml.safe_load(features_path.read_text(encoding="utf-8")) or {}
        fp = features_raw.get("feature_pipeline", {})
        requested = fp.get("requested_features", []) if isinstance(fp, dict) else []
        if not isinstance(requested, list) or not requested:
            raise ValueError(
                f"{features_path} must define feature_pipeline.requested_features"
            ) from exc
        meta_path = cfg_dir / "meta.yaml"
        meta_raw = (
            yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            if meta_path.exists()
            else {}
        )
        meta = meta_raw.get("strategy", meta_raw) if isinstance(meta_raw, dict) else {}
        print(
            "   ℹ️  Feature-only config detected; building FeatureStore from "
            f"{features_path}"
        )
        return SimpleNamespace(
            features=SimpleNamespace(requested_features=[str(x) for x in requested]),
            meta=meta if isinstance(meta, dict) else {},
        )


def _get_expected_output_columns(features_cfg: dict, requested_features: list) -> set:
    """Return the set of output column names from all requested features."""
    cols: set = set()
    for feat_name in requested_features:
        if feat_name in features_cfg:
            for col in features_cfg[feat_name].get("output_columns", [feat_name]):
                cols.add(col)
        else:
            cols.add(feat_name)
    return cols


def _find_missing_features(
    features_cfg: dict, requested_features: list, existing_columns: set
) -> list:
    """Find requested features whose output columns are not fully present."""
    missing = []
    for feat_name in requested_features:
        if feat_name in features_cfg:
            outputs = set(features_cfg[feat_name].get("output_columns", [feat_name]))
            if not outputs.issubset(existing_columns):
                missing.append(feat_name)
        else:
            if feat_name not in existing_columns:
                missing.append(feat_name)
    return missing


def _find_donor_months(
    store: "FeatureStore",
    spec: FeatureStoreSpec,
    months_needed: list,
    root_dir: Path,
) -> dict:
    """For months not in current layer, find donor layers with the same data."""
    if not months_needed:
        return {}
    donors: dict = {}
    for layer_dir in sorted(
        root_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if (
            not layer_dir.is_dir()
            or layer_dir.name == spec.layer
            or layer_dir.name.startswith(".")
        ):
            continue
        check_dir = layer_dir / spec.symbol / spec.timeframe
        if not check_dir.exists():
            continue
        donor_spec = FeatureStoreSpec(
            layer=layer_dir.name, symbol=spec.symbol, timeframe=spec.timeframe
        )
        for m in months_needed:
            if m not in donors and store.has_month(donor_spec, m):
                donors[m] = donor_spec
        if len(donors) == len(months_needed):
            break
    return donors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build monthly FeatureStore from a config directory."
    )
    p.add_argument(
        "--config", required=True, help="Config directory containing features.yaml."
    )
    p.add_argument(
        "--features-yaml",
        default=None,
        help="Feature manifest path (default: {config}/features.yaml). "
        "Use config/strategies/_shared/features_all.yaml for full ~940-col layer.",
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols. If not provided, will use --universe-config if specified.",
    )
    p.add_argument(
        "--universe-config",
        default=None,
        help="Path to universe config YAML (e.g., config/download/crypto_4h_token_universe_groups.yaml). "
        "If provided and --symbols is not set, will load all symbols from the config.",
    )
    p.add_argument(
        "--universe-set",
        default="starter_a",
        help="Universe set name to use from universe config (default: starter_a).",
    )
    p.add_argument(
        "--universe-groups",
        default=None,
        help="Comma-separated groups to include (e.g., 'highcap,alt'). If not specified, includes all groups.",
    )
    p.add_argument("--timeframe", required=True, help="Timeframe (e.g., 240T).")
    p.add_argument("--data-path", default="data/parquet_data")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--root", default="feature_store", help="FeatureStore root dir.")
    p.add_argument(
        "--layer",
        default=None,
        help="FeatureStore layer (dataset id). If not specified, auto-generated from config content. "
        "You can pass a versioned name like heavy_v6 for manual invalidation.",
    )
    p.add_argument(
        "--warmup-months", type=int, default=3
    )  # 3 months for 540-bar percentile window
    p.add_argument("--warmup-bars", type=int, default=0)
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete existing layer data and rebuild from scratch. "
        "Without this flag, existing months are skipped.",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        default=True,
        help="Allow partial tick data (some symbols may not have data for full range). "
        "Missing months are skipped instead of raising an error. Default: True.",
    )
    p.add_argument(
        "--no-reuse",
        action="store_true",
        default=False,
        help="Disable cross-layer reuse. By default, missing months are copied from "
        "other layers with same symbol/timeframe before computing new features.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallelize across symbols (ProcessPool via per-symbol subprocess). "
        "Default 1 (serial). Use 2–8 on multi-core research hosts. "
        "Does not change feature values.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config).resolve()
    features_yaml: Path | None = None
    if args.features_yaml:
        features_yaml = Path(args.features_yaml)
        if not features_yaml.is_absolute():
            features_yaml = (Path.cwd() / features_yaml).resolve()
    cfg = _load_feature_store_build_config(cfg_dir, features_yaml=features_yaml)

    # Resolve symbols: from --symbols or --universe-config
    if args.symbols:
        symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    elif args.universe_config:
        # Load from universe config
        universe_cfg = load_universe_config(args.universe_config)
        groups = (
            [g.strip() for g in args.universe_groups.split(",") if g.strip()]
            if args.universe_groups
            else None
        )
        symbols = universe_cfg.resolve_symbols_usdt(
            universe_set=args.universe_set, groups=groups
        )
        print(
            f"   📋 Loaded {len(symbols)} symbols from universe config: {args.universe_config}"
        )
        if groups:
            print(f"   📋 Groups: {', '.join(groups)}")
        print(
            f"   📋 Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}"
        )
    else:
        raise ValueError(
            "Either --symbols or --universe-config must be provided. "
            "Use --universe-config to load all symbols from config/download/crypto_4h_token_universe_groups.yaml"
        )

    if not symbols:
        raise ValueError("No symbols resolved. Check --symbols or --universe-config.")

    root = Path(args.root).resolve()
    store = FeatureStore(root)

    # Auto-generate layer name if not specified (unified handling for both CLI and direct script calls)
    features_file = "features.yaml"
    if features_yaml is not None:
        features_file = features_yaml.name
    layer = resolve_layer_name(args.layer, cfg_dir, features_file=features_file)

    # Force rebuild: delete existing layer data
    if args.force_rebuild:
        layer_path = root / layer
        if layer_path.exists():
            import shutil

            print(f"   🗑️  --force-rebuild: Deleting existing layer '{layer}'...")
            shutil.rmtree(layer_path)
            print(f"   ✅ Deleted: {layer_path}")
        else:
            print(
                f"   ℹ️  --force-rebuild: Layer '{layer}' does not exist, nothing to delete."
            )

    warmup_months = max(0, int(args.warmup_months))
    warmup_bars = max(0, int(args.warmup_bars))

    # IMPORTANT: disable FeatureComputer's own monthly cache so warmup context can flow across month boundaries.
    feature_loader = StrategyFeatureLoader(use_monthly_cache=False)

    # 💡 动态注入 freq 参数：从 strategy meta.yaml 的 timeframe 读取
    # 注意：cfg.meta 直接对应 meta.yaml 的 strategy 节点内容
    meta_timeframe = (cfg.meta or {}).get("timeframe")

    if meta_timeframe:
        print(f"   ℹ️  Detected strategy.timeframe from meta.yaml: {meta_timeframe}")
        # 注入到需要 freq 参数的特征配置中
        feature_deps = feature_loader.feature_deps.get("features", {})
        freq_required_features = [
            "vpin_base_aligned_features_f",
            "trade_cluster_base_aligned_features_f",
            "trade_cluster_semantic_scores_f",
        ]
        injected_count = 0
        for feat_name in freq_required_features:
            if feat_name in feature_deps:
                compute_params = feature_deps[feat_name].setdefault(
                    "compute_params", {}
                )
                if "freq" not in compute_params:
                    compute_params["freq"] = meta_timeframe
                    print(f"   ✅ Injected freq='{meta_timeframe}' to {feat_name}")
                    injected_count += 1
                else:
                    # 已有配置，不覆盖（保留用户显式配置的优先级）
                    print(
                        f"   ℹ️  {feat_name} already has freq='{compute_params['freq']}', skipping"
                    )
        if injected_count == 0 and freq_required_features:
            print(
                f"   ℹ️  No freq injection needed (features not in requested list or already configured)"
            )
    else:
        print(
            "   ⚠️  No strategy.timeframe found in meta.yaml, freq parameter will not be injected"
        )

    feature_cache_version = getattr(feature_loader.computer, "cache_version", None)
    requested = cfg.features.requested_features

    # Compute expected output columns for incremental feature detection
    features_cfg = feature_loader.feature_deps.get("features", {})
    expected_output_cols = _get_expected_output_columns(features_cfg, requested)

    # Global statistics
    stats = {
        "symbols_processed": 0,
        "symbols_failed": 0,
        "months_skipped": 0,
        "months_built": 0,
        "months_incremental": 0,
        "months_reused": 0,
        "months_failed": 0,
        "failed_symbols": [],
        "failed_months": [],
    }

    # Pre-check: verify all symbols have tick data for requested range
    if args.start_date and args.end_date:
        print("\n🔍 Pre-checking tick data availability...")
        expected_months = (
            pd.date_range(start=args.start_date, end=args.end_date, freq="MS")
            .strftime("%Y-%m")
            .tolist()
        )
        tick_data_path = Path(args.data_path)
        all_missing = {}
        for sym in symbols:
            missing = []
            for m in expected_months:
                tick_file = tick_data_path / f"{sym}_{m}.parquet"
                if not tick_file.exists():
                    missing.append(m)
            if missing:
                all_missing[sym] = missing

        if all_missing:
            if args.allow_partial:
                print(
                    "   ⚠️  Partial tick data detected (some symbols listed later than start date):"
                )
                for sym, months in all_missing.items():
                    months_str = ", ".join(months[:5])
                    if len(months) > 5:
                        months_str += f"... (+{len(months) - 5} more)"
                    print(f"     - {sym}: missing {len(months)} month(s): {months_str}")
                print(
                    "   ℹ️  --allow-partial: missing months will be skipped automatically\n"
                )
            else:
                error_lines = ["\n❌ Missing tick data detected!"]
                error_lines.append(
                    f"   Requested range: {args.start_date} to {args.end_date}"
                )
                error_lines.append(f"   Data path: {args.data_path}\n")
                for sym, months in all_missing.items():
                    months_str = ", ".join(months[:5])
                    if len(months) > 5:
                        months_str += f"... (+{len(months) - 5} more)"
                    error_lines.append(
                        f"   - {sym}: missing {len(months)} month(s): {months_str}"
                    )
                error_lines.append("\n💡 Please convert the missing tick data first:")
                error_lines.append(
                    "   mlbot data convert --pattern '<SYMBOL>-aggTrades-*.zip'"
                )
                error_lines.append(
                    "\n💡 Or use --allow-partial to skip missing months."
                )
                raise ValueError("\n".join(error_lines))
        print("   ✅ All symbols have complete tick data for requested range\n")

    workers = max(1, int(getattr(args, "workers", 1) or 1))
    if workers > 1 and len(symbols) > 1:
        # Symbol-level fan-out: each child rebuilds one symbol with --workers 1.
        # Avoids pickling StrategyFeatureLoader; FeatureStore months are independent.
        import subprocess
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(
            f"\n🚀 Parallel FeatureStore build: {len(symbols)} symbols × workers={workers}"
        )
        script = str(Path(__file__).resolve())

        def _one(sym: str) -> tuple[str, int]:
            cmd = [
                sys.executable,
                script,
                "--config",
                str(cfg_dir),
                "--timeframe",
                str(args.timeframe),
                "--data-path",
                str(args.data_path),
                "--root",
                str(root),
                "--layer",
                str(layer),
                "--symbols",
                sym,
                "--workers",
                "1",
                "--warmup-months",
                str(warmup_months),
                "--warmup-bars",
                str(warmup_bars),
            ]
            if features_yaml is not None:
                cmd.extend(["--features-yaml", str(features_yaml)])
            if args.start_date:
                cmd.extend(["--start-date", str(args.start_date)])
            if args.end_date:
                cmd.extend(["--end-date", str(args.end_date)])
            if args.force_rebuild:
                cmd.append("--force-rebuild")
            if args.no_reuse:
                cmd.append("--no-reuse")
            if args.allow_partial:
                cmd.append("--allow-partial")
            print(f"  → launch {sym}")
            rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
            return sym, rc

        fails: list[str] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_one, sym): sym for sym in symbols}
            for fut in as_completed(futs):
                sym, rc = fut.result()
                if rc != 0:
                    fails.append(sym)
                    print(f"  ❌ child failed: {sym} rc={rc}")
                else:
                    print(f"  ✅ child done: {sym}")
        if fails:
            raise SystemExit(
                f"Parallel FeatureStore build failed for: {', '.join(fails)}"
            )
        print(
            f"\n✅ Parallel build finished for {len(symbols)} symbols (layer={layer})"
        )
        return

    for sym_idx, sym in enumerate(symbols, 1):
        print(f"\n{'='*60}")
        print(f"📊 Processing symbol {sym_idx}/{len(symbols)}: {sym}")
        print(f"{'='*60}")

        try:
            # Calculate actual start date including warmup period
            actual_start = args.start_date
            if warmup_months > 0 and args.start_date:
                actual_start_ts = pd.to_datetime(args.start_date) - pd.DateOffset(
                    months=warmup_months
                )
                actual_start = actual_start_ts.strftime("%Y-%m-%d")
                print(
                    f"  🔄 Loading data from {actual_start} (warmup {warmup_months} months before {args.start_date})"
                )

            df_raw = load_raw_data(
                data_path=args.data_path,
                symbol=sym,
                start_date=actual_start,
                end_date=args.end_date,
                timeframe=args.timeframe,
            )
            if df_raw.empty:
                print(f"  ⚠️  No raw data loaded for symbol={sym}, skipping")
                stats["symbols_failed"] += 1
                stats["failed_symbols"].append((sym, "No raw data"))
                continue
            if "symbol" not in df_raw.columns:
                df_raw["symbol"] = sym
            df_raw = df_raw.sort_index()

            spec = FeatureStoreSpec(
                layer=str(layer), symbol=str(sym), timeframe=str(args.timeframe)
            )
            monthly_groups = df_raw.groupby(pd.Grouper(freq="M"))
            base_cols = ["open", "high", "low", "close", "volume", "_symbol", "symbol"]

            # Parse start_date and end_date for month filtering.
            # NOTE: df_raw index is normalized to UTC (tz-aware) by load_raw_data().
            # Use tz-aware timestamps here to avoid tz-naive vs tz-aware comparison errors.
            start_ts = (
                pd.to_datetime(args.start_date, utc=True) if args.start_date else None
            )
            end_ts = pd.to_datetime(args.end_date, utc=True) if args.end_date else None

            # Count months to process
            all_months = []
            for period, df_month in monthly_groups:
                if df_month.empty:
                    continue
                month_start = df_month.index.min()
                month_end = df_month.index.max()
                if start_ts is not None and month_end < start_ts:
                    continue
                if end_ts is not None and month_start > end_ts:
                    continue
                all_months.append(period.strftime("%Y-%m"))

            # Classify months: complete / partial (need incremental) / new
            complete_months = []
            partial_months = []  # (month, missing_features)
            new_months = []
            for m in all_months:
                if store.has_month(spec, m):
                    try:
                        meta = store.read_month_meta(spec, m)
                        existing_cols = set(meta.get("columns", []))
                    except Exception:
                        existing_cols = set()
                    missing = _find_missing_features(
                        features_cfg, requested, existing_cols
                    )
                    if missing:
                        partial_months.append((m, missing))
                    else:
                        complete_months.append(m)
                else:
                    new_months.append(m)

            stats["months_skipped"] += len(complete_months)

            # Repair metadata for complete months missing feature_cache_version.
            # Older builds may not have written this field; patching the sidecar
            # JSON here ensures the version gate in train pipeline won't
            # incorrectly mark these months as stale.
            if feature_cache_version and complete_months:
                _repaired = 0
                for _cm in complete_months:
                    try:
                        _meta = store.read_month_meta(spec, _cm)
                        _md = _meta.get("metadata", {}) or {}
                        if _md.get("feature_cache_version") != feature_cache_version:
                            _md["feature_cache_version"] = feature_cache_version
                            if "config_dir" not in _md:
                                _md["config_dir"] = str(cfg_dir)
                            _meta["metadata"] = _md
                            _meta_path = (
                                root
                                / layer
                                / sym
                                / str(args.timeframe)
                                / f"{_cm}.meta.json"
                            )
                            _meta_path.write_text(
                                json.dumps(_meta, ensure_ascii=False, indent=2)
                            )
                            _repaired += 1
                    except Exception:
                        pass
                if _repaired:
                    print(
                        f"  \U0001f527 Repaired metadata for {_repaired} month(s) (added feature_cache_version)"
                    )

            # Find donor layers for new months (cross-layer reuse)
            donor_map: dict = {}
            if new_months and not args.no_reuse and not args.force_rebuild:
                # --force-rebuild: 跳过 donor 复用，全部从头计算，避免从旧损坏层搬运数据
                donor_map = _find_donor_months(store, spec, new_months, root)

            if complete_months:
                print(
                    f"  \u23ed\ufe0f  Complete {len(complete_months)} month(s): "
                    f"{', '.join(complete_months[:5])}"
                    f"{'...' if len(complete_months) > 5 else ''}"
                )
            if partial_months:
                sample_missing = partial_months[0][1]
                print(
                    f"  \U0001f504 Incremental {len(partial_months)} month(s): "
                    f"+{len(sample_missing)} features "
                    f"({', '.join(sample_missing[:3])}"
                    f"{'...' if len(sample_missing) > 3 else ''})"
                )
            if new_months:
                reuse_count = len([m for m in new_months if m in donor_map])
                fresh_count = len(new_months) - reuse_count
                parts = []
                if reuse_count:
                    parts.append(f"{reuse_count} reusable")
                if fresh_count:
                    parts.append(f"{fresh_count} from scratch")
                print(
                    f"  \U0001f528 New {len(new_months)} month(s): {', '.join(parts)}"
                )
            if not complete_months and not partial_months and not new_months:
                print(f"  \u26a0\ufe0f  No months to process for {sym}")

            # Process each month with error handling
            for period, df_month in monthly_groups:
                if df_month.empty:
                    continue

                month_start = df_month.index.min()
                month_end = df_month.index.max()

                # Filter months by start_date and end_date if provided
                if start_ts is not None and month_end < start_ts:
                    continue
                if end_ts is not None and month_start > end_ts:
                    continue

                month_str = period.strftime("%Y-%m")

                # Determine action: skip / incremental / reuse+incremental / full build
                features_to_compute = requested
                merge_mode = False

                if store.has_month(spec, month_str):
                    try:
                        meta = store.read_month_meta(spec, month_str)
                        existing_cols = set(meta.get("columns", []))
                    except Exception:
                        existing_cols = set()
                    missing_feats = _find_missing_features(
                        features_cfg, requested, existing_cols
                    )
                    if not missing_feats:
                        continue  # complete, skip
                    features_to_compute = missing_feats
                    merge_mode = True
                    print(
                        f"\n  \U0001f504 Incremental {sym} {month_str}: "
                        f"+{len(missing_feats)} features "
                        f"({', '.join(missing_feats[:3])}"
                        f"{'...' if len(missing_feats) > 3 else ''})"
                    )
                elif month_str in donor_map:
                    # Copy from donor layer first
                    donor_spec = donor_map[month_str]
                    try:
                        donor_df = store.read_month(donor_spec, month_str)

                        # ── Donor OHLC 完整性校验: 拒绝坏数据传播 ──
                        _donor_ok = True
                        _donor_rows = len(donor_df)
                        for _dc in ["close", "high", "low"]:
                            if _dc in donor_df.columns:
                                _dn = int(donor_df[_dc].isna().sum())
                                if _donor_rows > 0 and _dn / _donor_rows > 0.5:
                                    print(
                                        f"  ⚠️  Donor {donor_spec.layer}/{month_str} "
                                        f"has corrupted {_dc} (NaN={_dn}/{_donor_rows}), "
                                        f"skipping donor → build from scratch"
                                    )
                                    _donor_ok = False
                                    break
                        if not _donor_ok:
                            print(f"\n  📅 Building {sym} {month_str}...")
                        else:
                            store.write_month(
                                spec,
                                month_str,
                                donor_df,
                                base_columns=list(donor_df.columns),
                                feature_columns=[],
                                overwrite=True,
                            )
                            existing_cols = set(donor_df.columns)
                            missing_feats = _find_missing_features(
                                features_cfg, requested, existing_cols
                            )
                            if not missing_feats:
                                stats["months_reused"] += 1
                                print(
                                    f"  \U0001f4cb Reused {sym} {month_str} "
                                    f"from layer {donor_spec.layer}"
                                )
                                continue
                            features_to_compute = missing_feats
                            merge_mode = True
                            print(
                                f"\n  \U0001f4cb Reused {sym} {month_str} from {donor_spec.layer}, "
                                f"+{len(missing_feats)} features to compute"
                            )
                    except Exception as e:
                        print(
                            f"  \u26a0\ufe0f  Donor reuse failed for {month_str}: {e}, "
                            f"building from scratch"
                        )
                else:
                    print(f"\n  \U0001f4c5 Building {sym} {month_str}...")

                try:
                    if warmup_months > 0:
                        warmup_start = pd.Timestamp(month_start) - pd.DateOffset(
                            months=warmup_months
                        )
                        df_window = df_raw.loc[
                            (df_raw.index >= warmup_start) & (df_raw.index <= month_end)
                        ]
                    elif warmup_bars > 0:
                        pos_end = df_raw.index.searchsorted(month_start, side="left")
                        pos_start = max(0, pos_end - warmup_bars)
                        df_window = df_raw.iloc[pos_start:].loc[:month_end]
                    else:
                        df_window = df_raw.loc[
                            (df_raw.index >= month_start) & (df_raw.index <= month_end)
                        ]

                    df_feats_window = feature_loader.load_features_from_requested(
                        df_window, requested_features=features_to_compute, fit=True
                    )
                    if "symbol" not in df_feats_window.columns:
                        df_feats_window["symbol"] = sym
                    df_feats_month = df_feats_window.loc[
                        (df_feats_window.index >= month_start)
                        & (df_feats_window.index <= month_end)
                    ]

                    # Determine feature columns to write
                    if merge_mode:
                        # Only write newly computed feature columns
                        feature_cols = []
                        for feat_name in features_to_compute:
                            if feat_name in features_cfg:
                                for col in features_cfg[feat_name].get(
                                    "output_columns", [feat_name]
                                ):
                                    if (
                                        col in df_feats_month.columns
                                        and col not in base_cols
                                    ):
                                        feature_cols.append(col)
                            elif (
                                feat_name in df_feats_month.columns
                                and feat_name not in base_cols
                            ):
                                feature_cols.append(feat_name)
                        feature_cols = list(dict.fromkeys(feature_cols))
                    else:
                        feature_cols = [
                            c for c in df_feats_month.columns if c not in base_cols
                        ]

                    # ── OHLC 完整性校验: 及早暴露 cache 污染 ──
                    _ohlc_cols = [
                        c
                        for c in ["close", "high", "low"]
                        if c in df_feats_month.columns
                    ]
                    if _ohlc_cols:
                        _total_rows = len(df_feats_month)
                        for _oc in _ohlc_cols:
                            _nan_ct = int(df_feats_month[_oc].isna().sum())
                            _nan_ratio = _nan_ct / _total_rows if _total_rows > 0 else 0
                            if _nan_ratio > 0.5:
                                raise ValueError(
                                    f"OHLC 完整性校验失败: {sym}/{month_str} "
                                    f"{_oc} NaN={_nan_ct}/{_total_rows} ({_nan_ratio:.0%}). "
                                    f"可能是 timeframe cache 污染, 请删除 cache/timeframes/{sym}_*.parquet 后重试"
                                )

                    store.write_month(
                        spec,
                        month_str,
                        df_feats_month,
                        base_columns=base_cols,
                        feature_columns=feature_cols,
                        overwrite=False,
                        merge_existing=merge_mode,
                        metadata={
                            "config_dir": str(cfg_dir),
                            "warmup_months": warmup_months,
                            "warmup_bars": warmup_bars,
                            "requested_features": features_to_compute,
                            "feature_cache_version": feature_cache_version,
                        },
                    )
                    if merge_mode:
                        stats["months_incremental"] += 1
                    else:
                        stats["months_built"] += 1
                    label = "incremental" if merge_mode else "built"
                    print(f"  \u2705 Successfully {label} {sym} {month_str}")
                except Exception as e:
                    stats["months_failed"] += 1
                    error_msg = str(e)
                    stats["failed_months"].append((sym, month_str, error_msg))
                    print(f"  \u274c Failed to build {sym} {month_str}: {error_msg}")
                    import traceback

                    print(f"     Traceback: {traceback.format_exc()}")
                    # Continue to next month instead of crashing

            stats["symbols_processed"] += 1
            n_actions = len(partial_months) + len(new_months)
            print(
                f"\u2705 Completed {sym}: {len(complete_months)} skipped, {n_actions} processed"
            )

        except Exception as e:
            stats["symbols_failed"] += 1
            error_msg = str(e)
            stats["failed_symbols"].append((sym, error_msg))
            print(f"  ❌ Failed to process symbol {sym}: {error_msg}")
            import traceback

            print(f"     Traceback: {traceback.format_exc()}")
            # Continue to next symbol instead of crashing

    # Print summary statistics
    print(f"\n{'='*60}")
    print("📊 Build Summary")
    print(f"{'='*60}")
    print(f"  ✅ Symbols processed: {stats['symbols_processed']}/{len(symbols)}")
    print(f"  ❌ Symbols failed: {stats['symbols_failed']}")
    print(f"  ⏭️  Months skipped (complete): {stats['months_skipped']}")
    print(f"  🔄 Months incremental (features added): {stats['months_incremental']}")
    print(f"  📋 Months reused (from other layers): {stats['months_reused']}")
    print(f"  🔨 Months built (from scratch): {stats['months_built']}")
    print(f"  ❌ Months failed: {stats['months_failed']}")

    if stats["failed_symbols"]:
        print(f"\n  ⚠️  Failed symbols ({len(stats['failed_symbols'])}):")
        for sym, error in stats["failed_symbols"][:10]:
            print(f"     - {sym}: {error[:100]}")
        if len(stats["failed_symbols"]) > 10:
            print(f"     ... and {len(stats['failed_symbols']) - 10} more")

    if stats["failed_months"]:
        print(f"\n  \u26a0\ufe0f  Failed months ({len(stats['failed_months'])}):")
        for sym, month, error in stats["failed_months"][:10]:
            print(f"     - {sym} {month}: {error[:100]}")
        if len(stats["failed_months"]) > 10:
            print(f"     ... and {len(stats['failed_months']) - 10} more")

    if stats["months_failed"] > 0 or stats["symbols_failed"] > 0:
        print(
            f"\n  💡 Tip: Re-run the command to retry failed months (existing months will be skipped)"
        )

    skew = summarize_column_skew(str(layer), scan_layer_columns(root, str(layer)))
    print(f"\n{'='*60}")
    print("📐 Layer column health")
    print(f"{'='*60}")
    print(format_skew_report(skew))
    if skew.skewed:
        stats["layer_column_skew"] = True
        stats["layer_min_cols"] = skew.min_cols
        stats["layer_max_cols"] = skew.max_cols
        stats["layer_thin_months"] = len(skew.thin)
        print(
            "\n  ⚠️  Layer is column-skewed. Wide IC / tree_full intersection "
            "will collapse until thin months are incrementally rebuilt."
        )

    meta_path = root / f"{layer}.meta.json"
    prior_symbols: list[str] = []
    if meta_path.exists():
        try:
            prior = json.loads(meta_path.read_text(encoding="utf-8"))
            prior_symbols = [str(s) for s in (prior.get("symbols") or [])]
        except (json.JSONDecodeError, OSError):
            prior_symbols = []
    symbols_out = sorted(set(prior_symbols) | set(symbols))

    # Save metadata with statistics
    meta = {
        "config_dir": str(cfg_dir),
        "timeframe": str(args.timeframe),
        "symbols": symbols_out,
        "layer": str(layer),
        "warmup_months": warmup_months,
        "warmup_bars": warmup_bars,
        "last_write_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_stats": stats,
        "column_health": {
            "min_cols": skew.min_cols,
            "max_cols": skew.max_cols,
            "n_files": skew.n_files,
            "skewed": skew.skewed,
            "thin_months": len(skew.thin),
        },
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n✅ Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
