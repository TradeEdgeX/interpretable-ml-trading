"""
Entry Filter 公共模块 — backtest 和 live 共用。

职责：
  - 加载 entry_filters.yaml 配置
  - 计算衍生 entry filter 特征（批量 DataFrame / 单 bar dict）
  - 构建条件 mask / 检查单 bar 条件
  - 并联组合：默认 OR；可选 ``combination_mode: and``（见 ``check_entry_filters_or_single``）

数据流：
  backtest: DataFrame 批量 → compute_derived_entry_features() → apply_entry_filters_or()
  live:     Dict 单 bar  → compute_derived_entry_features_single() → check_entry_filters_or_single()
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.config.strategy_layout import resolve_strategy_package_under_root

# ================================================================
# Operator mapping (条件运算符 → 函数)
# ================================================================

_OP_MAP: Dict[str, Callable] = {
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
}

# Scalar version for single-bar checks
_OP_MAP_SCALAR: Dict[str, Callable[[float, float], bool]] = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


# ================================================================
# Config loading
# ================================================================


def load_entry_filters_config(
    strategy: str,
    strategies_root: str = "config/strategies",
    *,
    research: bool = False,
    live_layout: bool = False,
) -> Dict[str, Any]:
    """加载 entry_filters.yaml 配置。

    Args:
        research: True → 读根目录研究文件 (含全部候选 + disabled);
                  False → 读 archetypes/ 生产文件 (默认, backtest/live 用).
        live_layout: ``True`` 不向 ``bad-candidates/`` 回退（与实盘磁盘布局一致）。
    """
    pkg = resolve_strategy_package_under_root(
        Path(strategies_root),
        strategy,
        allow_bad_candidates=not live_layout,
    )
    if research:
        # 研究文件: config/strategies/{strategy}/entry_filters.yaml
        path = pkg / "entry_filters.yaml"
        if not path.exists():
            # fallback to archetypes
            path = pkg / "archetypes" / "entry_filters.yaml"
    else:
        path = pkg / "archetypes" / "entry_filters.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_available_filters(
    entry_cfg: Dict[str, Any],
) -> List[str]:
    """返回 entry_filters.yaml 中所有 enabled=true 的 filter id 列表"""
    ids = ["none"]
    for f in entry_cfg.get("filters") or []:
        if f.get("enabled", True):
            ids.append(f["id"])
    return ids


# ================================================================
# Derived entry features — batch (DataFrame)
# ================================================================


def compute_derived_entry_features(df: pd.DataFrame) -> None:
    """计算衍生 entry filter 特征（从已有列派生的正交维度）。

    三个正交维度：
    1. ef_vol_regime_shift:   波动率 regime 切换（bb_width 5-bar 变化率）
       负值 = 波动率正在下降 → squeeze 中 → 爆发前兆
    2. ef_liquidity_silence:  流动性沉默分位（成交量历史百分位）
       低值 = 市场极度安静 → 参与者在等待
    3. ef_consolidation_bars: 回踩盘整持续 bar 数（连续 was_in_pullback==1 的 bar 数）
       高值 = 盘整越久，蓄势越充分

    NOTE: 这些特征直接添加为 df 的新列（in-place）。
    """
    # --- 1. vol_regime_shift: bb_width 5-bar 变化 ---
    if "bb_width_normalized_pct" in df.columns:
        if "symbol" in df.columns:
            df["ef_vol_regime_shift"] = df.groupby("symbol")[
                "bb_width_normalized_pct"
            ].diff(5)
        else:
            df["ef_vol_regime_shift"] = df["bb_width_normalized_pct"].diff(5)
        df["ef_vol_regime_shift"] = df["ef_vol_regime_shift"].fillna(0.0)

    # --- 2. liquidity_silence: 直接用 vol_percentile_approx ---
    # vol_percentile_approx [0,1], 低 = 成交量极低 = 流动性沉默
    if "vol_percentile_approx" in df.columns:
        df["ef_liquidity_silence"] = df[
            "vol_percentile_approx"
        ]  # NaN = warmup 不足，禁止静默降级为 0.5
    elif "bpc_vol_ratio" in df.columns:
        # fallback: vol_ratio < 0.6 → 低于均量 60% → 沉默
        df["ef_liquidity_silence"] = df["bpc_vol_ratio"].fillna(1.0)

    # --- 3. consolidation_bars: 连续 was_in_pullback==1 的 bar 数 ---
    if "bpc_was_in_pullback" in df.columns:
        wip = df["bpc_was_in_pullback"].astype(float)
        if "symbol" in df.columns:
            cons = []
            for sym, grp in df.groupby("symbol"):
                vals = grp["bpc_was_in_pullback"].values
                cnt = np.zeros(len(vals), dtype=float)
                c = 0
                for i, v in enumerate(vals):
                    if v == 1:
                        c += 1
                    else:
                        c = 0
                    cnt[i] = c
                cons.append(pd.Series(cnt, index=grp.index))
            df["ef_consolidation_bars"] = pd.concat(cons).reindex(df.index)
        else:
            vals = wip.values
            cnt = np.zeros(len(vals), dtype=float)
            c = 0
            for i, v in enumerate(vals):
                if v == 1:
                    c += 1
                else:
                    c = 0
                cnt[i] = c
            df["ef_consolidation_bars"] = cnt
        # 归一化到 [0, 1]（上限 40 bar ≈ 近 7 天 4H bar）
        df["ef_consolidation_bars"] = (df["ef_consolidation_bars"] / 40.0).clip(0, 1)


# ================================================================
# Derived entry features — live (single bar, stateful)
# ================================================================


class DerivedEntryFeatureState:
    """Live 用有状态的衍生特征计算器。

    每收到一根新 bar 的特征 dict，更新内部状态并返回 ef_* 衍生值。

    用法::

        state = DerivedEntryFeatureState()
        # 每根新 bar:
        ef_features = state.update(features_dict)
        # ef_features = {"ef_vol_regime_shift": ..., "ef_liquidity_silence": ..., "ef_consolidation_bars": ...}
    """

    def __init__(self, diff_window: int = 5, consolidation_cap: int = 40):
        self._diff_window = diff_window
        self._consolidation_cap = consolidation_cap
        # bb_width 历史 buffer（最近 diff_window+1 值）
        self._bb_width_history: deque = deque(maxlen=diff_window + 1)
        # 连续 was_in_pullback==1 的计数
        self._consolidation_count: int = 0

    def update(self, features: Dict[str, float]) -> Dict[str, float]:
        """接收当前 bar 特征 dict，返回 ef_* 衍生特征 dict。"""
        result: Dict[str, float] = {}

        # --- 1. ef_vol_regime_shift ---
        bb_val = features.get("bb_width_normalized_pct")
        if bb_val is not None:
            self._bb_width_history.append(float(bb_val))
            if len(self._bb_width_history) > self._diff_window:
                result["ef_vol_regime_shift"] = (
                    self._bb_width_history[-1]
                    - self._bb_width_history[-1 - self._diff_window]
                )
            else:
                result["ef_vol_regime_shift"] = 0.0
        else:
            result["ef_vol_regime_shift"] = 0.0

        # --- 2. ef_liquidity_silence ---
        vol_pct = features.get("vol_percentile_approx")
        if vol_pct is not None:
            result["ef_liquidity_silence"] = float(vol_pct)
        else:
            vol_ratio = features.get("bpc_vol_ratio")
            if vol_ratio is not None:
                result["ef_liquidity_silence"] = float(vol_ratio)
            else:
                result["ef_liquidity_silence"] = 0.5

        # --- 3. ef_consolidation_bars ---
        wip = features.get("bpc_was_in_pullback")
        if wip is not None and float(wip) == 1:
            self._consolidation_count += 1
        else:
            self._consolidation_count = 0
        result["ef_consolidation_bars"] = min(
            self._consolidation_count / self._consolidation_cap, 1.0
        )

        return result

    def reset(self) -> None:
        """重置状态（例如换 symbol 时）。"""
        self._bb_width_history.clear()
        self._consolidation_count = 0


# ================================================================
# Per-filter direction scope (long / short / both)
# ================================================================


def _normalize_filter_direction(filt: Dict[str, Any]) -> Optional[str]:
    """Return 'long', 'short', or None if filter applies to both sides."""
    raw = filt.get("direction")
    if raw is None:
        return None
    side = str(raw).strip().lower()
    if side in ("long", "buy", "1", "+1"):
        return "long"
    if side in ("short", "sell", "-1"):
        return "short"
    return None


def entry_filter_applies_to_direction(
    filt: Dict[str, Any],
    direction: Optional[int],
) -> bool:
    """True when filter should be evaluated for this direction (+1 long / -1 short)."""
    side = _normalize_filter_direction(filt)
    if side is None or direction is None:
        return True
    if direction > 0:
        return side == "long"
    if direction < 0:
        return side == "short"
    return True


def _direction_vacuous_pass_mask(
    df: pd.DataFrame,
    filt: Dict[str, Any],
) -> pd.Series:
    """Rows where a direction-scoped filter does not apply → vacuous pass."""
    side = _normalize_filter_direction(filt)
    if side is None or "entry_direction" not in df.columns:
        return pd.Series(False, index=df.index)
    ed = pd.to_numeric(df["entry_direction"], errors="coerce").fillna(0.0)
    if side == "long":
        return ed != 1.0
    return ed != -1.0


# ================================================================
# Condition mask — batch (DataFrame)
# ================================================================


def _build_mask_from_conditions(
    df: pd.DataFrame,
    conditions: List[Dict[str, Any]],
    silent: bool = False,
) -> pd.Series:
    """从 conditions 列表构建 AND 组合的 boolean mask"""
    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        feat = cond["feature"]
        op_str = cond["operator"]
        val = cond["value"]
        if feat not in df.columns:
            if not silent:
                print(f"   ⚠️  Missing feature '{feat}', condition skipped")
            continue
        op_fn = _OP_MAP.get(op_str)
        if op_fn is None:
            if not silent:
                print(f"   ⚠️  Unknown operator '{op_str}', condition skipped")
            continue
        mask = mask & op_fn(df[feat].astype(float), float(val))
    return mask


# ================================================================
# Condition check — live (single bar dict)
# ================================================================


def check_conditions_single(
    features: Dict[str, float],
    conditions: List[Dict[str, Any]],
) -> bool:
    """检查单个 bar 的特征 dict 是否满足所有 conditions (AND)。

    Returns:
        True if ALL conditions are met, False otherwise.
        Missing features → condition treated as not met.
    """
    for cond in conditions:
        feat = cond["feature"]
        op_str = cond["operator"]
        threshold = float(cond["value"])

        val = features.get(feat)
        if val is None:
            return False  # missing feature → fail

        op_fn = _OP_MAP_SCALAR.get(op_str)
        if op_fn is None:
            return False

        if not op_fn(float(val), threshold):
            return False

    return True


def _or_bundle_ids(entry_cfg: Dict[str, Any]) -> set[str]:
    """Optional filter ids OR'd together; all other filters AND with that bundle."""
    raw = entry_cfg.get("or_bundle_ids") or entry_cfg.get("or_bundle") or []
    if not isinstance(raw, list):
        return set()
    return {str(x).strip() for x in raw if str(x).strip()}


def _filter_passes_single(
    features: Dict[str, float],
    filt: Dict[str, Any],
    *,
    direction: Optional[int],
) -> Optional[bool]:
    """None = filter skipped (no conditions); else pass/fail."""
    conditions = filt.get("conditions", [])
    if not conditions:
        return None
    if not entry_filter_applies_to_direction(filt, direction):
        return True
    return check_conditions_single(features, conditions)


def _live_combo_pass(
    enabled: List[Dict[str, Any]],
    entry_cfg: Dict[str, Any],
    features: Dict[str, float],
    *,
    direction: Optional[int],
) -> bool:
    or_ids = _or_bundle_ids(entry_cfg)
    mode = str(entry_cfg.get("combination_mode") or "or").strip().lower()
    if mode not in {"or", "and"}:
        mode = "or"

    def _applicable(filt: Dict[str, Any]) -> bool:
        """True if *filt* should be evaluated for the current *direction*."""
        return entry_filter_applies_to_direction(filt, direction)

    # Helper: evaluate a single filter's conditions (no direction check —
    # caller has already filtered by applicability).
    def _eval_conditions(filt: Dict[str, Any]) -> Optional[bool]:
        conditions = filt.get("conditions", [])
        if not conditions:
            return None
        return check_conditions_single(features, conditions)

    if or_ids:
        bundle = [f for f in enabled if str(f.get("id", "")).strip() in or_ids]
        rest = [f for f in enabled if str(f.get("id", "")).strip() not in or_ids]

        # Bundle: OR of applicable filters
        bundle_ok: List[bool] = []
        for filt in bundle:
            if not _applicable(filt):
                continue  # OR mode: exclude non-applicable filters
            r = _eval_conditions(filt)
            if r is not None:
                bundle_ok.append(r)
        # Rest: AND of applicable filters; non-applicable → vacuous pass
        rest_ok: List[bool] = []
        for filt in rest:
            if not _applicable(filt):
                continue  # vacuous: don't contribute to AND check
            r = _eval_conditions(filt)
            if r is not None:
                rest_ok.append(r)

        if not bundle_ok and not rest_ok:
            return True
        ok_bundle = any(bundle_ok) if bundle_ok else True
        ok_rest = all(rest_ok) if rest_ok else True
        return ok_bundle and ok_rest

    # ── OR mode: only evaluate applicable filters ──
    if mode == "or":
        per_filter_ok: List[bool] = []
        for filt in enabled:
            if not _applicable(filt):
                continue  # exclude opposite-direction filters from OR
            r = _eval_conditions(filt)
            if r is not None:
                per_filter_ok.append(r)
        if not per_filter_ok:
            return True
        return any(per_filter_ok)

    # ── AND mode: vacuous pass is correct (direction filter should NOT
    #     block opposite-direction entries) ──
    per_filter_ok: List[bool] = []
    for filt in enabled:
        r = _filter_passes_single(features, filt, direction=direction)
        if r is not None:
            per_filter_ok.append(r)

    if not per_filter_ok:
        return True
    return all(per_filter_ok)


def _build_direction_aware_combo(
    filter_triples: List[Tuple[str, pd.Series, Optional[str]]],
    entry_cfg: Dict[str, Any],
    df: pd.DataFrame,
) -> pd.Series:
    """Build direction-aware combination mask from (id, mask, side) triples.

    Direction-scoped filters (side='long'/'short') are only applied to entries
    of the matching direction.  In OR mode this prevents opposite-direction
    vacuous passes from making the combo always-True.
    """
    or_ids = _or_bundle_ids(entry_cfg)
    mode = str(entry_cfg.get("combination_mode") or "or").strip().lower()
    if mode not in ("or", "and"):
        mode = "or"

    ed = pd.to_numeric(df["entry_direction"], errors="coerce").fillna(0.0)

    def _masks_for_direction(target_dir: int) -> List[pd.Series]:
        """Return condition masks for filters applicable to *target_dir*."""
        masks: List[pd.Series] = []
        for _fid, m, side in filter_triples:
            if side is None:
                masks.append(m)
            elif target_dir == 1 and side == "long":
                masks.append(m)
            elif target_dir == -1 and side == "short":
                masks.append(m)
            # else: filter does not apply to this direction -> excluded
        return masks

    # -- or_bundle_ids: bundle OR + rest AND, per-direction --
    if or_ids:
        # Split triples into bundle (OR'd together) and rest (AND'd)
        bundle_triples = [(fid, m, s) for fid, m, s in filter_triples if fid in or_ids]
        rest_triples = [
            (fid, m, s) for fid, m, s in filter_triples if fid not in or_ids
        ]

        def _masks_for(triples: list, target_dir: int) -> List[pd.Series]:
            masks: List[pd.Series] = []
            for _fid, m, side in triples:
                if side is None:
                    masks.append(m)
                elif target_dir == 1 and side == "long":
                    masks.append(m)
                elif target_dir == -1 and side == "short":
                    masks.append(m)
            return masks

        def _or_all(masks: List[pd.Series]) -> pd.Series:
            """OR all masks; empty → True (vacuous pass)."""
            if not masks:
                return pd.Series(True, index=df.index)
            c = pd.Series(False, index=df.index)
            for m in masks:
                c = c | m
            return c

        def _and_all(masks: List[pd.Series]) -> pd.Series:
            """AND all masks; empty → True (vacuous pass)."""
            if not masks:
                return pd.Series(True, index=df.index)
            c = pd.Series(True, index=df.index)
            for m in masks:
                c = c & m
            return c

        def _per_dir(target_dir: int) -> pd.Series:
            bundle_masks = _masks_for(bundle_triples, target_dir)
            rest_masks = _masks_for(rest_triples, target_dir)
            bundle_or = _or_all(bundle_masks)
            rest_and = _and_all(rest_masks)
            return bundle_or & rest_and

        long_c = _per_dir(1)
        short_c = _per_dir(-1)

        combo = pd.Series(False, index=df.index, dtype=bool)
        combo.loc[ed == 1.0] = long_c.loc[ed == 1.0].to_numpy(dtype=bool)
        combo.loc[ed == -1.0] = short_c.loc[ed == -1.0].to_numpy(dtype=bool)
        combo.loc[ed == 0.0] = True
        return combo

    # -- AND mode: vacuous pass is correct (direction filter should NOT block
    #     opposite-direction entries) --
    if mode == "and":
        combo = pd.Series(True, index=df.index)
        for _fid, m, side in filter_triples:
            if side is None:
                combo = combo & m
            elif side == "long":
                combo = combo & (m | (ed != 1.0))
            elif side == "short":
                combo = combo & (m | (ed != -1.0))
        return combo

    # -- OR mode: per-direction evaluation --
    long_masks = _masks_for_direction(1)
    short_masks = _masks_for_direction(-1)

    long_combo = pd.Series(False, index=df.index)
    for m in long_masks:
        long_combo = long_combo | m
    if not long_masks:
        long_combo = pd.Series(True, index=df.index)

    short_combo = pd.Series(False, index=df.index)
    for m in short_masks:
        short_combo = short_combo | m
    if not short_masks:
        short_combo = pd.Series(True, index=df.index)

    combo = pd.Series(False, index=df.index, dtype=bool)
    combo.loc[ed == 1.0] = long_combo.loc[ed == 1.0].to_numpy(dtype=bool)
    combo.loc[ed == -1.0] = short_combo.loc[ed == -1.0].to_numpy(dtype=bool)
    combo.loc[ed == 0.0] = True  # neutral entries always pass
    return combo


# -- Legacy batch combo (kept for reference; new code uses _build_direction_aware_combo) --


def _batch_combo_mask(
    filter_masks: List[Tuple[str, pd.Series]],
    entry_cfg: Dict[str, Any],
    index: pd.Index,
) -> pd.Series:
    """Legacy combo helper — does NOT handle direction scoping.

    Prefer _build_direction_aware_combo for new code.
    """
    or_ids = _or_bundle_ids(entry_cfg)
    if or_ids:
        by_id = {fid: m for fid, m in filter_masks}
        bundle_masks = [by_id[fid] for fid in or_ids if fid in by_id]
        rest_masks = [m for fid, m in filter_masks if fid not in or_ids]
        if not bundle_masks and not rest_masks:
            return pd.Series(True, index=index)
        combo = pd.Series(True, index=index)
        if bundle_masks:
            b = bundle_masks[0]
            for m in bundle_masks[1:]:
                b = b | m
            combo = combo & b
        for m in rest_masks:
            combo = combo & m
        return combo

    mode = str(entry_cfg.get("combination_mode") or "or").strip().lower()
    if mode not in {"or", "and"}:
        mode = "or"

    if mode == "and":
        combo = filter_masks[0][1]
        for _, m in filter_masks[1:]:
            combo = combo & m
        return combo

    combo = pd.Series(False, index=index)
    for _, m in filter_masks:
        combo = combo | m
    return combo


def check_entry_filters_or_single(
    features: Dict[str, float],
    entry_cfg: Dict[str, Any],
    *,
    direction: Optional[int] = None,
) -> bool:
    """Live 单 bar entry_filters.yaml 顶层组合。

    每个 filter：其 ``conditions`` 始终 **内部 AND**（见 ``check_conditions_single``）。
    **多个 filter**：由 ``combination_mode`` 控制并联语义（缺省 ``or``）：

    - ``or`` — 任一 **适用方向** 的 enabled filter 通过 ⇒ 允许入场。
      方向作用域 filter（``direction: long/short``）仅在匹配方向上参与 OR；
      不匹配方向被排除，避免两个反向 filter 互为 vacuous pass 导致全员放行。
    - ``and`` — 所有这类 filter 均需通过 ⇒ 允许入场。
      方向不匹配的 filter 视为 vacuous pass（不阻塞反向信号）。
    - ``or_bundle_ids`` — 列表内 filter **OR** 成一组，与其余 filter **AND**
      （E2a：vol|delta 且 anti-chase）

    若无 enabled filter、或均无有效 conditions ⇒ 放行（等价无门）。

    Args:
        features: 当前 bar 的特征字典（含 ef_* 衍生特征）
        entry_cfg: load_entry_filters_config() 返回的配置
        direction: 交易方向 (+1=long, -1=short, None=不区分)

    Returns:
        True = 允许入场, False = 等待
    """
    filters_list = entry_cfg.get("filters") or []
    enabled = [f for f in filters_list if f.get("enabled", False)]

    if not enabled:
        return True  # no filters → all pass

    return _live_combo_pass(enabled, entry_cfg, features, direction=direction)


# ================================================================
# Apply filters — batch (DataFrame, in-place mutation)
# ================================================================


def apply_entry_filter(
    df: pd.DataFrame,
    filter_name: str,
    entry_cfg: Optional[Dict[str, Any]] = None,
    silent: bool = False,
) -> int:
    """
    Config-driven 入场时机过滤器（单个 filter）。

    从 entry_filters.yaml 读取 filter 定义（conditions），
    将不满足条件的 bar 的 entry_direction 置 0。

    Returns:
        过滤后剩余的入场信号数
    """
    if filter_name == "none":
        return int((df["entry_direction"] != 0).sum())

    n_before = int((df["entry_direction"] != 0).sum())

    # 查找 filter 定义
    filter_def = None
    if entry_cfg:
        for f in entry_cfg.get("filters") or []:
            if f.get("id") == filter_name:
                filter_def = f
                break

    if filter_def is None:
        if not silent:
            print(f"   ⚠️  Filter '{filter_name}' not found in entry_filters.yaml")
        return n_before

    conditions = filter_def.get("conditions", [])
    if not conditions:
        if not silent:
            print(f"   ⚠️  Filter '{filter_name}' has no conditions")
        return n_before

    # 构建 mask
    mask = _build_mask_from_conditions(df, conditions, silent=silent)

    # 应用: 不满足 mask 的行 entry_direction → 0
    df.loc[~mask, "entry_direction"] = 0.0
    n_after = int((df["entry_direction"] != 0).sum())

    if not silent:
        pct = n_after / n_before * 100 if n_before > 0 else 0
        desc = filter_def.get("description", "")
        print(
            f"   🔍 Entry filter '{filter_name}': {n_before} → {n_after} entries ({pct:.1f}% pass)"
        )
        print(f"      {desc}")

    return n_after


def apply_entry_filters_or(
    df: pd.DataFrame,
    entry_cfg: Dict[str, Any],
    silent: bool = False,
) -> int:
    """
    对所有 enabled=true 的 entry filter 并联应用顶层 ``combination_mode``。

    - 每个 filter 内部的 conditions：**AND**
    - 多个 filter：**OR（默认）** 或 **AND**（``combination_mode: and``）
    - 方向作用域 filter（``direction: long/short``）：
      **OR 模式** — 仅在匹配方向上评估，不匹配方向排除出 OR 组合
      **AND 模式** — 不匹配方向视为 vacuous pass（不阻塞反向信号）

    Returns:
        过滤后剩余的入场信号数
    """
    filters_list = entry_cfg.get("filters") or []
    enabled = [f for f in filters_list if f.get("enabled", False)]

    if not enabled:
        if not silent:
            print("   ℹ️  No enabled entry filters, all entries pass")
        return int((df["entry_direction"] != 0).sum())

    mode = str(entry_cfg.get("combination_mode") or "or").strip().lower()
    if mode not in {"or", "and"}:
        mode = "or"

    n_before = int((df["entry_direction"] != 0).sum())

    # Build filter masks (pure conditions, no vacuous pass mixed in).
    # Direction handling is deferred to combo logic.
    _filter_triples: List[Tuple[str, pd.Series, Optional[str]]] = []
    filter_stats = []
    for filt in enabled:
        conditions = filt.get("conditions", [])
        if not conditions:
            continue
        filt_mask = _build_mask_from_conditions(df, conditions, silent=True)
        side = _normalize_filter_direction(filt)
        fid = str(filt.get("id", "")).strip()
        # Stats: for display, show how many entries pass this filter's conditions
        # (entry_direction != 0), taking direction scope into account.
        vacuous = _direction_vacuous_pass_mask(df, filt)
        display_mask = filt_mask | vacuous
        filt_pass = int((display_mask & (df["entry_direction"] != 0)).sum())
        filter_stats.append((fid, filt_pass))
        _filter_triples.append((fid, filt_mask, side))

    if not _filter_triples:
        return n_before

    combo_mask = _build_direction_aware_combo(_filter_triples, entry_cfg, df)
    if _or_bundle_ids(entry_cfg):
        top = "OR-bundle+AND"
    elif mode == "and":
        top = "AND"
    else:
        top = "OR"

    # 应用: 不满足并联 mask 的 bar → entry_direction = 0
    df.loc[~combo_mask, "entry_direction"] = 0.0
    n_after = int((df["entry_direction"] != 0).sum())

    if not silent:
        pct = n_after / n_before * 100 if n_before > 0 else 0
        print(
            f"   🔍 Entry Filter ({top}, {len(_filter_triples)} filters): "
            f"{n_before} → {n_after} entries ({pct:.1f}% pass)"
        )
        for fid, fpass in filter_stats:
            print(f"      ✅ {fid}: {fpass} entries")

    return n_after
