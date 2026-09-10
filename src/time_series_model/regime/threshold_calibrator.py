"""Tier-0 regime threshold calibration — 纯函数核心。

职责:
    - 在标注好的 features parquet 上扫一个慢变量阈值（例如 ``tpc_semantic_chop``）
      找出稳定 plateau；
    - 将新 plateau 与各 strategy ``regime.yaml`` 中的 ``last_calibration.plateau`` 比较；
    - 通过 ``plateau_stability.decide_plateau_update`` 输出 ADOPT / KEEP / ALERT 决策。

不做 IO（除读取 parquet/yaml）。原子写入与 ACK 在
``scripts/regime_threshold_calibrate.py`` driver 中完成。
"""

from __future__ import annotations

import copy
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from scripts.plateau_stability import (
    PlateauRange,
    decide_plateau_update,
    plateau_range_from_dict,
)


# ---------------------------------------------------------------------------
# Plateau scan (类似 locked_prefilter_parquet_tune._suggest_parquet_bindings_coordinate)
# ---------------------------------------------------------------------------

_OPS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
}


def _quantile_candidates(series: pd.Series, n: int) -> List[float]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(s) < 10:
        return sorted({float(s.median())}) if len(s) else []
    qs = np.linspace(0.05, 0.95, max(5, min(n, 50)))
    return sorted({float(x) for x in s.quantile(qs)})


def _lift_at_threshold(
    df: pd.DataFrame,
    feature: str,
    operator_str: str,
    threshold: float,
    label_col: str,
    baseline_bad: float,
    *,
    min_pass_rate: float,
    max_pass_rate: float,
) -> float:
    op_func = _OPS.get(operator_str)
    if op_func is None or feature not in df.columns:
        return float("-inf")
    mask = op_func(df[feature].astype(float), float(threshold))
    pr = float(mask.mean())
    if pr < min_pass_rate or pr > max_pass_rate or mask.sum() < 5:
        return float("-inf")
    sub = df.loc[mask, label_col]
    if len(sub) == 0:
        return float("-inf")
    bad = float((sub == 0).mean())
    return float(baseline_bad - bad)


@dataclass
class ChopPlateauResult:
    feature: str
    operator: str
    plateau: PlateauRange
    scores: List[Tuple[float, float]] = field(default_factory=list)
    baseline_bad_rate: float = 0.0
    label_col: str = "success_no_rr_extreme"
    n_rows: int = 0


def scan_chop_plateau(
    df: pd.DataFrame,
    *,
    feature: str = "tpc_semantic_chop",
    operator_str: str = "<=",
    label_col: str = "success_no_rr_extreme",
    scan_points: int = 25,
    plateau_band_fraction: float = 0.02,
    min_pass_rate: float = 0.01,
    max_pass_rate: float = 0.99,
) -> Optional[ChopPlateauResult]:
    """对 feature 扫一段 plateau。返回稳定区间（None=无足够数据）。

    plateau 定义：score 与 max score 差距 ≤ ``plateau_band_fraction * (max - min)``
    的所有阈值连成一段，取首尾作为 [start, end]，中点作为 mid。
    """
    if feature not in df.columns or label_col not in df.columns:
        return None
    baseline_bad = float((df[label_col] == 0).mean())
    candidates = _quantile_candidates(df[feature], scan_points)
    if not candidates:
        return None
    scores: List[Tuple[float, float]] = []
    for thr in candidates:
        s = _lift_at_threshold(
            df,
            feature,
            operator_str,
            thr,
            label_col,
            baseline_bad,
            min_pass_rate=min_pass_rate,
            max_pass_rate=max_pass_rate,
        )
        scores.append((float(thr), float(s)))
    valid = [(t, s) for t, s in scores if s > float("-inf")]
    if not valid:
        return None
    best = max(s for _, s in valid)
    span = best - min(s for _, s in valid)
    eps = max(1e-12, span * plateau_band_fraction) if span > 0 else 0.0
    band = [t for t, s in valid if s >= best - eps]
    if not band:
        return None
    start, end = float(min(band)), float(max(band))
    mid = (start + end) / 2.0
    plateau = PlateauRange(start=start, end=end, mid=mid)
    return ChopPlateauResult(
        feature=feature,
        operator=operator_str,
        plateau=plateau,
        scores=scores,
        baseline_bad_rate=baseline_bad,
        label_col=label_col,
        n_rows=len(df),
    )


# ---------------------------------------------------------------------------
# Regime YAML I/O & rule lookup
# ---------------------------------------------------------------------------


def load_regime_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def find_rule(
    raw_regime: Dict[str, Any],
    *,
    feature: str,
    operator_str: str,
) -> Optional[Dict[str, Any]]:
    """在 ``rules`` 顶层查找 (feature, operator) 匹配的规则（不递归 any_of）。"""
    rules = raw_regime.get("rules") or []
    for r in rules:
        if not isinstance(r, dict):
            continue
        if (
            str(r.get("feature", "")) == feature
            and str(r.get("operator", "")) == operator_str
        ):
            return r
    return None


def get_current_value(
    raw_regime: Dict[str, Any], *, feature: str, operator_str: str
) -> Optional[float]:
    r = find_rule(raw_regime, feature=feature, operator_str=operator_str)
    if r is None:
        return None
    v = r.get("value")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_last_plateau(
    raw_regime: Dict[str, Any], *, feature: str, operator_str: str
) -> Optional[PlateauRange]:
    """从 last_calibration.plateaus[{feature,operator}].plateau 读出上一次 plateau。"""
    last_cal = raw_regime.get("last_calibration") or {}
    plateaus = last_cal.get("plateaus") or []
    for entry in plateaus:
        if not isinstance(entry, dict):
            continue
        if (
            str(entry.get("feature", "")) == feature
            and str(entry.get("operator", "")) == operator_str
        ):
            return plateau_range_from_dict(entry.get("plateau"))
    return None


def build_updated_regime(
    raw_regime: Dict[str, Any],
    *,
    feature: str,
    operator_str: str,
    chosen_value: float,
    new_plateau: PlateauRange,
    timestamp_iso: str,
    data_source: str,
    decision_reason: str,
    action: str,
) -> Dict[str, Any]:
    """生成更新后的 regime dict（不写盘）。"""
    out = copy.deepcopy(raw_regime)
    rule = find_rule(out, feature=feature, operator_str=operator_str)
    if rule is None:
        raise ValueError(
            f"regime.yaml 中没有匹配的 rule: feature={feature}, op={operator_str}"
        )
    if action == "ADOPT":
        rule["value"] = float(chosen_value)

    last_cal = out.get("last_calibration")
    if not isinstance(last_cal, dict):
        last_cal = {}
        out["last_calibration"] = last_cal
    last_cal["timestamp"] = timestamp_iso
    last_cal["data_source"] = data_source
    plateaus = last_cal.get("plateaus")
    if not isinstance(plateaus, list):
        plateaus = []
        last_cal["plateaus"] = plateaus
    plateaus[:] = [
        p
        for p in plateaus
        if not (
            isinstance(p, dict)
            and str(p.get("feature", "")) == feature
            and str(p.get("operator", "")) == operator_str
        )
    ]
    plateaus.append(
        {
            "feature": feature,
            "operator": operator_str,
            "plateau": {
                "start": float(new_plateau.start),
                "end": float(new_plateau.end),
                "mid": float(new_plateau.mid),
            },
            "action": action,
            "reason": decision_reason,
        }
    )
    return out


# ---------------------------------------------------------------------------
# 多策略 atomic calibration proposal
# ---------------------------------------------------------------------------


@dataclass
class StrategyCalibration:
    strategy: str
    regime_yaml_path: Path
    parquet_path: Path
    raw_regime: Dict[str, Any] = field(default_factory=dict)
    current_value: Optional[float] = None
    last_plateau: Optional[PlateauRange] = None
    new_plateau: Optional[PlateauRange] = None
    decision: Optional[Dict[str, Any]] = None
    updated_regime: Optional[Dict[str, Any]] = None
    skipped_reason: Optional[str] = None


def calibrate_strategies(
    items: Iterable[StrategyCalibration],
    *,
    feature: str = "tpc_semantic_chop",
    operator_str: str = "<=",
    label_col: str = "success_no_rr_extreme",
    scan_points: int = 25,
    policy: str = "keep_if_no_overlap",
    timestamp_iso: str,
) -> List[StrategyCalibration]:
    """对每个 strategy 跑 plateau scan + stability 决策；返回 in-place 填充的列表。"""
    out: List[StrategyCalibration] = []
    for item in items:
        item.raw_regime = load_regime_yaml(item.regime_yaml_path)
        if not item.raw_regime:
            item.skipped_reason = f"regime.yaml not found: {item.regime_yaml_path}"
            out.append(item)
            continue
        item.current_value = get_current_value(
            item.raw_regime, feature=feature, operator_str=operator_str
        )
        item.last_plateau = get_last_plateau(
            item.raw_regime, feature=feature, operator_str=operator_str
        )
        if not item.parquet_path.exists():
            item.skipped_reason = f"labeled parquet not found: {item.parquet_path}"
            out.append(item)
            continue
        df = pd.read_parquet(item.parquet_path)
        scan = scan_chop_plateau(
            df,
            feature=feature,
            operator_str=operator_str,
            label_col=label_col,
            scan_points=scan_points,
        )
        if scan is None:
            item.skipped_reason = (
                f"plateau scan returned no valid range for {feature} "
                f"(parquet={item.parquet_path})"
            )
            out.append(item)
            continue
        item.new_plateau = scan.plateau

        current_value = (
            float(item.current_value)
            if item.current_value is not None
            else float(scan.plateau.mid)
        )
        decision = decide_plateau_update(
            old=item.last_plateau,
            new=scan.plateau,
            current_value=current_value,
            policy=policy,
        )
        item.decision = decision
        item.updated_regime = build_updated_regime(
            item.raw_regime,
            feature=feature,
            operator_str=operator_str,
            chosen_value=float(decision["chosen_value"]),
            new_plateau=scan.plateau,
            timestamp_iso=timestamp_iso,
            data_source=str(item.parquet_path),
            decision_reason=str(decision.get("reason", "")),
            action=str(decision.get("action", "ALERT")),
        )
        out.append(item)
    return out
