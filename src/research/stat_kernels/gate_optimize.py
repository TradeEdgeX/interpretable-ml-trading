"""Gate rule lift optimization (canonical kernel for research + legacy script)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.research.gate_when import (
    is_feature_allowed_for_gate_deny,
    load_allowed_gate_deny_features,
    parse_gate_when_condition,
)
from src.research.stat_kernels.gate_lift import scan_thresholds_for_lift
from src.research.stat_kernels.plateau import find_stable_lift_plateau
from src.research.stat_kernels.robustness import (
    UnifiedOptimizationConfig,
    compute_robustness_score,
)
from src.time_series_model.archetype import GateRule

_REQUIRE_POS_EFFECT_WARNED = False


def _load_allowed_gate_deny_features_for_strategy(strategy: Optional[str]) -> List[str]:
    return load_allowed_gate_deny_features(strategy)


def _is_feature_allowed_for_gate_deny(feature: str, patterns: List[str]) -> bool:
    return is_feature_allowed_for_gate_deny(feature, patterns)


def _parse_gate_when_condition(when: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    return parse_gate_when_condition(when)


def optimize_gate_rule_unified(
    df: pd.DataFrame,
    rule: GateRule,
    label_col: str = "is_good",
    config: Optional[UnifiedOptimizationConfig] = None,
    step: float = 0.05,
    rr_col: Optional[str] = None,
    strategy: Optional[str] = None,
    allowed_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    统一的门控规则优化函数

    Args:
        df: 包含特征和标签的 DataFrame
        rule: GateRule 对象
        label_col: 标签列名
        config: 优化配置
        step: 阈值扫描步长
        rr_col: 连续收益列名 (e.g. forward_rr / bpc_impulse_return_atr).
            仅当 config.require_positive_effect=True 时使用,
            用于剔除 mean_rr(allow) <= mean_rr(deny) 的病态 gate.
        strategy: 策略名; 用于读取 features_gate.yaml 的白名单 (覆盖特征层
            deny 允许清单). 仅当 allowed_features 未显式传入时使用.
        allowed_features: 显式传入的白名单 fnmatch patterns; 空 = 允许所有.

    Returns:
        优化结果
    """
    if config is None:
        config = UnifiedOptimizationConfig()

    # 从规则的 when 条件中提取特征和运算符
    when = rule.when
    feature_col, operator, current_threshold = _parse_gate_when_condition(when)

    if feature_col is None or operator is None:
        return {
            "rule_id": rule.id,
            "status": "skip",
            "reason": f"Could not parse rule when condition: {when}",
        }

    # 检查特征是否存在
    if feature_col not in df.columns:
        return {
            "rule_id": rule.id,
            "feature": feature_col,
            "status": "skip",
            "reason": f"Feature {feature_col} not found in DataFrame",
        }

    # features_gate.yaml 白名单守门:
    # locked / frozen / promote_never_disable 规则是用户显式决策, 永远保留;
    # 其余规则只有在特征命中策略白名单时才允许进入优化循环.
    _is_protected_user_rule = bool(
        getattr(rule, "locked", False)
        or getattr(rule, "frozen", False)
        or getattr(rule, "promote_never_disable", False)
    )
    if not _is_protected_user_rule:
        _whitelist = (
            allowed_features
            if allowed_features is not None
            else _load_allowed_gate_deny_features_for_strategy(strategy)
        )
        if _whitelist and not _is_feature_allowed_for_gate_deny(
            feature_col, _whitelist
        ):
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "status": "skip",
                "reason": (
                    f"Feature {feature_col} not in features_gate.yaml "
                    f"allowed_gate_deny_features (strategy={strategy!r}); "
                    "gate is execution-layer only."
                ),
            }

    # 确定阈值范围
    # 对于 quantile 特征，范围是 [0, 1]
    if "_pct" in feature_col or "quantile" in feature_col:
        threshold_range = (0.05, 0.95)  # 避免边界值
    else:
        # 使用数据分位数确定范围
        q_low = df[feature_col].quantile(0.05)
        q_high = df[feature_col].quantile(0.95)
        threshold_range = (q_low, q_high)
        step = max(step, (q_high - q_low) / 20)  # 自适应步长

    # 扫描阈值
    results = scan_thresholds_for_lift(
        df, feature_col, operator, threshold_range, step, label_col
    )

    # 先进行lift筛选（可行性检查）
    valid_results = [
        r
        for r in results
        if r["lift"] >= config.min_lift
        and config.min_pass_rate <= r["pass_rate_all"] <= config.max_pass_rate
        and r.get("n_good", 0) >= config.min_samples_good
        and r.get("n_bad", 0) >= config.min_samples_bad
    ]

    # ── require_positive_effect: 连续收益口径补护栏 ──
    # binary lift > 0 并不等价于 mean_rr(allow) > mean_rr(deny), 因为 range_deny
    # 形式的规则可能把中腰砍掉而留下两端尾部 (尾部 good 比例高但亏损大). 当配置
    # require_positive_effect 时, 额外要求候选阈值的连续 effect >= -tol.
    if config.require_positive_effect and valid_results:
        if rr_col is None or rr_col not in df.columns:
            # rr_col 不可用时直接 skip 本护栏, 打印一次警告 (用全局 flag 避免刷屏)
            global _REQUIRE_POS_EFFECT_WARNED  # noqa: PLW0603
            try:
                _REQUIRE_POS_EFFECT_WARNED
            except NameError:
                _REQUIRE_POS_EFFECT_WARNED = False
            if not _REQUIRE_POS_EFFECT_WARNED:
                print(
                    f"    ⚠️ require_positive_effect=True 但 rr_col={rr_col!r} 不可用, 本次 skip 此护栏"
                )
                _REQUIRE_POS_EFFECT_WARNED = True
        else:
            _rr_vals = df[rr_col].to_numpy(dtype=float, copy=False)
            _feat_vals = df[feature_col].to_numpy(dtype=float, copy=False)
            _feat_valid = ~np.isnan(_feat_vals)
            _kept: list = []
            _skipped_neg_effect = 0
            for r in valid_results:
                _th = r.get("threshold")
                if _th is None or not np.isfinite(_th):
                    _kept.append(r)
                    continue
                if operator == "lt":
                    _deny = _feat_valid & (_feat_vals < _th)
                elif operator == "le":
                    _deny = _feat_valid & (_feat_vals <= _th)
                elif operator == "gt":
                    _deny = _feat_valid & (_feat_vals > _th)
                elif operator == "ge":
                    _deny = _feat_valid & (_feat_vals >= _th)
                else:
                    _kept.append(r)
                    continue
                _allow = _feat_valid & ~_deny
                if not _deny.any() or not _allow.any():
                    _kept.append(r)
                    continue
                _rr_allow_mean = float(np.nanmean(_rr_vals[_allow]))
                _rr_deny_mean = float(np.nanmean(_rr_vals[_deny]))
                _effect = _rr_allow_mean - _rr_deny_mean
                r["mean_rr_allow"] = _rr_allow_mean
                r["mean_rr_deny"] = _rr_deny_mean
                r["mean_effect"] = _effect
                if _effect < -float(config.positive_effect_tol):
                    _skipped_neg_effect += 1
                    continue
                _kept.append(r)
            if _skipped_neg_effect > 0:
                print(
                    f"    🛡️  require_positive_effect: 跳过 {_skipped_neg_effect} 条 mean effect<=0 候选 "
                    f"(feat={feature_col}, rr={rr_col})"
                )
            valid_results = _kept

    if not valid_results:
        # =========================================================================
        # Hard Gate NaN Lift 例外通道
        # 语义：Hard Gate 的合法性来源是"结构稳定性 + 执行风险厌恶"，不是 lift
        # 只有同时满足所有严格条件，才允许 lift=NaN 的 Hard Gate 进入 robustness 仲裁
        # =========================================================================
        if config.allow_hard_nan_lift and rule.tag and "gate_" in rule.id:
            # 检查是否有符合 NaN lift 例外条件的阈值
            nan_lift_candidates = []
            for r in results:
                # 条件 1: lift 是 NaN（pass_rate_bad 极低）
                if not np.isfinite(r.get("lift", 0)):
                    # 条件 2: pass_rate_bad 必须极低
                    if r.get("pass_rate_bad", 1.0) > config.nan_lift_max_pass_rate_bad:
                        continue
                    # 条件 3: pass_rate_good 必须足够高
                    if r.get("pass_rate_good", 0) < config.nan_lift_min_pass_rate_good:
                        continue
                    # 条件 4: 覆盖率必须足够
                    n_valid = r.get("n_valid", 0)
                    n_all = len(df)
                    coverage = n_valid / n_all if n_all > 0 else 0
                    if coverage < config.nan_lift_min_coverage:
                        continue
                    nan_lift_candidates.append(r)

            if nan_lift_candidates:
                # 在候选中找 robustness 最高的
                best_nan_result = None
                best_nan_robustness = -1

                for r in nan_lift_candidates:
                    robustness = compute_robustness_score(
                        df, feature_col, operator, r["threshold"], label_col, config
                    )
                    # 条件 5: robustness 必须足够高
                    if robustness.overall_score >= config.nan_lift_min_robustness:
                        if robustness.overall_score > best_nan_robustness:
                            best_nan_robustness = robustness.overall_score
                            best_nan_result = {
                                **r,
                                "robustness_score": robustness.to_dict(),
                                "recommended_threshold": r["threshold"],
                                "recommended_threshold_type": "robust_but_unproven",
                            }

                if best_nan_result:
                    # 条件 6: 检查阈值区间宽度（排除点估计）
                    # 找到所有通过条件的阈值范围
                    valid_thresholds = [
                        r["threshold"]
                        for r in nan_lift_candidates
                        if compute_robustness_score(
                            df, feature_col, operator, r["threshold"], label_col, config
                        ).overall_score
                        >= config.nan_lift_min_robustness
                    ]
                    if len(valid_thresholds) >= 2:
                        interval_width = max(valid_thresholds) - min(valid_thresholds)
                        if interval_width >= config.nan_lift_min_plateau_width:
                            return {
                                "rule_id": rule.id,
                                "feature": feature_col,
                                "operator": operator,
                                "current_threshold": current_threshold,
                                "status": "robust_but_unproven",  # 明确标记语义
                                "eligibility": "deny_only",  # 只能作为 deny-only safety gate
                                "robustness_selection": True,
                                "nan_lift_exception": True,
                                "nan_lift_reason": "Hard Gate with pass_rate_bad < 1%, structural stability validated",
                                "interval_width": interval_width,
                                **best_nan_result,
                                "scan_results": results,
                            }

        return {
            "rule_id": rule.id,
            "feature": feature_col,
            "status": "no_valid_threshold",
            "reason": "No threshold meets basic lift/pass_rate requirements",
            "scan_results": results,
        }

    # 寻找稳定的平台区间
    # ❗ Bug fix: 必须传入 valid_results，不能在不可执行域上建立 plateau
    # ❗ 问题 4 修复: 传入实际步长，保证连续性判断一致
    stable_plateau = find_stable_lift_plateau(valid_results, config, actual_step=step)

    if stable_plateau is None:
        # ❗ 设计决策点：Hard gate 没有 plateau 时是否允许 fallback
        # strict_hard=True: 不允许 fallback，Hard gate 必须有结构支撑
        # strict_hard=False: 允许 fallback 到 robustness 单点（过渡态）
        if config.strict_hard:
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "operator": operator,
                "current_threshold": current_threshold,
                "status": "no_stable_plateau_strict",
                "reason": "Hard gate requires stable plateau (strict mode enabled)",
                "scan_results": results,
            }

        # 找不到稳定平台，使用robustness导向的选择策略
        # 在满足基础条件的阈值中选择robustness最高的
        best_result = None
        best_robustness_score = -1

        for r in valid_results:
            robustness = compute_robustness_score(
                df, feature_col, operator, r["threshold"], label_col, config
            )
            if robustness.overall_score > best_robustness_score:
                best_robustness_score = robustness.overall_score
                best_result = {
                    **r,
                    "robustness_score": robustness.to_dict(),
                    "recommended_threshold": r["threshold"],
                    "recommended_threshold_type": "robust_fallback",  # ❗ Bug fix #3
                }

        if best_result:
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "operator": operator,
                "current_threshold": current_threshold,
                "status": "no_stable_plateau",
                "robustness_selection": True,
                "fallback_warning": "Hard gate (weak mode): robustness fallback enabled",
                **best_result,
                "scan_results": results,
            }
        else:
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "status": "no_robust_threshold",
                "reason": "No robust threshold found even with basic requirements met",
                "scan_results": results,
            }

    # 在稳定平台内选择最稳健的阈值
    plateau_results = stable_plateau["interval_details"]
    best_result_in_plateau = None
    best_robustness_in_plateau = -1

    for r in plateau_results:
        robustness = compute_robustness_score(
            df, feature_col, operator, r["threshold"], label_col, config
        )
        if robustness.overall_score > best_robustness_in_plateau:
            best_robustness_in_plateau = robustness.overall_score
            best_result_in_plateau = {
                **r,
                "robustness_score": robustness.to_dict(),
                "recommended_threshold": r["threshold"],
                "recommended_threshold_type": "plateau_best",  # ❗ Bug fix #3
            }

    # 如果平台内的最佳点与中位数不同，提供选择依据
    if best_result_in_plateau["threshold"] != stable_plateau["plateau_mid"]:
        # 比较两者，选择robustness更好的
        # 找到mid附近的结果（使用更宽松的匹配）
        mid_candidates = [
            r
            for r in plateau_results
            if abs(r["threshold"] - stable_plateau["plateau_mid"]) < step * 1.5
        ]
        if mid_candidates:
            mid_result = min(
                mid_candidates,
                key=lambda r: abs(r["threshold"] - stable_plateau["plateau_mid"]),
            )
            mid_robustness = compute_robustness_score(
                df, feature_col, operator, mid_result["threshold"], label_col, config
            )

            if best_robustness_in_plateau > mid_robustness.overall_score:
                recommended_threshold = best_result_in_plateau["threshold"]
                recommended_threshold_type = "plateau_best"  # ❗ Bug fix #3
            else:
                recommended_threshold = stable_plateau["plateau_mid"]
                recommended_threshold_type = "plateau_mid"  # ❗ Bug fix #3
                best_result_in_plateau = {
                    **mid_result,
                    "robustness_score": mid_robustness.to_dict(),
                    "recommended_threshold": stable_plateau["plateau_mid"],
                    "recommended_threshold_type": "plateau_mid",
                }
        else:
            # 如果找不到mid附近的结果，使用best_result_in_plateau
            recommended_threshold = best_result_in_plateau["threshold"]
            recommended_threshold_type = "plateau_best"  # ❗ Bug fix #3
    else:
        recommended_threshold = best_result_in_plateau["threshold"]
        recommended_threshold_type = "plateau_best"  # ❗ Bug fix #3

    return {
        "rule_id": rule.id,
        "feature": feature_col,
        "operator": operator,
        "current_threshold": current_threshold,
        "status": "stable_plateau_found",
        "robustness_selection": True,
        **stable_plateau,
        "recommended_threshold": recommended_threshold,
        "recommended_threshold_type": recommended_threshold_type,  # ❗ Bug fix #3
        "best_result_in_plateau": best_result_in_plateau,
        "scan_results": results,
    }
