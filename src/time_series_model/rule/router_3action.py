from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

try:
    from src.time_series_model.rule.regime import (
        PhysicsRegimeConfig,
        classify_regime,
    )

    _PHYSICS_REGIME_AVAILABLE = True
except ImportError:
    _PHYSICS_REGIME_AVAILABLE = False


@dataclass(frozen=True)
class Rule3ActionConfig:
    """
    Pure rule router based only on nnmultihead heads (no extra strategy features).

    It maps each timestamp into one of:
      - NO_TRADE
      - MEAN
      - TREND

    Assumptions:
    - pred_mfe_atr/pred_mae_atr/pred_t_to_mfe are in "training space" by default (log1p),
      unless preds_in_log1p=False.
    """

    # Input columns
    pred_dir_prob_col: str = "pred_dir_prob"
    pred_mfe_col: str = "pred_mfe_atr"
    pred_mae_col: str = "pred_mae_atr"
    pred_ttm_col: str = "pred_t_to_mfe"

    # Decision thresholds (in ATR units after inverse-transform)
    mfe_min: float = 0.4
    eff_min: float = 1.05  # mfe/(mae+eps) minimum to trade

    # Trend conditions
    dir_conf_trend_min: float = 0.25  # abs(p-0.5)*2
    mfe_trend_min: float = 0.8
    ttm_trend_min: float = 8.0
    # Trend confirm mode:
    # - "and" (legacy): dir_conf AND mfe AND ttm
    # - "or": dir_conf AND (mfe OR ttm)
    trend_confirm_mode: str = "and"

    # Mean conditions
    eff_mean_min: float = 1.15
    ttm_mean_max: float = 12.0

    eps: float = 1e-9


@dataclass(frozen=True)
class QualityScoreConfig:
    """
    Trade quality score for Regime->Execution gating.
    """

    quality_trend_min: float = 0.6
    quality_mean_min: float = 0.8
    quality_te_min: float = 0.45

    # If True, use eff * dir_conf. Otherwise use eff only.
    use_dir_conf: bool = True


def _dir_conf(p: np.ndarray) -> np.ndarray:
    # p in [0,1] -> confidence in [0,1]
    return np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _maybe_expm1(x: np.ndarray, *, preds_in_log1p: bool) -> np.ndarray:
    if not preds_in_log1p:
        return x
    # model heads are strictly positive; expm1 keeps 0->0
    # Clip to avoid overflow when preds are extreme (e.g. bad model weights / untrained run).
    # expm1(15) ~= 3.27e6 which is already "effectively infinite" for our thresholds.
    return np.expm1(np.clip(x, 0.0, 15.0))


def _apply_temperature_scaling(
    p: np.ndarray, *, temperature: float, clip: tuple[float, float]
) -> np.ndarray:
    temp = float(temperature) if temperature else 1.0
    lo, hi = clip
    p = np.clip(p, lo, hi)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-(logit / temp)))


def _apply_isotonic_scaling(
    p: np.ndarray, *, clip: tuple[float, float], config: Dict[str, Any]
) -> np.ndarray:
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "isotonic calibration requires scikit-learn. "
            "Install scikit-learn or use temperature scaling."
        ) from exc
    lo, hi = clip
    p = np.clip(p, lo, hi)
    x = np.array(config.get("x", []), dtype=float)
    y = np.array(config.get("y", []), dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        raise ValueError("isotonic calibration requires matching x/y arrays.")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    return np.clip(iso.transform(p), lo, hi)


def _apply_linear_calibration(
    x: np.ndarray, *, scale: float, bias: float, clip: tuple[float, float]
) -> np.ndarray:
    lo, hi = clip
    return np.clip(x * float(scale) + float(bias), lo, hi)


def _apply_router_calibration(
    p: np.ndarray,
    mfe_p: np.ndarray,
    mae_p: np.ndarray,
    ttm_p: np.ndarray,
    *,
    calibration: Dict[str, Any],
    preds_in_log1p: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not calibration:
        return p, mfe_p, mae_p, ttm_p
    space = str(calibration.get("space", "log1p")).strip().lower()
    if space == "log1p" and not preds_in_log1p:
        raise ValueError(
            "router calibration expects log1p preds, but preds_in_log1p=False."
        )
    # dir prob calibration
    dir_cfg = calibration.get("dir_prob") or {}
    method = str(dir_cfg.get("method", "temperature")).lower()
    clip = tuple(dir_cfg.get("clip", [1e-6, 1 - 1e-6]))
    if method == "temperature":
        p = _apply_temperature_scaling(
            p, temperature=float(dir_cfg.get("temperature", 1.0)), clip=clip
        )
    elif method == "isotonic":
        p = _apply_isotonic_scaling(p, clip=clip, config=dir_cfg)
    else:
        raise ValueError(f"Unknown dir_prob calibration method: {method}")

    # linear bias correction in log space for regression heads
    for key, arr in (
        ("mfe_atr", mfe_p),
        ("mae_atr", mae_p),
        ("t_to_mfe", ttm_p),
    ):
        cfg = calibration.get(key) or {}
        if not cfg:
            continue
        arr[:] = _apply_linear_calibration(
            arr,
            scale=float(cfg.get("scale", 1.0)),
            bias=float(cfg.get("bias", 0.0)),
            clip=tuple(cfg.get("clip", [0.0, 15.0])),
        )
    return p, mfe_p, mae_p, ttm_p


def _slope(arr: np.ndarray, *, window: int = 3) -> np.ndarray:
    if window <= 0:
        return np.zeros_like(arr, dtype=float)
    prev = np.roll(arr, window)
    prev[:window] = np.nan
    return arr - prev


def compute_trade_quality(
    df: pd.DataFrame,
    *,
    cfg: Rule3ActionConfig = Rule3ActionConfig(),
    score_cfg: QualityScoreConfig = QualityScoreConfig(),
    preds_in_log1p: bool = True,
    calibration: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    missing = [
        c
        for c in [
            cfg.pred_dir_prob_col,
            cfg.pred_mfe_col,
            cfg.pred_mae_col,
            cfg.pred_ttm_col,
        ]
        if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required pred columns for quality: {missing}")

    p = pd.to_numeric(df[cfg.pred_dir_prob_col], errors="coerce").to_numpy(dtype=float)
    mfe_p = pd.to_numeric(df[cfg.pred_mfe_col], errors="coerce").to_numpy(dtype=float)
    mae_p = pd.to_numeric(df[cfg.pred_mae_col], errors="coerce").to_numpy(dtype=float)
    ttm_p = pd.to_numeric(df[cfg.pred_ttm_col], errors="coerce").to_numpy(dtype=float)
    if calibration:
        p, mfe_p, mae_p, ttm_p = _apply_router_calibration(
            p,
            mfe_p,
            mae_p,
            ttm_p,
            calibration=calibration,
            preds_in_log1p=preds_in_log1p,
        )

    mfe = _maybe_expm1(mfe_p, preds_in_log1p=preds_in_log1p)
    mae = _maybe_expm1(mae_p, preds_in_log1p=preds_in_log1p)
    ttm = _maybe_expm1(ttm_p, preds_in_log1p=preds_in_log1p)
    dconf = _dir_conf(np.clip(p, 0.0, 1.0))

    eff = np.where(
        np.isfinite(mfe) & np.isfinite(mae),
        mfe / (mae + float(cfg.eps)),
        0.0,
    )

    if score_cfg.use_dir_conf:
        quality = eff * dconf
    else:
        quality = eff

    out = pd.DataFrame(index=df.index)
    out["trade_quality"] = quality.astype(float)
    out["mfe_atr"] = mfe
    out["mae_atr"] = mae
    out["t_to_mfe"] = ttm
    out["eff"] = eff
    out["dir_conf"] = dconf
    return out


def compute_mode_3action(
    df: pd.DataFrame,
    *,
    cfg: Rule3ActionConfig = Rule3ActionConfig(),
    preds_in_log1p: bool = True,
    calibration: Optional[Dict[str, Any]] = None,
    out_col: str = "mode",
) -> pd.DataFrame:
    """
    Return a dataframe with columns:
      - mode (str): NO_TRADE/MEAN/TREND
      - mode_action (int): 0/1/2
      - optional derived diagnostics: mfe, mae, ttm, eff, dir_conf
    """
    missing = [
        c
        for c in [
            cfg.pred_dir_prob_col,
            cfg.pred_mfe_col,
            cfg.pred_mae_col,
            cfg.pred_ttm_col,
        ]
        if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required pred columns for rule router: {missing}")

    p = pd.to_numeric(df[cfg.pred_dir_prob_col], errors="coerce").to_numpy(dtype=float)
    mfe_p = pd.to_numeric(df[cfg.pred_mfe_col], errors="coerce").to_numpy(dtype=float)
    mae_p = pd.to_numeric(df[cfg.pred_mae_col], errors="coerce").to_numpy(dtype=float)
    ttm_p = pd.to_numeric(df[cfg.pred_ttm_col], errors="coerce").to_numpy(dtype=float)
    if calibration:
        p, mfe_p, mae_p, ttm_p = _apply_router_calibration(
            p,
            mfe_p,
            mae_p,
            ttm_p,
            calibration=calibration,
            preds_in_log1p=preds_in_log1p,
        )

    mfe = _maybe_expm1(mfe_p, preds_in_log1p=preds_in_log1p)
    mae = _maybe_expm1(mae_p, preds_in_log1p=preds_in_log1p)
    ttm = _maybe_expm1(ttm_p, preds_in_log1p=preds_in_log1p)

    dconf = _dir_conf(np.clip(p, 0.0, 1.0))
    # Avoid invalid divisions from NaN/Inf.
    eff = np.where(
        np.isfinite(mfe) & np.isfinite(mae),
        mfe / (mae + float(cfg.eps)),
        0.0,
    )

    # Default: NO_TRADE
    mode = np.full(len(df), "NO_TRADE", dtype=object)
    action = np.zeros(len(df), dtype=int)

    tradable = (
        np.isfinite(mfe)
        & np.isfinite(mae)
        & (mfe >= float(cfg.mfe_min))
        & (eff >= float(cfg.eff_min))
    )

    # Trend: strong directional confidence, plus confirmation
    if str(cfg.trend_confirm_mode).lower() == "or":
        trend = (
            tradable
            & (dconf >= float(cfg.dir_conf_trend_min))
            & ((mfe >= float(cfg.mfe_trend_min)) | (ttm >= float(cfg.ttm_trend_min)))
        )
    else:
        trend = (
            tradable
            & (dconf >= float(cfg.dir_conf_trend_min))
            & (mfe >= float(cfg.mfe_trend_min))
            & (ttm >= float(cfg.ttm_trend_min))
        )

    # Mean: better efficiency and quicker ttm (wins fast)
    mean = (
        tradable
        & ~trend
        & (eff >= float(cfg.eff_mean_min))
        & (ttm <= float(cfg.ttm_mean_max))
    )

    mode[mean] = "MEAN"
    action[mean] = 1
    mode[trend] = "TREND"
    action[trend] = 2

    out = pd.DataFrame(index=df.index)
    out[out_col] = mode.astype(str)
    out["mode_action"] = action.astype(int)
    out["mfe_atr"] = mfe
    out["mae_atr"] = mae
    out["t_to_mfe"] = ttm
    out["eff"] = eff
    out["dir_conf"] = dconf
    return out


def compute_mode_3action_regime_aware(
    df: pd.DataFrame,
    *,
    rule_cfg: Rule3ActionConfig = Rule3ActionConfig(),
    score_cfg: QualityScoreConfig = QualityScoreConfig(),
    preds_in_log1p: bool = True,
    calibration: Optional[Dict[str, Any]] = None,
    out_col: str = "mode",
    use_physics_regime: bool = True,
    physics_regime_cfg: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Regime-aware router: Physics/Regime classification before mode selection.

    Flow:
    1. Classify Physics/Regime (feasibility check)
    2. Regime determines allowed archetype class
    3. Router selects within allowed options

    This prevents executing strategies in incompatible physical regimes.

    Args:
        df: DataFrame with required features and head outputs
        rule_cfg: Router configuration
        score_cfg: Quality score configuration
        preds_in_log1p: Whether predictions are in log1p space
        calibration: Calibration dictionary
        out_col: Output column name for mode
        use_physics_regime: Whether to enable Physics/Regime filtering
        physics_regime_cfg: Physics/Regime configuration (optional)

    Returns:
        DataFrame with mode, regime, and diagnostics
    """
    if not _PHYSICS_REGIME_AVAILABLE:
        # Fallback to legacy head-only router if physics_regime not available
        return compute_mode_3action(
            df,
            cfg=rule_cfg,
            preds_in_log1p=preds_in_log1p,
            calibration=calibration,
            out_col=out_col,
        )

    # Step 1: Classify Physics/Regime
    if use_physics_regime:
        if physics_regime_cfg is None:
            from src.time_series_model.rule.regime import PhysicsRegimeConfig

            physics_regime_cfg = PhysicsRegimeConfig()

        regime_df = classify_regime(df, cfg=physics_regime_cfg)
        regime = regime_df["regime"].astype(str).values
    else:
        # If disabled, set all to allow all regimes (for backward compatibility)
        regime = np.full(len(df), "TC_REGIME", dtype=object)
        regime_df = pd.DataFrame(index=df.index)
        regime_df["regime"] = regime

    # Step 2: Compute quality
    quality_df = compute_trade_quality(
        df,
        cfg=rule_cfg,
        score_cfg=score_cfg,
        preds_in_log1p=preds_in_log1p,
        calibration=calibration,
    )

    quality = quality_df["trade_quality"].to_numpy(dtype=float)

    # Step 3: Regime + quality mapping

    mode = np.full(len(df), "NO_TRADE", dtype=object)
    action = np.zeros(len(df), dtype=int)

    # Regime-based filtering
    tc_regime_mask = regime == "TC_REGIME"
    te_regime_mask = regime == "TE_REGIME"
    mean_regime_mask = regime == "MEAN_REGIME"
    no_trade_regime_mask = regime == "NO_TRADE"

    tc_ok = tc_regime_mask & (quality >= float(score_cfg.quality_trend_min))
    te_ok = te_regime_mask & (quality >= float(score_cfg.quality_te_min))
    mean_ok = mean_regime_mask & (quality >= float(score_cfg.quality_mean_min))

    # Apply regime constraints
    mode[tc_ok] = "TREND"
    action[tc_ok] = 2
    mode[te_ok] = "TREND"
    action[te_ok] = 2
    mode[mean_ok] = "MEAN"
    action[mean_ok] = 1

    # NO_TRADE regime overrides everything
    mode[no_trade_regime_mask] = "NO_TRADE"
    action[no_trade_regime_mask] = 0

    out = pd.DataFrame(index=df.index)
    out[out_col] = mode.astype(str)
    out["mode_action"] = action.astype(int)
    out = out.join(quality_df).join(regime_df)

    return out
