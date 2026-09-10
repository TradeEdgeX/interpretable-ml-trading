from __future__ import annotations

from typing import Any, Dict, Mapping


def _strategy_keys(archetype: str) -> list[str]:
    key = str(archetype or "").strip().lower()
    if not key:
        return []
    parts = [p for p in key.split("-") if p]
    keys: list[str] = []
    seen: set[str] = set()

    def _push(k: str) -> None:
        kk = str(k or "").strip().lower()
        if not kk or kk in seen:
            return
        seen.add(kk)
        keys.append(kk)

    _push(key)

    # strip timeframe suffix: "*-60t" / "*-240t"
    if parts and parts[-1].endswith("t") and parts[-1][:-1].isdigit():
        parts_no_tf = parts[:-1]
        _push("-".join(parts_no_tf))
    else:
        parts_no_tf = parts

    # family-direction key: "<family>-<long|short>"
    if len(parts_no_tf) >= 2 and parts_no_tf[1] in {"long", "short"}:
        _push("-".join(parts_no_tf[:2]))

    # family only
    if parts_no_tf:
        _push(parts_no_tf[0])

    return keys


def _as_dict(obj: Any) -> Dict[str, Any]:
    return dict(obj) if isinstance(obj, dict) else {}


def _value_by_add_number(
    raw: Any,
    add_number: int,
    default: float,
) -> float:
    if add_number <= 0:
        return default
    vals = raw
    if isinstance(vals, (int, float)):
        vals = [vals]
    if not isinstance(vals, list) or not vals:
        return default
    idx = min(add_number - 1, len(vals) - 1)
    try:
        return float(vals[idx])
    except Exception:
        return default


def resolve_strategy_add_position_config(
    *,
    archetype: str,
    add_position_rules: Mapping[str, Any] | None,
    per_strategy_limits: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Merge global add rules with strategy-specific overrides."""
    base = _as_dict(add_position_rules)
    limits = _as_dict(per_strategy_limits)
    merged = dict(base)
    strat_cfg: Dict[str, Any] = {}
    for key in _strategy_keys(archetype):
        cand = _as_dict(limits.get(key))
        if cand:
            strat_cfg = cand
            break

    strat_add = _as_dict(strat_cfg.get("add_position"))
    for field in ("max_add_times", "add_size_multipliers", "min_current_r_by_add"):
        if field in strat_cfg and field not in strat_add:
            strat_add[field] = strat_cfg[field]

    trigger_cfg = _as_dict(base.get("trigger"))
    trigger_cfg.update(_as_dict(strat_add.get("trigger")))
    if trigger_cfg:
        merged["trigger"] = trigger_cfg

    merged.update({k: v for k, v in strat_add.items() if k != "trigger"})
    return merged


def _trigger_type(add_position_cfg: Mapping[str, Any] | None) -> str:
    trig = _as_dict(_as_dict(add_position_cfg).get("trigger"))
    return str(trig.get("type", "")).strip().lower()


def resolve_float_r_ladder_only(add_position_cfg: Mapping[str, Any] | None) -> bool:
    """True iff ``add_position.trigger.type == "float_r_ladder_only"`` (事件回测浮盈阶梯路径)."""
    return _trigger_type(add_position_cfg) == "float_r_ladder_only"


def resolve_pullback_near_ema1200(add_position_cfg: Mapping[str, Any] | None) -> bool:
    """True iff add trigger is pullback-to-EMA1200 (passive ladder + near-EMA gate)."""
    return _trigger_type(add_position_cfg) == "pullback_near_ema1200"


def resolve_passive_add_ladder(add_position_cfg: Mapping[str, Any] | None) -> bool:
    """True for bar-driven add paths that skip PCM re-signal (float_r or pullback-EMA)."""
    tt = _trigger_type(add_position_cfg)
    return tt in {"float_r_ladder_only", "pullback_near_ema1200"}


def pullback_near_ema1200_allows(
    features: Mapping[str, Any] | None,
    add_position_cfg: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Gate for ``trigger.type: pullback_near_ema1200``.

    Requires ``|ema_1200_position| <= max_abs_ema_1200_position`` (default 0.02).
    Optional ``max_dist_atr``: ``|ema_1200_position| * close / atr <= max_dist_atr``.

    Float-profit gate remains ``min_current_r_by_add`` in try_add_position — this
    helper only checks the near-EMA geometry (never a martingale without float).
    """
    feats = _as_dict(features)
    trig = _as_dict(_as_dict(add_position_cfg).get("trigger"))
    try:
        ep = float(feats.get("ema_1200_position"))
    except (TypeError, ValueError):
        return False, "missing_ema_1200_position"
    if ep != ep:  # NaN
        return False, "nan_ema_1200_position"

    try:
        max_abs = float(trig.get("max_abs_ema_1200_position", 0.02) or 0.02)
    except (TypeError, ValueError):
        max_abs = 0.02
    max_abs = max(0.0, max_abs)
    if abs(ep) > max_abs + 1e-12:
        return False, f"ema_pos_abs>{max_abs:g}"

    max_dist_atr_raw = trig.get("max_dist_atr")
    if max_dist_atr_raw is not None and str(max_dist_atr_raw).strip() != "":
        try:
            max_dist_atr = float(max_dist_atr_raw)
        except (TypeError, ValueError):
            max_dist_atr = 0.0
        if max_dist_atr > 0:
            try:
                close = float(feats.get("close", 0) or 0)
                atr = float(feats.get("atr", 0) or 0)
            except (TypeError, ValueError):
                close, atr = 0.0, 0.0
            if atr <= 1e-12 or close <= 0:
                return False, "missing_atr_for_dist"
            dist_atr = abs(ep) * close / atr
            if dist_atr > max_dist_atr + 1e-12:
                return False, f"dist_atr>{max_dist_atr:g}"
    return True, ""


def resolve_add_position_size_multiplier(
    add_position_cfg: Mapping[str, Any] | None,
    add_number: int,
    signal: Mapping[str, Any] | None = None,
) -> float:
    if add_number <= 0:
        return 1.0
    cfg = _as_dict(add_position_cfg)
    sizing_mode = str(cfg.get("sizing_mode", "fixed_multiplier")).strip().lower()
    if sizing_mode == "target_leverage_gap":
        sig = dict(signal or {})
        target_lev = _value_by_add_number(
            cfg.get("target_leverage_by_add"), add_number, 0.0
        )
        current_lev = 0.0
        try:
            current_lev = max(0.0, float(sig.get("current_leverage", 0.0) or 0.0))
        except Exception:
            current_lev = 0.0
        base_lev = 1.0
        try:
            base_lev = max(
                1e-6,
                float(sig.get("base_leverage_unit", 1.0) or 1.0),
            )
        except Exception:
            base_lev = 1.0
        gap = max(0.0, target_lev - current_lev)
        mult = gap / base_lev
        max_total_lev = _value_by_add_number(
            cfg.get("max_total_leverage"), add_number, 0.0
        )
        if max_total_lev > 0:
            lev_room = max(0.0, max_total_lev - current_lev)
            mult = min(mult, lev_room / base_lev)
        max_step = _value_by_add_number(
            cfg.get("max_add_leverage_step"), add_number, 0.0
        )
        if max_step > 0:
            mult = min(mult, max_step / base_lev)
        max_notional_frac = _value_by_add_number(
            cfg.get("max_add_notional_frac"), add_number, 0.0
        )
        if max_notional_frac > 0:
            try:
                current_notional_frac = max(
                    0.0, float(sig.get("current_notional_frac", 0.0) or 0.0)
                )
                add_frac_cap = max(0.0, max_notional_frac - current_notional_frac)
                base_frac = max(
                    1e-9, float(sig.get("base_notional_frac", max_notional_frac))
                )
                mult = min(mult, add_frac_cap / base_frac)
            except Exception:
                pass
        min_add_usd = _value_by_add_number(
            cfg.get("min_add_notional_usd"), add_number, 0.0
        )
        if min_add_usd > 0:
            try:
                equity = max(0.0, float(sig.get("equity_usd", 0.0) or 0.0))
                base_frac = max(
                    1e-9, float(sig.get("base_notional_frac", max_notional_frac))
                )
                min_mult = min_add_usd / max(equity * base_frac, 1e-9)
                mult = max(mult, min_mult)
            except Exception:
                pass
        if mult <= 0:
            mult = _value_by_add_number(
                cfg.get("add_size_multipliers"), add_number, 1.0
            )
    else:
        mult = _value_by_add_number(cfg.get("add_size_multipliers"), add_number, 1.0)
    return mult if mult > 0 else 1.0


def resolve_add_position_max_times(
    add_position_cfg: Mapping[str, Any] | None,
) -> int:
    """Maximum add-on rounds for ladder sizing / gating.

    If ``add_size_multipliers`` or ``min_current_r_by_add`` is a non-empty list,
    the maximum of their lengths drives the cap (same ladder semantics as
    :func:`_value_by_add_number`). Otherwise ``max_add_times`` from merged
    YAML is used (default 1).

    Notes:
      - Live ``validate_add_position`` still enforces constitution
        ``per_strategy_limits.*.max_add_times`` independently; keep that in
        sync with ladder vector lengths when omitting redundant scalars here.
    """
    cfg = _as_dict(add_position_cfg)
    inferred = 0
    for key in ("add_size_multipliers", "min_current_r_by_add"):
        raw_l = cfg.get(key)
        if isinstance(raw_l, list) and len(raw_l) > 0:
            inferred = max(inferred, len(raw_l))
    if inferred > 0:
        return inferred
    # `or 1` would silently coerce an explicit max_add_times: 0 (disable adds)
    # back to 1 — 0 is falsy in Python. Only missing/None should default to 1.
    raw = cfg.get("max_add_times", 1)
    if raw is None:
        return 1
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 1


def resolve_add_position_min_current_r(
    add_position_cfg: Mapping[str, Any] | None,
    add_number: int,
    signal: Mapping[str, Any] | None = None,
) -> float:
    if add_number <= 0:
        return 0.0
    cfg = _as_dict(add_position_cfg)
    threshold_raw = max(
        0.0, _value_by_add_number(cfg.get("min_current_r_by_add"), add_number, 0.0)
    )

    # Unit semantics for min_current_r_by_add:
    # - initial_r (default): threshold is in current_r units
    # - atr: threshold is in ATR units and converted to current_r by parent_initial_r
    unit = (
        str(
            cfg.get(
                "min_current_r_unit", cfg.get("min_current_by_add_unit", "initial_r")
            )
            or "initial_r"
        )
        .strip()
        .lower()
    )
    if unit not in {"atr", "initial_r"}:
        unit = "initial_r"
    if unit != "atr":
        return threshold_raw

    sig = _as_dict(signal)
    try:
        parent_initial_r = float(sig.get("parent_initial_r", 0.0) or 0.0)
    except Exception:
        parent_initial_r = 0.0
    if parent_initial_r <= 0:
        # Safe fallback: if ATR conversion basis is unavailable, keep old behavior.
        return threshold_raw
    return max(0.0, threshold_raw / parent_initial_r)


def _get_num(signal: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in signal and signal.get(name) is not None:
            try:
                return float(signal.get(name))
            except Exception:
                continue
    return None


def add_regime_gate_allows(
    features: Mapping[str, Any] | None,
    add_position_cfg: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Optional post-trigger gate for add-on entries (event backtest / live alignment).

    YAML (under ``add_position``):

        add_regime_gate:
          enabled: true
          allow_if_all:
            - feature: bpc_semantic_chop_ts_q
              lte: 0.55
            - feature: add_ml_score
              gte: 0.52

    Rules in ``allow_if_all`` are AND-ed. If a listed feature is missing or
    non-finite on the current bar, that rule is skipped (pass-through).

    Rule schema supports either:
      - ``feature``: direct feature name
      - ``feature_by_side``: dict {"long": "...", "short": "..."} selected by
        signal field ``position_action`` (LONG/SHORT)
    """
    gate = _as_dict(_as_dict(add_position_cfg).get("add_regime_gate"))
    if not gate or not bool(gate.get("enabled")):
        return True, ""
    rules = gate.get("allow_if_all")
    if not isinstance(rules, list) or not rules:
        return True, ""
    feat_map: Dict[str, Any] = dict(features or {})
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        fname = str(rule.get("feature", "")).strip()
        if not fname:
            fbs = _as_dict(rule.get("feature_by_side"))
            pos_action = str(feat_map.get("position_action", "")).strip().lower()
            if pos_action in {"long", "buy"}:
                fname = str(fbs.get("long", "")).strip()
            elif pos_action in {"short", "sell"}:
                fname = str(fbs.get("short", "")).strip()
        if not fname:
            continue
        raw = feat_map.get(fname)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v != v:
            continue
        if bool(rule.get("align_with_side")):
            pos_action = str(feat_map.get("position_action", "")).strip().lower()
            if pos_action in {"short", "sell"}:
                v = -v
        if "lte" in rule:
            lim = float(rule["lte"])
            if v > lim + 1e-12:
                return False, f"{fname}>{lim:g}"
        if "gte" in rule:
            lim = float(rule["gte"])
            if v < lim - 1e-12:
                return False, f"{fname}<{lim:g}"
    return True, ""


def validate_add_position_trigger(
    *,
    archetype: str,
    direction: int,
    signal: Mapping[str, Any],
    add_position_cfg: Mapping[str, Any] | None,
    current_r: float,
) -> bool:
    cfg = _as_dict(add_position_cfg)
    add_seq = int(signal.get("add_position_seq", 1) or 1)
    if current_r < resolve_add_position_min_current_r(cfg, add_seq, signal):
        return False
    trigger = _as_dict(cfg.get("trigger"))
    if not trigger:
        return True

    trig_type = str(trigger.get("type", "")).strip().lower()
    # 与 resolve_float_r_ladder_only 一致：阶梯模式由事件回测单独路径处理，此处不附加特征条件。
    if trig_type == "float_r_ladder_only":
        return True
    if trig_type == "pullback_near_ema1200":
        ok, _why = pullback_near_ema1200_allows(signal, add_position_cfg)
        return bool(ok)

    return True


def latest_add_stop_price(
    *,
    side: str,
    add_entry: float,
    sl_distance: float,
) -> float | None:
    """Whole-book SL after an add: add_entry ± original SL distance."""
    try:
        px = float(add_entry)
        dist = float(sl_distance)
    except (TypeError, ValueError):
        return None
    if px <= 0.0 or dist <= 1e-12:
        return None
    if str(side or "").upper() in {"LONG", "BUY"}:
        return px - dist
    return px + dist


def tighten_stop_price(
    *,
    side: str,
    current: float | None,
    candidate: float,
) -> float:
    """Move SL only toward the position (never loosen)."""
    cand = float(candidate)
    if current is None:
        return cand
    try:
        cur = float(current)
    except (TypeError, ValueError):
        return cand
    if str(side or "").upper() in {"LONG", "BUY"}:
        return max(cur, cand)
    return min(cur, cand)


def origin_sl_distance(parent_pos: Mapping[str, Any]) -> float:
    """Frozen first-entry SL width. Do not recompute after a ratchet."""
    try:
        cached = float(parent_pos.get("_origin_sl_distance") or 0.0)
    except (TypeError, ValueError):
        cached = 0.0
    if cached > 1e-12:
        return cached
    try:
        ep = float(parent_pos.get("entry_price") or 0.0)
        sl = float(parent_pos.get("stop_loss_price") or 0.0)
    except (TypeError, ValueError):
        ep, sl = 0.0, 0.0
    if ep > 0.0 and sl > 0.0:
        return abs(ep - sl)
    try:
        return float(parent_pos.get("initial_risk_distance") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def apply_latest_add_stop_ratchet(
    positions: Mapping[str, Any],
    *,
    parent_pid: str,
    add_entry: float,
    side: str,
) -> float | None:
    """Move parent + add-leg stops to latest add entry ± original SL distance.

    Research flag ``add_position.ratchet_stop_to_latest_add`` (off in prod).
    """
    parent = positions.get(str(parent_pid))
    if not isinstance(parent, dict):
        return None
    dist = origin_sl_distance(parent)
    parent["_origin_sl_distance"] = float(dist)
    cand = latest_add_stop_price(side=side, add_entry=add_entry, sl_distance=dist)
    if cand is None:
        return None
    pid_parent = str(parent_pid)
    for pid, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        if str(pid) != pid_parent and str(pos.get("_parent_pid") or "") != pid_parent:
            continue
        old = pos.get("stop_loss_price")
        try:
            old_f = float(old) if old is not None else None
        except (TypeError, ValueError):
            old_f = None
        pos["stop_loss_price"] = tighten_stop_price(
            side=side, current=old_f, candidate=cand
        )
    return cand
