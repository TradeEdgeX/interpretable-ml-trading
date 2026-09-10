"""SRB-specific hooks for ``scripts/event_backtest.py`` (keeps the main script smaller).

Staged 2a/2b entry gating was removed from the event backtest loop; SRB follows the same
PCM → open path as other strategies. Remaining hooks: experiment feature injection,
optional wide-SR distance guard on new mothers, and post-open SR metadata for adds/L3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.time_series_model.live.srb_regime import (
    maybe_inject_srb_experiment_features,
    pick_srb_true_sr_level_with_source,
    should_reject_srb_wide_entry,
)


@dataclass
class SrbEventBacktestHooks:
    """Build once per backtest run; attach policies to simulators; call from the bar loop."""

    execution_raw: Dict[str, Any]
    add_policy: Optional[Dict[str, Any]]
    wide_entry_guard: Optional[Dict[str, Any]]

    @classmethod
    def try_from_strategies(
        cls,
        strategy_names: list[str],
        strats: Dict[str, Any],
    ) -> Optional["SrbEventBacktestHooks"]:
        if "srb" not in strategy_names:
            return None
        try:
            raw = dict(strats["srb"].archetype.execution.raw or {})
        except Exception:
            return None
        return cls(
            execution_raw=raw,
            add_policy=raw.get("srb_add_position_policy"),
            wide_entry_guard=raw.get("sr_wide_entry_guard"),
        )

    def attach_to_simulators(self, simulators: Dict[str, Any]) -> None:
        for sim in simulators.values():
            sim._srb_add_policy = self.add_policy
            sim._srb_wide_entry_guard = self.wide_entry_guard

    def inject_regime_features(
        self,
        *,
        sym: str,
        ts: Any,
        sym_bundle: Dict[str, Any],
        tf_srb: Optional[str],
        features_by_tf: Dict[str, Dict[str, Any]],
        primary_features: Dict[str, Any],
    ) -> None:
        if not tf_srb or tf_srb not in features_by_tf:
            return
        _df_srb = (sym_bundle.get("tf_features") or {}).get(tf_srb)
        if _df_srb is None or getattr(_df_srb, "empty", True):
            return
        maybe_inject_srb_experiment_features(
            df=_df_srb,
            ts=ts,
            exec_raw=self.execution_raw,
            out=features_by_tf[tf_srb],
        )
        for _k, _v in list(features_by_tf[tf_srb].items()):
            if str(_k).startswith("srb_"):
                primary_features[_k] = _v

    @staticmethod
    def sync_wide_sr_levels_on_simulator(
        simulator: Any, primary_features: Dict[str, Any]
    ) -> None:
        for _wk in ("wide_sr_upper_px", "wide_sr_lower_px"):
            _wv = primary_features.get(_wk)
            if _wv is None:
                continue
            try:
                _wf = float(_wv)
                if _wf == _wf:
                    setattr(simulator, f"_{_wk}", _wf)
            except (TypeError, ValueError):
                pass

    def reject_new_entry_wide_sr_guard(
        self,
        *,
        arch_lc: str,
        is_new_entry: bool,
        simulator: Any,
        entry_feats: Dict[str, Any],
        intent: Any,
        funnel: Dict[str, Any],
    ) -> bool:
        """Return True if this intent should be skipped (funnel key incremented)."""
        if arch_lc != "srb" or not is_new_entry:
            return False
        _wg = getattr(simulator, "_srb_wide_entry_guard", None) or {}
        if not _wg.get("enabled"):
            return False
        _min_atr = float(_wg.get("min_distance_atr", 0) or 0)
        _atr_wg = float(entry_feats.get("atr", 0) or 0)
        _px_wg = float(entry_feats.get("close", 0) or 0)
        _side_wg = str(intent.action or "").upper()
        if should_reject_srb_wide_entry(
            _side_wg,
            _px_wg,
            _atr_wg,
            entry_feats.get("wide_sr_lower_px"),
            entry_feats.get("wide_sr_upper_px"),
            _min_atr,
        ):
            funnel.setdefault("reject_srb_wide_sr_too_close", 0)
            funnel["reject_srb_wide_sr_too_close"] += 1
            return True
        return False

    def annotate_mother_on_open(
        self,
        *,
        opened: Any,
        arch_lc: str,
        is_new_entry: bool,
        simulator: Any,
        entry_feats: Dict[str, Any],
        entry_bar: Dict[str, Any],
    ) -> None:
        if opened is None or arch_lc != "srb" or not is_new_entry:
            return
        _srb_pos = simulator._positions.get(opened)
        if _srb_pos is None:
            return
        _srb_side = str(_srb_pos.get("side", "")).upper()
        try:
            _tsl_cfg = (self.execution_raw or {}).get("true_sr_level") or {}
        except Exception:
            _tsl_cfg = {}
        _fallback_atr = float(_tsl_cfg.get("wide_fallback_atr", 0) or 0)
        _prefer = _tsl_cfg.get("prefer")
        _entry_px = float(entry_bar.get("close", 0) or 0)
        _atr_e = float(entry_feats.get("atr", 0) or 0)
        _pick, _src = pick_srb_true_sr_level_with_source(
            _srb_side,
            _entry_px,
            _atr_e,
            narrow_support=entry_feats.get("srb_sr_support"),
            narrow_resistance=entry_feats.get("srb_sr_resistance"),
            wide_lower_px=entry_feats.get("wide_sr_lower_px"),
            wide_upper_px=entry_feats.get("wide_sr_upper_px"),
            fallback_atr=_fallback_atr,
            prefer=_prefer,
        )
        _srb_pos["_srb_true_sr_level"] = float(_pick)
        _srb_pos["_srb_true_sr_source"] = str(_src)
        for _k_src, _k_dst in (
            ("srb_sr_support", "_srb_l1_support"),
            ("srb_sr_resistance", "_srb_l1_resistance"),
            ("wide_sr_lower_px", "_srb_l3_lower"),
            ("wide_sr_upper_px", "_srb_l3_upper"),
        ):
            _raw = entry_feats.get(_k_src)
            if _raw is None:
                continue
            try:
                _vf = float(_raw)
                if _vf == _vf:
                    _srb_pos[_k_dst] = _vf
            except (TypeError, ValueError):
                pass
        try:
            _ewd = entry_feats.get("wide_sr_dist_atr")
            if _ewd is not None and _ewd == _ewd:
                _srb_pos["_srb_entry_wide_sr_dist_atr"] = float(_ewd)
        except (TypeError, ValueError):
            pass
