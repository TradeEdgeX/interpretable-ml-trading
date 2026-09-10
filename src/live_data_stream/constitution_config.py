"""Read constitution YAML paths and shared sections (trend/fat-tail + hedge multi-leg + publisher)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from src.config.strategy_layout import deep_merge_dicts

logger = logging.getLogger(__name__)
MULTI_LEG_STRATEGY_TYPES = frozenset({"grid", "dual_add_trend", "trend_scalp"})
SPOT_STRATEGY_TYPES = frozenset({"spot", "spot_accum"})


def resolve_constitution_yaml(
    strategies_root: str, *, override: Optional[str] = None
) -> str:
    if override and str(override).strip():
        return str(override).strip()
    config_root = os.path.join(strategies_root, "..")
    return os.getenv(
        "MLBOT_CONSTITUTION_YAML",
        os.path.join(config_root, "constitution", "constitution.yaml"),
    )


def _load_constitution_yaml_raw(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_extends_path(child_path: Path, extends: str) -> Path:
    raw = str(extends or "").strip()
    if not raw:
        raise ValueError(f"empty extends in {child_path}")
    base = Path(raw)
    if base.is_absolute():
        return base.resolve()
    return (child_path.parent / base).resolve()


def _load_constitution_with_extends(
    path: str,
    *,
    _stack: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Load constitution YAML, recursively merging ``extends`` (relative to file dir)."""
    p = Path(path).resolve()
    if not p.is_file():
        return {}
    key = str(p)
    stack = _stack or ()
    if key in stack:
        raise ValueError(f"constitution extends cycle: {' -> '.join((*stack, key))}")

    raw = _load_constitution_yaml_raw(key)
    extends = raw.pop("extends", None)
    if not extends:
        return dict(raw)

    parent_path = _resolve_extends_path(p, str(extends))
    if not parent_path.is_file():
        raise ValueError(
            f"constitution extends target not found: {parent_path} (from {p})"
        )
    parent = _load_constitution_with_extends(
        str(parent_path),
        _stack=(*stack, key),
    )
    return deep_merge_dicts(parent, raw)


def load_constitution_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        return _load_constitution_with_extends(path)
    except ValueError:
        raise
    except Exception as exc:
        logger.warning("读取 constitution 失败 %s: %s", path, exc)
        return {}


def enabled_archetypes_from_constitution(cfg: Dict[str, Any]) -> List[str]:
    """PCM 联合回测白名单 + trend/fat-tail LivePCM 注册候选（同一列表）。

    配置在 ``resource_allocation.enabled_archetypes``（或根级同名键）。
    推荐 YAML 显式 ``- srb`` 列表；也接受逗号分隔字符串（与 ``multi_leg.strategies`` 一致）。

    Semantics (fail-closed when the key is present):
    - **Missing** key → built-in full set (research back-compat:「未写白名单即全开」).
    - **Explicit empty** list / blank string → ``[]`` (do **not** expand to full set;
      live runners refuse to start on empty — clearing the list must not re-enable all).
    - Unknown type → ``[]`` (refuse unsafe configs rather than enable everything).

    Ids listed under root ``aux_strategies`` are always stripped (fail-closed vs
    accidental dual membership with CMS aux).
    """
    _ALL = ["bpc", "me", "srb", "tpc", "lv", "fbf", "msr", "fer"]
    ra = cfg.get("resource_allocation") or {}
    if "enabled_archetypes" in ra:
        raw = ra.get("enabled_archetypes")
        key_present = True
    elif "enabled_archetypes" in cfg:
        raw = cfg.get("enabled_archetypes")
        key_present = True
    else:
        raw = None
        key_present = False

    if not key_present:
        parts = [a.lower() for a in _ALL]
    elif raw is None or raw == "":
        parts = []
    elif isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = [str(a).lower().strip() for a in raw if str(a).strip()]
    else:
        parts = []
    aux_ids = {
        str(r.get("id") or "").strip().lower()
        for r in aux_strategies_from_constitution(cfg)
        if r.get("id")
    }
    if not aux_ids:
        return parts
    return [p for p in parts if p not in aux_ids]


def enabled_archetypes_key_present(cfg: Dict[str, Any]) -> bool:
    """True when YAML explicitly sets enabled_archetypes (live/research whitelist)."""
    ra = cfg.get("resource_allocation") or {}
    return "enabled_archetypes" in ra or "enabled_archetypes" in cfg


def intent_archetype_priority_tokens(cfg: Dict[str, Any]) -> List[str]:
    """Tokens for PCM archetype ordering (same-bar intent sort + LivePCM.register order helper).

    If ``resource_allocation.intent_selection_policy.archetype_priority`` is set and non-empty,
    use it. Otherwise use ``enabled_archetypes`` list order (single source with membership).
    """
    ra = cfg.get("resource_allocation") or {}
    isp = ra.get("intent_selection_policy") or {}
    raw = isp.get("archetype_priority")
    if raw is not None and raw != "":
        if isinstance(raw, str):
            parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
            if parts:
                return parts
        elif isinstance(raw, (list, tuple)) and raw:
            return [str(x).lower().strip() for x in raw if str(x).strip()]
    return list(enabled_archetypes_from_constitution(cfg))


def multi_leg_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sec = cfg.get("multi_leg")
    return sec if isinstance(sec, dict) else {}


def resolve_constitution_yaml_path(*, override: Optional[str] = None) -> str:
    """Path to constitution.yaml for metrics / tooling (env > override > live > config)."""
    raw = (override or os.getenv("MLBOT_CONSTITUTION_YAML", "")).strip()
    if raw and os.path.isfile(raw):
        return raw
    for candidate in (
        "live/highcap/config/constitution/constitution.yaml",
        "config/constitution/constitution.yaml",
    ):
        if os.path.isfile(candidate):
            return candidate
    return raw or "config/constitution/constitution.yaml"


def per_strategy_limits_from_constitution(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ra = cfg.get("resource_allocation") or {}
    raw = ra.get("per_strategy_limits") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def strategies_for_slot_metrics_from_constitution(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    constitution_path: Optional[str] = None,
) -> List[str]:
    """Strategy names for Prometheus slot gauge bootstrap (read from constitution, not hardcoded)."""
    if cfg is None:
        path = resolve_constitution_yaml_path(override=constitution_path)
        cfg = load_constitution_dict(path)
    seen: Set[str] = set()
    out: List[str] = []

    def _add(items: Iterable[Any]) -> None:
        for item in items:
            key = str(item or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)

    _add(enabled_archetypes_from_constitution(cfg))
    _add(per_strategy_limits_from_constitution(cfg).keys())
    _add(multi_leg_strategies_from_constitution(cfg))
    _add(spot_strategies_from_constitution(cfg))
    for members in archetype_groups_from_constitution(cfg).values():
        _add(members)
    return out


def console_live_strategies_from_constitution(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    constitution_path: Optional[str] = None,
) -> List[str]:
    """Live strategies for business console (account summary, regime ops).

    Unlike ``strategies_for_slot_metrics_from_constitution`` (Prometheus bootstrap),
    this does **not** union ``per_strategy_limits`` keys — those are risk caps for
    research/PCM candidates, not proof a strategy is live-enabled.

    Does **not** include ``aux_strategies`` (CMS Trade Map / manual aux only).
    """
    if cfg is None:
        path = resolve_constitution_yaml_path(override=constitution_path)
        cfg = load_constitution_dict(path)
    seen: Set[str] = set()
    out: List[str] = []

    def _add(items: Iterable[Any]) -> None:
        for item in items:
            key = str(item or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)

    policy = classic_slot_policy_from_constitution(cfg)
    _add(policy.get("trend_archetypes") or [])
    _add(multi_leg_strategies_from_constitution(cfg))
    _add(spot_strategies_from_constitution(cfg))
    return out


# Aux / paper strategies for CMS Trade Map only (never LivePCM / OMS).
AUX_EXECUTION_MODES = frozenset({"aux_manual", "sim", "paper"})
LIVE_EXECUTION_MODE = "live"


def aux_strategies_from_constitution(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    constitution_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Parse root-level ``aux_strategies`` (CMS-only; not ``enabled_archetypes``).

    YAML shapes::

        aux_strategies:
          - id: sr_confirm
            execution: aux_manual
            title: SRC

        # or bare ids (execution defaults to aux_manual):
        aux_strategies:
          - sr_confirm
    """
    if cfg is None:
        path = resolve_constitution_yaml_path(override=constitution_path)
        cfg = load_constitution_dict(path)
    raw = cfg.get("aux_strategies")
    if raw is None or raw == "":
        return []
    out: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def _append(sid: str, execution: str, title: str) -> None:
        key = str(sid or "").strip().lower()
        if not key or key in seen:
            return
        exe = str(execution or "aux_manual").strip().lower() or "aux_manual"
        if exe not in AUX_EXECUTION_MODES:
            # Unknown mode → treat as aux_manual (still not live).
            exe = "aux_manual"
        seen.add(key)
        out.append(
            {
                "id": key,
                "execution": exe,
                "title": str(title or key).strip() or key,
            }
        )

    if isinstance(raw, dict):
        for sid, meta in raw.items():
            if isinstance(meta, dict):
                _append(
                    str(sid),
                    str(meta.get("execution") or "aux_manual"),
                    str(meta.get("title") or sid),
                )
            else:
                _append(str(sid), "aux_manual", str(sid))
        return out
    if isinstance(raw, str):
        for part in raw.split(","):
            _append(part.strip(), "aux_manual", part.strip())
        return out
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str):
                _append(item, "aux_manual", item)
            elif isinstance(item, dict):
                _append(
                    str(item.get("id") or item.get("strategy") or ""),
                    str(item.get("execution") or "aux_manual"),
                    str(item.get("title") or item.get("id") or ""),
                )
        return out
    return out


def aux_strategy_ids_from_constitution(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    constitution_path: Optional[str] = None,
) -> List[str]:
    return [row["id"] for row in aux_strategies_from_constitution(cfg, constitution_path=constitution_path)]


def refuse_aux_strategy_for_live(strategy_id: str, cfg: Optional[Dict[str, Any]] = None) -> None:
    """Fail-closed: aux_strategies must never place live OMS / LivePCM orders."""
    sid = str(strategy_id or "").strip().lower()
    if not sid:
        return
    if sid in set(aux_strategy_ids_from_constitution(cfg)):
        raise RuntimeError(
            f"strategy {sid!r} is constitution aux_strategies "
            f"(execution≠live) — refuse LivePCM / OMS routing"
        )


def rolling_section_from_constitution(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    constitution_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Root-level ``rolling:`` (U-m rolling registry + account G0). Not PCM."""
    if cfg is None:
        path = resolve_constitution_yaml_path(override=constitution_path)
        cfg = load_constitution_dict(path)
    sec = cfg.get("rolling")
    return sec if isinstance(sec, dict) else {}


def rolling_account_risk_from_constitution(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    constitution_path: Optional[str] = None,
) -> Dict[str, Any]:
    """``rolling.account_risk`` — G0 knobs (g0_enabled / g0_max_dd)."""
    sec = rolling_section_from_constitution(cfg, constitution_path=constitution_path)
    risk = sec.get("account_risk")
    return risk if isinstance(risk, dict) else {}


def rolling_g0_defaults_from_constitution(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    constitution_path: Optional[str] = None,
) -> Tuple[bool, float]:
    """Return ``(g0_enabled, g0_max_dd)`` from constitution (defaults True / 0.30)."""
    risk = rolling_account_risk_from_constitution(cfg, constitution_path=constitution_path)
    enabled_raw = risk.get("g0_enabled", True)
    if isinstance(enabled_raw, str):
        enabled = enabled_raw.strip().lower() not in ("0", "false", "off", "no")
    else:
        enabled = bool(enabled_raw)
    try:
        max_dd = float(risk.get("g0_max_dd", 0.30) or 0.30)
    except (TypeError, ValueError):
        max_dd = 0.30
    if max_dd <= 0:
        max_dd = 0.30
    return enabled, max_dd


def rolling_accounts_from_constitution(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    constitution_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Parse ``rolling.accounts`` list (slug / symbol / desired_mode)."""
    sec = rolling_section_from_constitution(cfg, constitution_path=constitution_path)
    raw = sec.get("accounts")
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip().lower()
        if not slug:
            continue
        out.append(
            {
                "slug": slug,
                "symbol": str(item.get("symbol") or "").strip().upper(),
                "desired_mode": str(item.get("desired_mode") or "dry-run")
                .strip()
                .lower()
                or "dry-run",
            }
        )
    return out


def multi_leg_strategies_from_constitution(cfg: Dict[str, Any]) -> List[str]:
    raw = (multi_leg_section(cfg) or {}).get("strategies")
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [p.strip().lower() for p in raw.split(",") if p.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    return []


def spot_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sec = cfg.get("spot")
    return sec if isinstance(sec, dict) else {}


def spot_strategies_from_constitution(cfg: Dict[str, Any]) -> List[str]:
    raw = (spot_section(cfg) or {}).get("strategies")
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [p.strip().lower() for p in raw.split(",") if p.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    return []


def spot_account_from_constitution(cfg: Dict[str, Any]) -> Dict[str, Any]:
    account = (spot_section(cfg) or {}).get("account")
    return dict(account) if isinstance(account, dict) else {}


def spot_account_equity_anchor_usdt(
    account: Optional[Dict[str, Any]],
    *,
    default: float = 10000.0,
) -> float:
    """Offline equity anchor for spot backtest / deploy pct (live uses exchange sync).

    Accepts ``equity_usdt`` (canonical, same key as multi_leg.account) or legacy
    ``backtest_equity_usdt``.
    """
    if not isinstance(account, dict):
        return float(default)
    for key in ("equity_usdt", "backtest_equity_usdt"):
        raw = account.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return float(default)


def spot_strategy_limits_from_constitution(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = (spot_section(cfg) or {}).get("strategy_limits")
    return dict(raw) if isinstance(raw, dict) else {}


def archetype_groups_from_constitution(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    ra = cfg.get("resource_allocation") or {}
    groups = ra.get("archetype_groups") or {}
    if not isinstance(groups, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for group, raw in groups.items():
        if isinstance(raw, str):
            vals = [p.strip().lower() for p in raw.split(",") if p.strip()]
        elif isinstance(raw, (list, tuple)):
            vals = [str(x).strip().lower() for x in raw if str(x).strip()]
        else:
            vals = []
        if vals:
            out[str(group).strip().lower()] = vals
    return out


def classic_slot_policy_from_constitution(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ra = cfg.get("resource_allocation") or {}
    raw = ra.get("slot_policy") or {}
    policy = dict(raw) if isinstance(raw, dict) else {}
    groups = archetype_groups_from_constitution(cfg)
    group_name = str(policy.get("trend_group", "trend") or "trend").strip().lower()
    trend = groups.get(group_name, [])
    # Backward-compatible fallback for older local test fixtures only.
    if not trend and isinstance(policy.get("trend_archetypes"), (list, tuple, str)):
        legacy = policy.get("trend_archetypes")
        if isinstance(legacy, str):
            trend = [p.strip().lower() for p in legacy.split(",") if p.strip()]
        else:
            trend = [str(x).strip().lower() for x in legacy if str(x).strip()]
    # Trend slot pool: archetype_groups.trend ∩ enabled_archetypes, or enabled-only when
    # groups omitted (live constitution).
    if trend:
        enabled_set = set(enabled_archetypes_from_constitution(cfg))
        trend = [a for a in trend if a in enabled_set]
    elif enabled_archetypes_key_present(cfg):
        trend = list(enabled_archetypes_from_constitution(cfg))
    policy["trend_archetypes"] = trend
    policy["min_trend_slots_per_symbol"] = int(
        policy.get("min_trend_slots_per_symbol", 1) or 1
    )
    policy["max_trend_slots_per_symbol"] = int(
        policy.get(
            "max_trend_slots_per_symbol",
            1 if bool(policy.get("enforce_single_trend_per_symbol", False)) else 0,
        )
        or 0
    )
    return policy


def normalize_symbols_for_slot_validation(
    symbols: Optional[Union[str, Iterable[str]]],
) -> List[str]:
    """Coerce universe input to uppercase symbol tokens.

    Callers historically pass either a ``list[str]`` (live) or a comma-separated
    ``str`` from ``resolve_symbols_from_config`` (research); iterating a string
    byte-by-character would wrongly inflate slot requirements.
    """
    if symbols is None:
        return []
    if isinstance(symbols, str):
        raw = symbols.replace("|", ",").replace(";", ",")
        out: List[str] = []
        for chunk in raw.split(","):
            t = chunk.strip().upper()
            if t:
                out.append(t)
        return out
    return [str(s).strip().upper() for s in symbols if str(s).strip()]


def validate_classic_slot_capacity(
    *,
    constitution_cfg: Dict[str, Any],
    symbols: Optional[Union[str, Iterable[str]]],
) -> Dict[str, Any]:
    policy = classic_slot_policy_from_constitution(constitution_cfg)
    clean_symbols = sorted(set(normalize_symbols_for_slot_validation(symbols)))
    slots = constitution_cfg.get("slots") or {}
    slot_count = int(slots.get("slot_count", 0) or 0)
    min_per_symbol = int(policy.get("min_trend_slots_per_symbol", 1) or 1)
    required = len(clean_symbols) * max(min_per_symbol, 0)
    if required > 0 and slot_count < required:
        raise ValueError(
            "constitution slots.slot_count is too small for classic trend policy: "
            f"symbols={len(clean_symbols)} min_trend_slots_per_symbol={min_per_symbol} "
            f"required={required} slot_count={slot_count}"
        )
    return {
        "symbols": clean_symbols,
        "slot_count": slot_count,
        "required_trend_slots": required,
        "policy": policy,
    }


def _strategy_type_from_pipeline_entry(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("strategy_type", "") or "").strip().lower()


def partition_pipeline_strategies_by_type(
    pipeline_cfg: Dict[str, Any],
) -> Dict[str, Set[str]]:
    raw = pipeline_cfg.get("strategies") or {}
    if not isinstance(raw, dict):
        return {"classic": set(), "multi_leg": set(), "spot": set()}
    classic: Set[str] = set()
    multi_leg: Set[str] = set()
    spot: Set[str] = set()
    for key, scfg in raw.items():
        name = str(key or "").strip().lower()
        if not name:
            continue
        stype = _strategy_type_from_pipeline_entry(scfg)
        if stype in MULTI_LEG_STRATEGY_TYPES:
            multi_leg.add(name)
        elif stype in SPOT_STRATEGY_TYPES:
            spot.add(name)
        else:
            classic.add(name)
    return {"classic": classic, "multi_leg": multi_leg, "spot": spot}


def validate_pipeline_constitution_alignment(
    *,
    pipeline_cfg: Dict[str, Any],
    constitution_cfg: Dict[str, Any],
    context_label: str = "rolling_research",
) -> Dict[str, List[str]]:
    """Ensure every strategy in the pipeline YAML is authorized by the constitution.

    Subset semantics (single-strategy research): ``enabled_archetypes`` /
    ``multi_leg`` / ``spot`` may list more than the pipeline; only
    **pipeline ⊂ constitution** is required.
    Violations: a classic name not in ``enabled_archetypes``, or a multi-leg name not
    in ``multi_leg.strategies``, or a spot name not in ``spot.strategies``.
    """
    parts = partition_pipeline_strategies_by_type(pipeline_cfg)
    pipeline_classic = set(parts["classic"])
    pipeline_multi_leg = set(parts["multi_leg"])
    pipeline_spot = set(parts.get("spot") or set())
    const_classic = set(enabled_archetypes_from_constitution(constitution_cfg))
    const_multi_leg = set(multi_leg_strategies_from_constitution(constitution_cfg))
    const_spot = set(spot_strategies_from_constitution(constitution_cfg))

    # Backward-compatible inference: if strategy_type is omitted in pipeline YAML,
    # spot strategies can still be recognized by constitution spot allowlist.
    inferred_spot = pipeline_classic & const_spot
    if inferred_spot:
        pipeline_spot |= inferred_spot
        pipeline_classic -= inferred_spot

    classic_missing_in_const = sorted(pipeline_classic - const_classic)
    multi_missing_in_const = sorted(pipeline_multi_leg - const_multi_leg)
    spot_missing_in_const = sorted(pipeline_spot - const_spot)

    if classic_missing_in_const or multi_missing_in_const or spot_missing_in_const:
        msg_lines = [
            f"{context_label}: pipeline strategy not allowed by constitution "
            "(pipeline must be a subset of constitution lists)",
            f"  classic pipeline={sorted(pipeline_classic)}",
            f"  classic constitution={sorted(const_classic)}",
            f"  multi_leg pipeline={sorted(pipeline_multi_leg)}",
            f"  multi_leg constitution={sorted(const_multi_leg)}",
            f"  spot pipeline={sorted(pipeline_spot)}",
            f"  spot constitution={sorted(const_spot)}",
        ]
        if classic_missing_in_const:
            msg_lines.append(
                "  not in constitution.enabled_archetypes="
                f"{classic_missing_in_const}"
            )
        if multi_missing_in_const:
            msg_lines.append(
                "  not in constitution.multi_leg.strategies="
                f"{multi_missing_in_const}"
            )
        if spot_missing_in_const:
            msg_lines.append(
                "  not in constitution.spot.strategies="
                f"{spot_missing_in_const}"
            )
        raise ValueError("\n".join(msg_lines))

    return {
        "classic": sorted(pipeline_classic),
        "multi_leg": sorted(pipeline_multi_leg),
        "spot": sorted(pipeline_spot),
    }


def _float_or_none(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _literal_from_env(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return _float_or_none(raw)


def _chop_grid_execution_path_for_strategies_root(strategies_root: str) -> Path:
    from pathlib import Path

    root = Path(strategies_root)
    return root / "chop_grid" / "archetypes" / "execution.yaml"


def _trend_scalp_execution_path_for_strategies_root(strategies_root: str) -> Path:
    from pathlib import Path

    root = Path(strategies_root)
    return root / "trend_scalp" / "archetypes" / "execution.yaml"


def resolve_multi_leg_unit_notionals_from_constitution(
    ml: Dict[str, Any],
    *,
    equity_usdt: float,
    strategies_root: Optional[str] = None,
    strategies: Optional[list[str]] = None,
) -> Dict[str, float]:
    from src.config.multileg_sizing import resolve_multi_leg_unit_notionals

    sr = strategies_root or os.getenv(
        "MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies"
    )
    chop_exe = _chop_grid_execution_path_for_strategies_root(sr)
    trend_exe = _trend_scalp_execution_path_for_strategies_root(sr)
    return resolve_multi_leg_unit_notionals(
        ml,
        equity_usdt=float(equity_usdt),
        chop_grid_execution_path=chop_exe if chop_exe.is_file() else None,
        trend_scalp_execution_path=trend_exe if trend_exe.is_file() else None,
        strategies=strategies,
    )


def resolve_multi_leg_unit_notional_from_constitution(
    ml: Dict[str, Any],
    *,
    equity_usdt: float,
    strategies_root: Optional[str] = None,
    strategy: str = "chop_grid",
) -> float:
    from src.config.multileg_sizing import resolve_multi_leg_unit_notional

    sr = strategies_root or os.getenv(
        "MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies"
    )
    exe = _chop_grid_execution_path_for_strategies_root(sr)
    if strategy == "trend_scalp":
        trend_exe = _trend_scalp_execution_path_for_strategies_root(sr)
        return resolve_multi_leg_unit_notional(
            ml,
            equity_usdt=float(equity_usdt),
            trend_scalp_execution_path=trend_exe if trend_exe.is_file() else None,
            strategy="trend_scalp",
        )
    return resolve_multi_leg_unit_notional(
        ml,
        equity_usdt=float(equity_usdt),
        chop_grid_execution_path=exe if exe.is_file() else None,
        strategy="chop_grid",
    )


def load_multi_leg_backtest_risk_context(
    *,
    strategies_root: Optional[str] = None,
    constitution_yaml: Optional[str] = None,
    initial_capital: Optional[float] = None,
    strategy: str = "chop_grid",
):
    """Build BacktestAccountRiskTracker + unit_notional for multi-leg research scripts."""
    from src.time_series_model.core.constitution.account_risk_guard import (
        BacktestAccountRiskTracker,
    )

    sr = strategies_root or os.getenv("MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies")
    path = resolve_constitution_yaml(sr, override=constitution_yaml)
    ml = multi_leg_section(load_constitution_dict(path))
    account = ml.get("account") or {}
    equity = float(initial_capital if initial_capital is not None else account.get("equity_usdt", 10000.0) or 10000.0)
    if ml.get("unit_notional") is not None or isinstance(ml.get("sizing"), dict):
        unit = resolve_multi_leg_unit_notional_from_constitution(
            ml, equity_usdt=equity, strategies_root=sr, strategy=strategy
        )
    else:
        unit = 0.0
    tracker = BacktestAccountRiskTracker(
        limits=dict(ml.get("account_risk_limits") or {}),
        equity_usdt=equity,
    )
    return tracker, unit


def max_segment_starts_per_symbol_per_day_from_constitution(
    *,
    strategies_root: Optional[str] = None,
    constitution_yaml: Optional[str] = None,
) -> int:
    sr = strategies_root or os.getenv("MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies")
    path = resolve_constitution_yaml(sr, override=constitution_yaml)
    ml = multi_leg_section(load_constitution_dict(path))
    rs = ml.get("risk_limits") or {}
    if not isinstance(rs, dict):
        return 0
    raw = rs.get("max_segment_starts_per_symbol_per_day")
    return int(raw) if raw is not None else 0


def resolve_multi_leg_risk_limits_from_constitution(
    cfg: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    ml = multi_leg_section(cfg)
    account = ml.get("account") or {}
    if not isinstance(account, dict):
        account = {}
    rs = ml.get("risk_limits") or {}
    if not isinstance(rs, dict):
        rs = {}
    equity = _literal_from_env("MULTI_LEG_ACCOUNT_EQUITY_USDT")
    if equity is None:
        equity = _float_or_none(account.get("equity_usdt"))

    def _resolve_abs(key: str) -> Optional[float]:
        direct = _float_or_none(rs.get(key))
        if direct is not None:
            return direct
        pct = _float_or_none(rs.get(f"{key}_pct"))
        if pct is not None and equity is not None:
            return float(equity) * float(pct)
        return None

    # Place-size gate = chop_grid unit/equity from the same sizing formula
    # (segment_dd / (max_loss × 2 × levels)). Optional YAML override only.
    max_place_pct = _float_or_none(rs.get("max_place_notional_pct"))
    if max_place_pct is None:
        try:
            from src.config.multileg_sizing import chop_grid_unit_notional_pct

            unit_pct = chop_grid_unit_notional_pct(ml)
        except Exception:
            unit_pct = None
        if unit_pct is not None and unit_pct > 0.0:
            max_place_pct = float(unit_pct)

    return {
        "account_equity_usdt": equity,
        "max_drawdown_pct": (
            _literal_from_env("MULTI_LEG_MAX_DRAWDOWN_PCT")
            if _literal_from_env("MULTI_LEG_MAX_DRAWDOWN_PCT") is not None
            else _float_or_none(account.get("max_drawdown_pct"))
        ),
        "max_gross_notional": _resolve_abs("max_gross_notional"),
        "max_net_notional": _resolve_abs("max_net_notional"),
        "max_symbol_gross_notional": _resolve_abs("max_symbol_gross_notional"),
        "max_symbol_net_notional": _resolve_abs("max_symbol_net_notional"),
        "max_resting_orders": _float_or_none(rs.get("max_resting_orders")),
        "account_risk_limits": dict(ml.get("account_risk_limits") or {}),
        "max_place_notional_pct": max_place_pct,
    }


def resolve_multileg_sim_limits(cfg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Flatten constitution YAML into kwargs for ``simulate_account_with_constitution``."""
    ks = cfg.get("kill_switch") or {}
    if not isinstance(ks, dict):
        ks = {}
    ml = multi_leg_section(cfg)
    rs = ml.get("risk_limits") or {}
    if not isinstance(rs, dict):
        rs = {}
    acct_rs = ml.get("account_risk_limits") or {}
    if not isinstance(acct_rs, dict):
        acct_rs = {}

    ml_dd = resolve_multi_leg_risk_limits_from_constitution(cfg).get("max_drawdown_pct")
    ks_dd = _float_or_none(ks.get("max_dd"))
    if ml_dd is not None and ks_dd is not None:
        max_dd = min(float(ml_dd), float(ks_dd))
    else:
        max_dd = ml_dd if ml_dd is not None else ks_dd

    return {
        "max_drawdown_pct": max_dd,
        "daily_loss_limit_pct": _float_or_none(ks.get("daily_loss_limit")),
        "weekly_loss_limit_pct": _float_or_none(ks.get("weekly_loss_limit")),
        "monthly_loss_limit_pct": _float_or_none(ks.get("monthly_loss_limit")),
        "max_gross_notional_pct": _float_or_none(rs.get("max_gross_notional_pct")),
        "max_net_notional_pct": _float_or_none(rs.get("max_net_notional_pct")),
        "max_symbol_gross_notional_pct": _float_or_none(
            rs.get("max_symbol_gross_notional_pct")
        ),
        "max_symbol_net_notional_pct": _float_or_none(
            rs.get("max_symbol_net_notional_pct")
        ),
        "max_gross_leverage": _float_or_none(acct_rs.get("max_gross_leverage")),
    }


def pcm_resolve_registry_key(
    archetype_token: str, _me_logical: str, me_enabled_in_allowlist_fn
) -> str:
    """Map a constitution archetype token to the key used in ``LivePCM.register``."""
    del _me_logical  # legacy signature; ME always registers as logical ``me``
    tl = str(archetype_token).lower().strip()
    if not tl:
        return ""
    if me_enabled_in_allowlist_fn([tl]):
        return "me"
    return tl.split("-", 1)[0]


def pcm_archetype_priority_for_registry(
    cfg: Dict[str, Any],
    *,
    registry_keys: set[str],
    me_pkg: str,
    me_enabled_in_allowlist_fn,
) -> List[str]:
    """Resolve PCM registry key order from YAML (override or ``enabled_archetypes`` order)."""
    raw = intent_archetype_priority_tokens(cfg)
    if not raw:
        raw = ["bpc", "tpc", "srb", "me", "fbf", "msr", "lv"]
    out: List[str] = []
    seen_rk: set[str] = set()
    for p in raw:
        token = str(p).strip()
        if not token:
            continue
        rk = pcm_resolve_registry_key(token, me_pkg, me_enabled_in_allowlist_fn)
        if not rk or rk not in registry_keys or rk in seen_rk:
            continue
        seen_rk.add(rk)
        out.append(rk)
    if not out:
        for lk in ("tpc", "srb", "me", "bpc", "lv"):
            if lk in registry_keys and lk not in seen_rk:
                seen_rk.add(lk)
                out.append(lk)
    return out


def apply_multi_leg_args_from_constitution(args: Any) -> None:
    """Fill ``run_multi_leg_live`` argparse defaults from ``multi_leg:`` in constitution."""
    sr = os.getenv("MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies")
    ov = getattr(args, "constitution_yaml", None)
    if isinstance(ov, str) and not str(ov).strip():
        ov = None
    path = resolve_constitution_yaml(sr, override=ov)
    ml = multi_leg_section(load_constitution_dict(path))
    if not ml:
        return
    # Explicit ``strategies:`` (including empty list) overrides CLI defaults so
    # retired C sleeves cannot be re-enabled by argparse ``--strategies chop_grid``.
    if "strategies" in ml:
        strat = ml.get("strategies")
        if isinstance(strat, (list, tuple)):
            args.strategies = ",".join(
                str(x).strip() for x in strat if str(x).strip()
            )
        elif strat is None or strat == "":
            args.strategies = ""
        else:
            args.strategies = str(strat).strip()
    limits = resolve_multi_leg_risk_limits_from_constitution({"multi_leg": ml})
    for key in (
        "max_gross_notional",
        "max_net_notional",
        "max_symbol_gross_notional",
        "max_symbol_net_notional",
    ):
        if limits.get(key) is not None:
            setattr(args, key, float(limits[key] or 0.0))
    if limits.get("max_resting_orders") is not None:
        args.max_resting_orders = int(limits["max_resting_orders"] or 0)
    if limits.get("account_equity_usdt") is not None:
        setattr(args, "account_equity_usdt", float(limits["account_equity_usdt"] or 0.0))
    if limits.get("max_drawdown_pct") is not None:
        setattr(args, "max_drawdown_pct", float(limits["max_drawdown_pct"] or 0.0))
    # Only override the argparse default when the constitution actually specifies
    # sizing (explicit unit_notional or sizing.segment_dd_target); otherwise leave
    # the CLI/default value untouched (backward compatible with legacy configs).
    if ml.get("unit_notional") is not None or isinstance(ml.get("sizing"), dict):
        equity_for_sizing = float(
            getattr(args, "account_equity_usdt", None)
            or limits.get("account_equity_usdt")
            or 10000.0
        )
        strat_raw = getattr(args, "strategies", "") or ml.get("strategies") or ""
        if isinstance(strat_raw, (list, tuple)):
            strat_list = [str(x).strip() for x in strat_raw if str(x).strip()]
        else:
            strat_list = [
                s.strip()
                for s in str(strat_raw).replace("dual_add_trend", "trend_scalp").split(",")
                if s.strip()
            ]
        units = resolve_multi_leg_unit_notionals_from_constitution(
            ml,
            equity_usdt=equity_for_sizing,
            strategies_root=sr,
            strategies=strat_list or None,
        )
        setattr(args, "unit_notional_by_strategy", dict(units))
        args.unit_notional = float(
            units.get("chop_grid") or units.get("trend_scalp") or next(iter(units.values()))
        )
    from src.config.multileg_sizing import max_concurrent_multi_leg_symbols_from_ml
    cap = max_concurrent_multi_leg_symbols_from_ml(ml)
    if cap is not None:
        setattr(args, "max_concurrent_multi_leg_symbols", int(cap))
    rs = ml.get("risk_limits")
    if isinstance(rs, dict):
        cd = rs.get("strategy_switch_cooldown_bars")
        if cd is not None:
            setattr(args, "strategy_switch_cooldown_bars", int(cd))
        daily = rs.get("max_segment_starts_per_symbol_per_day")
        if daily is not None:
            setattr(args, "max_segment_starts_per_symbol_per_day", int(daily))
