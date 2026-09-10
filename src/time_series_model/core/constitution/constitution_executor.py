from __future__ import annotations

from datetime import datetime, timezone
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .state import ConstitutionState
from .runtime_state import (
    AddPositionRecord,
    ConstitutionRuntimeState,
    SlotRecord,
)
from .state_store import ConstitutionStatePaths, read_json, write_json
from .violation import ConstitutionViolation
from src.live_data_stream.constitution_config import load_constitution_dict
from src.order_management.storage import Storage

SLOT_RELEASE_REASONS = {
    "position_closed",
    "stop_loss_hit",
    "take_profit_hit",
    "order_failed",  # 下单失败 / 软拒绝未成交时释放预留 slot
    "stale_sync",  # 启动/定期同步: 服务端无对应持仓时释放
    "stale_sync_no_api",  # 定期同步: binance_api=None 时强制清空
}


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConstitutionConfig:
    """
    Runtime-ready view of constitution.yaml.
    This is intentionally a minimal subset for V1.1 enforcement.
    """

    version: int
    name: str
    constitution_hash: str

    # Kill-switch (hard stops)
    kill_enabled: bool
    daily_loss_limit: float
    weekly_loss_limit: float
    monthly_loss_limit: float
    max_dd: float
    max_turnover_mean: float
    max_cost_mean: float
    kill_on_any_hard_violation: bool
    cooldown_minutes: int
    daily_reset_timezone: Optional[str]
    # Soft day/week: derate risk, do not hard-halt (max_dd / monthly still hard).
    daily_soft_loss_limit: float = 0.0
    weekly_soft_loss_limit: float = 0.0
    derated_risk_per_slot: float = 0.0


def load_constitution_config(path: str | Path) -> ConstitutionConfig:
    p = Path(path)
    obj = load_constitution_dict(str(p)) or {}
    # Hash the *resolved* constitution so a parent edit reached via ``extends``
    # is visible in audit records.
    raw = yaml.safe_dump(obj, sort_keys=True, allow_unicode=True)
    ks = obj.get("kill_switch") or {}
    ss = obj.get("safety_state") or {}
    ss_ks = ks.get("safety_state") or {}
    cooldown_minutes = (
        ss.get("cooldown_minutes")
        or ss_ks.get("cooldown_minutes")
        or ks.get("cooldown_minutes")
        or 240
    )
    daily_reset_timezone = (
        ss.get("daily_reset_tz")
        or ss.get("daily_reset_timezone")
        or ss_ks.get("daily_reset_tz")
        or ss_ks.get("daily_reset_timezone")
        or ks.get("daily_reset_tz")
        or ks.get("daily_reset_timezone")
        or "UTC"
    )
    return ConstitutionConfig(
        version=int(obj.get("version", 1)),
        name=str(obj.get("name", "Constitution_v1")),
        constitution_hash=_sha256_text(raw),
        kill_enabled=bool(ks.get("enabled", True)),
        daily_loss_limit=float(ks.get("daily_loss_limit", 0.04)),
        weekly_loss_limit=float(ks.get("weekly_loss_limit", 0.08)),
        monthly_loss_limit=float(ks.get("monthly_loss_limit", 0.12)),
        max_dd=float(ks.get("max_dd", 0.20)),
        max_turnover_mean=float(ks.get("max_turnover_mean", 0.35)),
        max_cost_mean=float(ks.get("max_cost_mean", 0.002)),
        kill_on_any_hard_violation=bool(ks.get("kill_on_any_hard_violation", True)),
        cooldown_minutes=int(cooldown_minutes),
        daily_reset_timezone=str(daily_reset_timezone),
        daily_soft_loss_limit=float(ks.get("daily_soft_loss_limit") or 0.0),
        weekly_soft_loss_limit=float(ks.get("weekly_soft_loss_limit") or 0.0),
        derated_risk_per_slot=float(ks.get("derated_risk_per_slot") or 0.0),
    )


def _infer_base_dir(constitution_yaml: str | Path) -> Path:
    """Infer the base directory for resolving relative paths in constitution.

    策略: 找到包含 constitution.yaml 的最近的 config/ 的父目录。

    示例:
      config/constitution/constitution.yaml            → 项目根/
      live/highcap/config/constitution/constitution.yaml → live/highcap/
      /opt/mlbot/config/constitution/constitution.yaml   → /opt/mlbot/

    这样 persist_to: 'data/order_management.db' 在两侧分别解析到:
      研究: <项目根>/data/order_management.db
    实盘 (highcap bundle) 常与 OrderManager 共享 ``/app/data/order_management.db`` 挂载，
    YAML 中用 ``../data/order_management.db``（相对本 base）与之对齐；
    ``data/db/order_management.db`` 仅适合与 live_monitor 等同目录，勿与 MLBOT cwd 挂载混用。
    """
    # 1. 环境变量显式指定 (最高优先)
    env_base = os.getenv("MLBOT_LIVE_BASE_DIR")
    if env_base:
        return Path(env_base).resolve()

    p = Path(constitution_yaml).resolve()

    # 2. 向上找到包含 yaml 的最近 config/ 的父目录
    #    e.g. .../live/highcap/config/constitution/constitution.yaml
    #          ↑ config/ 的父目录是 live/highcap/ → 返回
    for parent in p.parents:
        if parent.name == "config" and p.is_relative_to(parent):
            return parent.parent

    # 3. Fallback: constitution.yaml → constitution/ → config/ → base
    try:
        return p.parents[2]
    except Exception:
        return p.parent


def canonical_order_management_db_path(
    *, constitution_base_dir: Path, raw_obj: Dict[str, Any]
) -> Path:
    """Single SQLite for orders + constitution safety/slots (.db persist targets).

    Precedence matches ``init_order_manager_from_env`` relative-path semantics:

    - If ``MLBOT_ORDER_MANAGEMENT_DB_PATH`` is set, resolve vs process cwd when relative.
    - Else resolve ``safety_state.persist_to`` (or nested kill_switch.safety_state) vs
      ``constitution_base_dir``, defaulting to ``data/order_management.db``.
    """
    env_p = (os.getenv("MLBOT_ORDER_MANAGEMENT_DB_PATH") or "").strip()
    if env_p:
        p = Path(env_p)
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()

    obj = raw_obj or {}
    ks = obj.get("kill_switch") or {}
    ss = obj.get("safety_state") or {}
    ss_ks = ks.get("safety_state") or {}
    raw = ss.get("persist_to") or ss_ks.get("persist_to") or "data/order_management.db"
    p = Path(str(raw))
    base = constitution_base_dir.resolve()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _strategy_keys(archetype: str, position_action: Optional[str] = None) -> list[str]:
    """Best-effort key candidates for per_strategy_limits lookup.

    PCM / TradeIntent often register a family-only archetype (e.g. ``bpc``) while
    constitution.yaml uses directional keys (``bpc-long``, ``bpc-short``). When
    ``position_action`` is LONG/BUY/SHORT, prepend ``{family}-long|short`` so
    add-position limits and risk caps resolve like LivePCM._limit_cfg_for_archetype.
    """
    key = str(archetype or "").strip().lower()
    parts = [p for p in key.split("-") if p]
    work = list(parts)
    if work and work[-1].endswith("t") and work[-1][:-1].isdigit():
        work = work[:-1]
    has_direction_token = len(work) >= 2 and work[1] in {"long", "short"}

    head: list[str] = []
    act = str(position_action or "").upper().strip()
    if not has_direction_token and work and act in ("LONG", "BUY", "SHORT"):
        side = "long" if act in ("LONG", "BUY") else "short"
        head.append(f"{work[0]}-{side}")

    cands: list[str] = head + [key]
    parts_tail = list(parts)
    if parts_tail and parts_tail[-1].endswith("t") and parts_tail[-1][:-1].isdigit():
        cands.append("-".join(parts_tail[:-1]))
        parts_tail = parts_tail[:-1]
    if len(parts_tail) >= 2 and parts_tail[1] in {"long", "short"}:
        cands.append("-".join(parts_tail[:2]))
    if parts_tail:
        cands.append(parts_tail[0])
    out: list[str] = []
    seen = set()
    for k in cands:
        kk = str(k).strip().lower()
        if kk and kk not in seen:
            seen.add(kk)
            out.append(kk)
    return out


class ConstitutionExecutor:
    """
    Single enforcement entry point.

    Rule: any capital/position/execution instruction must be validated here
    before it can be applied downstream.
    """

    def __init__(self, *, constitution_yaml: str | Path):
        self.constitution_yaml = str(constitution_yaml)
        self.cfg = load_constitution_config(constitution_yaml)
        self._base_dir = _infer_base_dir(constitution_yaml)
        self._raw_obj = load_constitution_dict(str(constitution_yaml)) or {}
        self._canonical_om_db = canonical_order_management_db_path(
            constitution_base_dir=Path(self._base_dir),
            raw_obj=self._raw_obj,
        )
        self._paths = self._load_state_paths()

    def meta(self) -> Dict[str, Any]:
        return {
            "constitution_yaml": str(self.constitution_yaml),
            "constitution_version": int(self.cfg.version),
            "constitution_name": str(self.cfg.name),
            "constitution_hash": str(self.cfg.constitution_hash),
        }

    def resolve_safety_db_path(self) -> Optional[Path]:
        return self._canonical_om_db

    def _resolve_per_strategy_limits(self) -> dict:
        """Return merged per-strategy limits from classic + spot sections."""
        obj = self._raw_obj or {}
        ra = obj.get("resource_allocation") or {}
        out = dict(ra.get("per_strategy_limits") or {})
        spot = obj.get("spot") or {}
        if isinstance(spot, dict):
            sl = spot.get("strategy_limits") or {}
            if isinstance(sl, dict):
                out.update(dict(sl))
        return out

    def resolve_max_new_entries_per_day(
        self, archetype: str, position_action: Optional[str] = None
    ) -> Optional[int]:
        """Return max_new_entries_per_day for a strategy, or None if unlimited."""
        limits = self._resolve_per_strategy_limits()
        for k in _strategy_keys(archetype, position_action):
            cand = limits.get(k) or {}
            if isinstance(cand, dict) and cand:
                v = cand.get("max_new_entries_per_day")
                if v is not None:
                    return int(v)
                return None
        return None

    def resolve_add_position_for_strategy(
        self, archetype: str, position_action: Optional[str] = None
    ) -> dict:
        """Compatibility accessor for add-position config.

        Global add_position_rules has been removed. This now returns a minimal
        config derived from per_strategy_limits for callers that still merge it
        with execution_profile.add_position.
        """
        limits = self._resolve_per_strategy_limits()
        strat_cfg: Dict[str, Any] = {}
        for k in _strategy_keys(archetype, position_action):
            cand = limits.get(k) or {}
            if isinstance(cand, dict) and cand:
                strat_cfg = cand
                break
        _mat = strat_cfg.get("max_add_times", 1)
        return {
            "enabled": bool(strat_cfg.get("allow_add_position", True)),
            # `or 1` here would silently turn an explicit max_add_times: 0
            # (fully disable adds) into 1 — 0 is falsy in Python.
            "max_add_times": int(_mat) if _mat is not None else 1,
            "require_locked_profit": bool(
                strat_cfg.get("require_locked_profit", False)
            ),
        }

    def resolve_risk_for_strategy(
        self, archetype: str, position_action: Optional[str] = None
    ) -> float:
        """Return effective risk fraction for a strategy.

        Logic: min(risk_per_slot, strategy.max_risk_per_trade)
        If strategy has no max_risk_per_trade, returns risk_per_slot.
        """
        obj = self._raw_obj or {}
        slots = obj.get("slots") or {}
        risk_per_slot = float(slots.get("risk_per_slot", 0.01))
        limits = self._resolve_per_strategy_limits()
        strat = {}
        for k in _strategy_keys(archetype, position_action):
            cand = limits.get(k) or {}
            if isinstance(cand, dict) and cand:
                strat = cand
                break
        strat_risk = strat.get("max_risk_per_trade")
        if strat_risk is not None:
            return min(risk_per_slot, float(strat_risk))
        return risk_per_slot

    def _load_state_paths(self) -> ConstitutionStatePaths:
        obj = self._raw_obj or {}
        slots = obj.get("slots") or {}
        slots_p = (slots.get("slot_state_tracking") or {}).get("persist_to") or None

        base = Path(self._base_dir).resolve()
        tmp = ConstitutionStatePaths(base_dir=base)

        def _split_persist_target(
            p: Optional[str],
        ) -> tuple[Optional[Path], Optional[Path]]:
            if not p:
                return None, None
            raw = str(p)
            if raw.lower().endswith(".db"):
                return None, tmp.resolve(raw)
            return tmp.resolve(raw), None

        slots_path, slots_db_path = _split_persist_target(slots_p)
        if slots_db_path is not None:
            slots_db_path = self._canonical_om_db
        add_position_db_path = slots_db_path
        return ConstitutionStatePaths(
            base_dir=base,
            slots_path=slots_path,
            slots_db_path=slots_db_path,
            add_position_path=None,
            add_position_db_path=add_position_db_path,
        )

    # -------------------------------------------------------------------------
    # Runtime state persistence (V1.1): slots / add-position
    # -------------------------------------------------------------------------
    def load_runtime_state(self) -> ConstitutionRuntimeState:
        st = ConstitutionRuntimeState()

        # Slots
        if self._paths.slots_db_path:
            storage = Storage(str(self._paths.slots_db_path))
            obj = storage.get_slots_state() or {}
        elif self._paths.slots_path:
            obj = read_json(self._paths.slots_path)
        else:
            obj = {}
        active = (obj.get("active") or {}) if isinstance(obj, dict) else {}
        if isinstance(active, dict):
            for pid, rec in active.items():
                if not pid:
                    continue
                r = rec or {}
                st.slots.active[str(pid)] = SlotRecord(
                    position_id=str(pid),
                    symbol=(
                        str(r.get("symbol")) if r.get("symbol") is not None else None
                    ),
                    archetype=(
                        str(r.get("archetype"))
                        if r.get("archetype") is not None
                        else None
                    ),
                    opened_at=(
                        str(r.get("opened_at"))
                        if r.get("opened_at") is not None
                        else None
                    ),
                    closed_at=(
                        str(r.get("closed_at"))
                        if r.get("closed_at") is not None
                        else None
                    ),
                    close_reason=(
                        str(r.get("close_reason"))
                        if r.get("close_reason") is not None
                        else None
                    ),
                )

        # Add-position
        if self._paths.add_position_db_path:
            storage = Storage(str(self._paths.add_position_db_path))
            obj = storage.get_add_position_state() or {}
        elif self._paths.add_position_path:
            obj = read_json(self._paths.add_position_path)
        else:
            obj = {}
        pos = (obj.get("positions") or {}) if isinstance(obj, dict) else {}
        if isinstance(pos, dict):
            for pid, rec in pos.items():
                if not pid:
                    continue
                r = rec or {}
                st.add_position.positions[str(pid)] = AddPositionRecord(
                    position_id=str(pid),
                    add_count=int(r.get("add_count", 0)),
                    locked_profit=bool(r.get("locked_profit", False)),
                    current_r=(
                        float(r["current_r"])
                        if r.get("current_r") is not None
                        else None
                    ),
                    updated_at=(
                        str(r.get("updated_at"))
                        if r.get("updated_at") is not None
                        else None
                    ),
                    last_add_at=(
                        str(r.get("last_add_at"))
                        if r.get("last_add_at") is not None
                        else None
                    ),
                )

        self._rehydrate_add_counts_from_open_positions(st)
        return st

    def save_runtime_state(self, st: ConstitutionRuntimeState) -> None:
        if self._paths.slots_db_path:
            storage = Storage(str(self._paths.slots_db_path))
            storage.upsert_slots_state(payload=st.slots.as_dict())
        if self._paths.slots_path:
            write_json(self._paths.slots_path, st.slots.as_dict())
        if self._paths.add_position_db_path:
            storage = Storage(str(self._paths.add_position_db_path))
            storage.upsert_add_position_state(payload=st.add_position.as_dict())
        if self._paths.add_position_path:
            write_json(self._paths.add_position_path, st.add_position.as_dict())

    def reserve_slot(
        self,
        *,
        st: ConstitutionRuntimeState,
        position_id: str,
        symbol: Optional[str] = None,
        archetype: Optional[str] = None,
        opened_at: Optional[str] = None,
    ) -> None:
        slots = (self._raw_obj or {}).get("slots") or {}
        if not bool(slots.get("enabled", True)):
            return
        slot_count = int(slots.get("slot_count", 2))
        pid = str(position_id).strip()
        if not pid:
            raise ConstitutionViolation(
                code="SLOT_BAD_ID", message="position_id is empty", context=self.meta()
            )
        if pid in st.slots.active:
            return
        if st.slots.active_count() >= slot_count:
            raise ConstitutionViolation(
                code="SLOT_FULL",
                message=f"Slot capacity exceeded: active={st.slots.active_count()} slot_count={slot_count}",
                context={"active_slots": list(st.slots.active.keys()), **self.meta()},
            )
        st.slots.active[pid] = SlotRecord(
            position_id=pid,
            symbol=str(symbol) if symbol is not None else None,
            archetype=str(archetype) if archetype is not None else None,
            opened_at=opened_at or _iso_now(),
        )

    def release_slot(
        self,
        *,
        st: ConstitutionRuntimeState,
        position_id: str,
        reason: str,
        closed_at: Optional[str] = None,
    ) -> None:
        if str(reason) not in SLOT_RELEASE_REASONS:
            return
        pid = str(position_id).strip()
        if not pid:
            return
        rec = st.slots.active.get(pid)
        if not rec:
            return
        # Closed record not persisted separately in v1; we just free the slot.
        st.slots.active.pop(pid, None)

    def _rehydrate_add_counts_from_open_positions(
        self, st: ConstitutionRuntimeState
    ) -> None:
        """Backfill add_count when add_position_state was empty after restart."""
        if self._paths.add_position_db_path is None:
            return
        try:
            storage = Storage(str(self._paths.add_position_db_path))
            open_rows = storage.get_open_positions() or []
        except Exception:
            return
        for row in open_rows:
            pid = str(getattr(row, "position_id", "") or "").strip()
            if not pid:
                continue
            sqlite_add = int(getattr(row, "add_count", 0) or 0)
            rec = st.add_position.positions.get(pid)
            existing = int(rec.add_count) if rec is not None else 0
            inferred = sqlite_add
            if inferred <= 0 and hasattr(storage, "count_filled_buy_orders_since"):
                try:
                    et = getattr(row, "entry_time", None)
                    n_buys = int(
                        storage.count_filled_buy_orders_since(
                            str(getattr(row, "symbol", "") or ""),
                            since=et,
                        )
                        or 0
                    )
                    inferred = max(0, n_buys - 1)
                except Exception:
                    inferred = 0
            merged = max(existing, sqlite_add, inferred)
            if merged <= 0:
                continue
            if rec is None:
                st.add_position.positions[pid] = AddPositionRecord(
                    position_id=pid,
                    add_count=merged,
                )
            elif merged > existing:
                rec.add_count = merged

    def validate_add_position(
        self,
        *,
        st: ConstitutionRuntimeState,
        position_id: str,
        archetype: Optional[str],
        current_r: Optional[float],
        locked_profit: Optional[bool] = None,
        position_action: Optional[str] = None,
        max_add_times_cap: Optional[int] = None,
    ) -> None:
        # 1. Check per-strategy allow_add_position
        arch_key = str(archetype or "").strip().lower()
        limits = self._resolve_per_strategy_limits()
        strat_cfg = {}
        for k in _strategy_keys(arch_key, position_action):
            cand = limits.get(k) or {}
            if isinstance(cand, dict) and cand:
                strat_cfg = cand
                break
        allow = strat_cfg.get("allow_add_position")
        if allow is not None and not bool(allow):
            raise ConstitutionViolation(
                code="ADD_POSITION_STRATEGY_FORBIDDEN",
                message=f"strategy '{arch_key}' does not allow add_position",
                context={"archetype": arch_key, **self.meta()},
            )

        pid = str(position_id).strip()
        if not pid:
            raise ConstitutionViolation(
                code="ADD_POSITION_BAD_ID",
                message="position_id is empty",
                context=self.meta(),
            )
        rec = st.add_position.positions.get(pid)
        add_count = int(rec.add_count) if rec is not None else 0
        # `or 1` would silently coerce an explicit max_add_times: 0 (disable
        # adds entirely) back to 1 — 0 is falsy in Python. Only missing/None
        # should fall back to the default.
        _mat_raw = strat_cfg.get("max_add_times", 1)
        max_add_times = int(_mat_raw) if _mat_raw is not None else 1
        if max_add_times_cap is not None:
            max_add_times = min(max_add_times, int(max_add_times_cap))
        if add_count >= max_add_times:
            raise ConstitutionViolation(
                code="ADD_POSITION_MAX_TIMES",
                message="max_add_times exceeded",
                context={"position_id": pid, "add_count": add_count, **self.meta()},
            )
        if bool(strat_cfg.get("require_locked_profit", False)) and not bool(
            locked_profit
        ):
            raise ConstitutionViolation(
                code="ADD_POSITION_LOCKED_PROFIT_REQUIRED",
                message="locked_profit is required before add_position",
                context={
                    "position_id": pid,
                    "archetype": arch_key,
                    "locked_profit": bool(locked_profit),
                    **self.meta(),
                },
            )

        # 全局 add_position_rules 已移除，仅保留 per_strategy_limits 约束。

    def record_add_position(
        self,
        *,
        st: ConstitutionRuntimeState,
        position_id: str,
        current_r: Optional[float],
        locked_profit: Optional[bool] = None,
    ) -> None:
        pid = str(position_id).strip()
        if not pid:
            return
        inferred_locked = bool(locked_profit) if locked_profit is not None else False
        rec = st.add_position.positions.get(pid)
        add_count = int(rec.add_count) if rec is not None else 0
        _now = _iso_now()
        st.add_position.positions[pid] = AddPositionRecord(
            position_id=pid,
            add_count=int(add_count + 1),
            locked_profit=inferred_locked,
            current_r=current_r,
            updated_at=_now,
            last_add_at=_now,
        )

    def validate_drawdown(self, *, state: ConstitutionState) -> None:
        """
        V1.1: implement kill-switch checks that are universally applicable.
        """

        if not self.cfg.kill_enabled:
            return

        reasons: List[str] = []
        if state.drawdown is not None and float(state.drawdown) > float(
            self.cfg.max_dd
        ):
            reasons.append("max_dd")
        if float(state.daily_loss) >= float(self.cfg.daily_loss_limit):
            reasons.append("daily_loss_limit")
        if float(state.weekly_loss) >= float(self.cfg.weekly_loss_limit):
            reasons.append("weekly_loss_limit")
        if float(state.monthly_loss) >= float(self.cfg.monthly_loss_limit):
            reasons.append("monthly_loss_limit")
        if bool(state.data_bad):
            reasons.append("data_bad")
        if bool(state.hard_violation):
            reasons.append("hard_violation")

        if reasons and bool(self.cfg.kill_on_any_hard_violation):
            raise ConstitutionViolation(
                code="KILL_SWITCH",
                message=f"Kill-switch triggered: {', '.join(reasons)}",
                context={"reasons": reasons, **state.as_dict(), **self.meta()},
            )

    def validate_capital_allocation(
        self,
        *,
        state: ConstitutionState,
        per_mode_budget: Dict[str, float],
        per_symbol_budget: Dict[str, float],
        overrides: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Guard PCM output. V1.1 keeps this intentionally strict and simple.
        """

        self.validate_drawdown(state=state)

        # Budgets must be non-negative and finite.
        for k, v in (per_mode_budget or {}).items():
            fv = float(v)
            if not (fv >= 0.0):
                raise ConstitutionViolation(
                    code="CAPITAL_BUDGET_NEGATIVE",
                    message=f"per_mode_budget[{k}] is negative: {fv}",
                    context={
                        "per_mode_budget": per_mode_budget,
                        **state.as_dict(),
                        **self.meta(),
                    },
                )

        for k, v in (per_symbol_budget or {}).items():
            fv = float(v)
            if not (fv >= 0.0):
                raise ConstitutionViolation(
                    code="SYMBOL_BUDGET_NEGATIVE",
                    message=f"per_symbol_budget[{k}] is negative: {fv}",
                    context={
                        "per_symbol_budget": per_symbol_budget,
                        **state.as_dict(),
                        **self.meta(),
                    },
                )

        # Human override must be audited (tag+reason), otherwise it's an illegal bypass.
        if overrides:
            for o in overrides:
                tag = str((o or {}).get("tag") or "").strip()
                reason = str((o or {}).get("reason") or "").strip()
                if not tag or not reason:
                    raise ConstitutionViolation(
                        code="HUMAN_OVERRIDE_UNAUDITED",
                        message="Human override must include non-empty tag and reason.",
                        context={"override": o, **state.as_dict(), **self.meta()},
                    )
