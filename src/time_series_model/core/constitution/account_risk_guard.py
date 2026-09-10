"""Account-level gross leverage / margin guards for live and backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class AccountRiskSnapshot:
    equity: float
    gross_notional: float
    available_margin: Optional[float] = None
    used_initial_margin: float = 0.0


def resolve_account_risk_limits(raw: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = dict(raw or {})
    if not cfg:
        return {"enabled": False}
    out = dict(cfg)
    out["enabled"] = bool(cfg.get("enabled", False))
    out["fail_closed"] = bool(cfg.get("fail_closed", True))
    out["margin_stress_leverage"] = float(cfg.get("margin_stress_leverage", 5.0) or 5.0)
    return out


def _cfg_float(cfg: Mapping[str, Any], key: str) -> Optional[float]:
    val = cfg.get(key)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _first_float(obj: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        try:
            val = obj.get(key)
        except Exception:
            val = None
        if val is None or val == "":
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def gross_notional_from_position_rows(
    positions: Iterable[Mapping[str, Any]],
) -> float:
    gross = 0.0
    for pos in positions or []:
        notional = _first_float(pos, ("notional", "_entry_notional_usdt"))
        if notional is None:
            size = _first_float(pos, ("size", "contracts", "_qty_base", "qty")) or 0.0
            mark = (
                _first_float(pos, ("mark_price", "markPrice"))
                or _first_float(pos, ("entry_price", "entryPrice"))
                or 0.0
            )
            notional = size * mark
        gross += abs(float(notional or 0.0))
    return gross


def snapshot_from_binance_balance(
    *,
    balance: Mapping[str, Any],
    positions: Iterable[Mapping[str, Any]],
    features_fallback: Optional[Mapping[str, Any]] = None,
) -> AccountRiskSnapshot:
    info = balance.get("info") if isinstance(balance, dict) else {}
    usdt = balance.get("USDT") if isinstance(balance, dict) else {}
    if not isinstance(info, dict):
        info = {}
    if not isinstance(usdt, dict):
        usdt = {}
    features_fallback = features_fallback or {}
    # G0 / account equity numeraire = wallet/realized (locked 2026-07-26).
    # Prefer totalWalletBalance; do not min() with totalMarginBalance (MTM) —
    # floating loss must not inflate daily_loss/max_dd or reshape opens.
    # wallet=0 wipe must not fall through to margin residue. Never chain with
    # ``or`` (float 0.0 is a valid wipe).
    wallet = _first_float(info, ("totalWalletBalance",))
    if wallet is None:
        wallet = _first_float(usdt, ("total",))
    margin = _first_float(info, ("totalMarginBalance",))
    if wallet is not None:
        equity = max(0.0, wallet)
    elif margin is not None:
        equity = max(0.0, margin)
    else:
        fb = _first_float(features_fallback, ("equity",))
        equity = 0.0 if fb is None else max(0.0, fb)
    available = _first_float(info, ("availableBalance", "maxWithdrawAmount"))
    if available is None:
        available = _first_float(usdt, ("free",))
    if available is None:
        available = _first_float(features_fallback, ("account_available_balance",))
    used_initial_raw = _first_float(
        info,
        (
            "totalPositionInitialMargin",
            "totalInitialMargin",
            "totalOpenOrderInitialMargin",
        ),
    )
    if used_initial_raw is None:
        used_initial_raw = _first_float(usdt, ("used",))
    used_initial = 0.0 if used_initial_raw is None else float(used_initial_raw)
    return AccountRiskSnapshot(
        equity=float(equity),
        gross_notional=gross_notional_from_position_rows(positions),
        available_margin=available,
        used_initial_margin=float(used_initial),
    )


def snapshot_for_backtest(
    *,
    equity_usdt: float,
    gross_notional: float,
    margin_stress_leverage: float = 5.0,
) -> AccountRiskSnapshot:
    eq = float(max(0.0, equity_usdt))
    gross = float(max(0.0, gross_notional))
    stress = max(1e-9, float(margin_stress_leverage or 5.0))
    used_initial = gross / stress
    available = max(0.0, eq - used_initial) if eq > 0 else None
    return AccountRiskSnapshot(
        equity=eq,
        gross_notional=gross,
        available_margin=available,
        used_initial_margin=used_initial,
    )


def evaluate_account_risk(
    *,
    limits: Mapping[str, Any],
    snapshot: AccountRiskSnapshot,
    proposed_notional: float,
) -> List[str]:
    cfg = resolve_account_risk_limits(limits)
    if not bool(cfg.get("enabled", False)):
        return []
    proposed = float(max(0.0, proposed_notional))
    if proposed <= 0:
        return []

    equity = float(snapshot.equity or 0.0)
    if equity <= 0:
        return ["account_equity_unavailable"]

    current_gross = float(max(0.0, snapshot.gross_notional or 0.0))
    projected_gross = current_gross + proposed
    projected_gross_lev = projected_gross / equity
    margin_stress_leverage = float(cfg.get("margin_stress_leverage", 5.0) or 5.0)
    projected_order_margin = proposed / max(1e-9, margin_stress_leverage)
    used_initial = float(max(0.0, snapshot.used_initial_margin or 0.0))
    projected_initial_pct = (used_initial + projected_order_margin) / equity
    projected_available_pct: Optional[float] = None
    if snapshot.available_margin is not None:
        projected_available_pct = (
            float(snapshot.available_margin) - projected_order_margin
        ) / equity

    violations: List[str] = []
    max_gross_lev = _cfg_float(cfg, "max_gross_leverage")
    if max_gross_lev is not None and projected_gross_lev > max_gross_lev:
        violations.append(
            f"projected_gross_leverage={projected_gross_lev:.3f}>{max_gross_lev:.3f}"
        )
    max_initial_pct = _cfg_float(cfg, "max_projected_initial_margin_pct")
    if max_initial_pct is not None and projected_initial_pct > max_initial_pct:
        violations.append(
            f"projected_initial_margin_pct={projected_initial_pct:.3f}>{max_initial_pct:.3f}"
        )
    min_available_pct = _cfg_float(cfg, "min_projected_available_margin_pct")
    if (
        min_available_pct is not None
        and projected_available_pct is not None
        and projected_available_pct < min_available_pct
    ):
        violations.append(
            f"projected_available_margin_pct={projected_available_pct:.3f}<{min_available_pct:.3f}"
        )
    return violations


class BacktestAccountRiskTracker:
    """Track open gross exposure for offline multi-leg / research simulators."""

    def __init__(
        self,
        *,
        limits: Mapping[str, Any] | None,
        equity_usdt: float,
    ) -> None:
        self.limits = resolve_account_risk_limits(limits)
        self.equity_usdt = float(max(0.0, equity_usdt))
        self.open_gross_notional = 0.0
        self.rejected_count = 0

    @property
    def enabled(self) -> bool:
        return bool(self.limits.get("enabled", False))

    def allow_open(self, proposed_notional: float) -> tuple[bool, str]:
        if not self.enabled:
            return True, ""
        snap = snapshot_for_backtest(
            equity_usdt=self.equity_usdt,
            gross_notional=self.open_gross_notional,
            margin_stress_leverage=float(
                self.limits.get("margin_stress_leverage", 5.0) or 5.0
            ),
        )
        violations = evaluate_account_risk(
            limits=self.limits,
            snapshot=snap,
            proposed_notional=proposed_notional,
        )
        if violations:
            self.rejected_count += 1
            return False, violations[0]
        return True, ""

    def on_open(self, notional: float) -> None:
        self.open_gross_notional += float(max(0.0, notional))

    def on_close(self, notional: float) -> None:
        self.open_gross_notional = max(
            0.0, self.open_gross_notional - float(max(0.0, notional))
        )


AccountSnapshotProvider = Callable[[], AccountRiskSnapshot]
