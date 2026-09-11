"""Family → eval harness contract.

A harness here is the *evaluation* machine (clock, fills, costs, universe),
not the Cursor agent loop. Promote evidence is valid only when the family
runs on its registered harness. Using B ``event_backtest`` to rank Rolling
P&L is a contract error — the same class as swapping an LLM eval set.

See ``docs/design/2026-08-23_human_hypothesis_vs_auto_mine_CN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

# Names that may appear in DECISION.md front-matter ``harness:``.
KNOWN_HARNESSES: Tuple[str, ...] = (
    "event_backtest",
    "phase1_scan_only",
    # Buy-and-hold panel (entry date × universe × fixed years).
    "cohort_hold",
)


@dataclass(frozen=True)
class HarnessSpec:
    """One strategy family's evaluation contract."""

    family: str
    harness: str
    entry: str
    forbid_event_backtest: bool = False
    notes: str = ""


# Public extract: only ship families that still have a strategy pack.
# Private detector families (srb / tpc / bpc / fade / rolling / ashare) stay
# in the private repo's registry.
_SPECS: Tuple[HarnessSpec, ...] = (
    HarnessSpec(
        "ma_cross",
        "event_backtest",
        "python -m scripts.event_backtest --strategy ma_cross --no-kill-switch",
        notes="Public textbook MA demo — not a live sleeve",
    ),
    HarnessSpec(
        "tenbagger_smallcap",
        "cohort_hold",
        "python scripts/research/cohort_hold.py",
        forbid_event_backtest=True,
        notes="A-share small-cap hold panel; not the 2h event clock",
    ),
    HarnessSpec(
        "funding_fade",
        "event_backtest",
        "python -m scripts.event_backtest --variant-grid "
        "config/experiments/20260910_funding_fade/funding_fade_grid.yaml",
        notes="Fade funding z-score; own pack, not ma_cross",
    ),
    HarnessSpec(
        "ashare_monday",
        "event_backtest",
        "python -m scripts.event_backtest --variant-grid "
        "config/experiments/20260911_ashare_monday_rebound/monday_rebound_grid.yaml",
        notes="A-share Monday rebound; own pack, not ma_cross",
    ),
    HarnessSpec(
        "btc_lead_alts",
        "event_backtest",
        "python -m scripts.event_backtest --variant-grid "
        "config/experiments/20260911_btc_lead_ai_alts/btc_lead_alts_grid.yaml",
        notes="BTC prior-bar lead into AI alts; own pack, not ma_cross",
    ),
    HarnessSpec(
        "p99_bb_chase",
        "event_backtest",
        "python -m scripts.event_backtest --variant-grid "
        "config/experiments/20260911_p99_bb_break_chase/p99_bb_chase_grid.yaml",
        notes="P99 notional + Bollinger chase; own pack, not ma_cross",
    ),
)

_ALIASES: Dict[str, str] = {
    "ma": "ma_cross",
}

_BY_FAMILY: Dict[str, HarnessSpec] = {s.family: s for s in _SPECS}


class WrongHarnessError(ValueError):
    """Family was pointed at an eval machine that cannot produce promote evidence."""


def normalize_family(name: str) -> str:
    key = str(name or "").strip().lower()
    return _ALIASES.get(key, key)


def spec_for(family: str) -> Optional[HarnessSpec]:
    return _BY_FAMILY.get(normalize_family(family))


def required_harness(family: str) -> Optional[str]:
    spec = spec_for(family)
    return spec.harness if spec else None


def event_backtest_reject_reason(families: Iterable[str]) -> Optional[str]:
    """If any family is forbidden on event_backtest, return a one-block error."""
    bad: list[HarnessSpec] = []
    for raw in families:
        spec = spec_for(raw)
        if spec and spec.forbid_event_backtest:
            bad.append(spec)
    if not bad:
        return None
    lines = [
        "WRONG HARNESS: event_backtest cannot produce promote evidence for:",
    ]
    for spec in bad:
        lines.append(f"  - {spec.family} → use {spec.harness}")
        lines.append(f"    {spec.entry}")
        if spec.notes:
            lines.append(f"    ({spec.notes})")
    lines.append("See src/research/harness_registry.py.")
    return "\n".join(lines)


def refuse_event_backtest(families: Iterable[str]) -> None:
    """Raise :class:`WrongHarnessError` when EB is the wrong court."""
    reason = event_backtest_reject_reason(families)
    if reason:
        raise WrongHarnessError(reason)


def harness_mismatch_issue(family: Optional[str], harness: Optional[str]) -> Optional[str]:
    """Front-matter check: declared harness must match the family's contract."""
    if not family or not harness:
        return None
    spec = spec_for(family)
    if not spec:
        return None
    if harness == spec.harness:
        return None
    return (
        f"harness: {harness!r} does not match family {spec.family!r} "
        f"(required {spec.harness!r})"
    )


def list_specs() -> Tuple[HarnessSpec, ...]:
    return _SPECS


def event_backtest_families() -> frozenset[str]:
    """Families whose registered court is event_backtest."""
    return frozenset(s.family for s in _SPECS if s.harness == "event_backtest")
