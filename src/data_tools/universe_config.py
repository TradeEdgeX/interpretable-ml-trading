from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass(frozen=True)
class UniverseConfig:
    quote: str
    universe_sets: dict

    def resolve_tokens(
        self,
        *,
        universe_set: str,
        groups: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        if universe_set not in self.universe_sets:
            raise KeyError(
                f"Unknown universe_set={universe_set}. Available: {list(self.universe_sets.keys())}"
            )
        data = self.universe_sets[universe_set]
        grp = (data.get("groups") or {}) if isinstance(data, dict) else {}
        if not grp:
            raise ValueError(f"Invalid universe_set={universe_set}: missing groups")

        if groups:
            out = {}
            for g in groups:
                if g not in grp:
                    raise KeyError(f"Unknown group={g}. Available: {list(grp.keys())}")
                out[g] = list(grp[g] or [])
            return out
        return {k: list(v or []) for k, v in grp.items()}

    def resolve_symbols_usdt(
        self,
        *,
        universe_set: str,
        groups: Optional[List[str]] = None,
    ) -> List[str]:
        token_groups = self.resolve_tokens(universe_set=universe_set, groups=groups)
        tokens: List[str] = []
        for g in token_groups.values():
            tokens.extend(g)
        # de-dup preserve order
        tokens = list(
            dict.fromkeys([t.strip().upper() for t in tokens if str(t).strip()])
        )

        quote = str(self.quote).upper().strip()
        if not quote:
            quote = "USDT"
        return [f"{t}{quote}" if not t.endswith(quote) else t for t in tokens]


def load_universe_config(path: str | Path) -> UniverseConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    quote = obj.get("quote", "USDT")
    universe_sets = obj.get("universe_sets", {}) or {}
    return UniverseConfig(quote=str(quote), universe_sets=universe_sets)
