"""Minimal Storage so ConstitutionExecutor can load empty slot state."""

from __future__ import annotations

from typing import Any, Dict, Optional


class Storage:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_slots_state(self) -> Dict[str, Any]:
        return {}

    def get_add_position_state(self) -> Dict[str, Any]:
        return {}

    def save_slots_state(self, *_a: Any, **_k: Any) -> None:
        return None

    def save_add_position_state(self, *_a: Any, **_k: Any) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        def _noop(*_a: Any, **_k: Any) -> Optional[Any]:
            return None

        return _noop
