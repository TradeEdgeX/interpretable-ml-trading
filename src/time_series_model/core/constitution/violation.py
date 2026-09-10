from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=False)
class ConstitutionViolation(Exception):
    """
    Raised when constitution is violated.

    This is an Exception on purpose: violations are meant to hard-stop the pipeline
    unless explicitly allowed by policy (e.g., in research-only mode).
    """

    code: str
    message: str
    context: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": str(self.code),
            "message": str(self.message),
            "context": dict(self.context or {}),
        }

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
