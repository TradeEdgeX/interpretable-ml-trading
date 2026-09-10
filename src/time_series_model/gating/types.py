from __future__ import annotations

from enum import Enum


class GateDecision(str, Enum):
    """
    Discrete gate decisions.

    Important: gate outputs must NOT be continuous scores directly scaling position size,
    otherwise you silently create a high-DOF control path (hard to audit / easy to bypass).
    """

    VETO = "VETO"
    THROTTLE_25 = "THROTTLE_25"
    THROTTLE_50 = "THROTTLE_50"
    ALLOW = "ALLOW"
