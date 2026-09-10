from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import os


# Default paths (same as scripts/run_live.py). Override via env vars.
_DEFAULT_CONSTITUTION_YAML = "config/constitution/constitution.yaml"


def resolve_live_runtime_paths(
    *,
    runtime_config_path: Optional[str | Path] = None,
    defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Resolve live runtime paths. Precedence: env var > defaults > built-in default.
    Used by legacy EventDrivenStrategy. New pipeline (run_live.py) uses env directly.
    """
    _ = runtime_config_path  # unused; kept for API compatibility
    defaults = defaults or {}
    constitution = str(
        os.getenv(
            "MLBOT_CONSTITUTION_YAML",
            defaults.get("constitution_yaml", _DEFAULT_CONSTITUTION_YAML),
        )
    )
    return {"constitution_yaml": constitution}
