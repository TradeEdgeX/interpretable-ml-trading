"""Resolve feature_dependencies.yaml for a strategy archetypes directory."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_DEPS = "config/feature_dependencies.yaml"


def resolve_feature_deps_path(
    archetypes_dir: str | None,
    *,
    repo_root: Path | None = None,
) -> str:
    """Prefer ``<strategies_root>/feature_dependencies.yaml`` when present."""
    if not archetypes_dir:
        return _DEFAULT_DEPS
    arch = Path(archetypes_dir).resolve()
    # .../config_experiments/<tree>/<strategy>/archetypes -> strategies_root
    strategies_root = arch.parent.parent
    override = strategies_root / "feature_dependencies.yaml"
    if override.is_file():
        return str(override)
    return _DEFAULT_DEPS
