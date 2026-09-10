from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.config.strategy_layout import (
    PACKAGED_POLICY_REQUIRED_PROFILE_STEMS,
    resolve_strategy_profile_path,
)


@dataclass(frozen=True)
class StrategyValidationIssue:
    strategy_name: str
    code: str
    message: str
    path: str = ""


_MULTILEG_TYPES = frozenset({"grid", "dual_add_trend", "trend_scalp"})


def strategy_type_from_entry(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("strategy_type", "") or "").strip().lower()


def resolve_strategy_config_dir(entry: Dict[str, Any], project_root: Path) -> Optional[Path]:
    cfg_dir = str((entry or {}).get("config", "") or "").strip()
    if not cfg_dir:
        return None
    root = Path(cfg_dir)
    if not root.is_absolute():
        root = project_root / root
    return root


def validate_strategy_package(
    *,
    strategy_name: str,
    strategy_type: str,
    strategy_cfg: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Path],
    required_profiles: Sequence[str] = PACKAGED_POLICY_REQUIRED_PROFILE_STEMS,
) -> List[StrategyValidationIssue]:
    issues: List[StrategyValidationIssue] = []
    st = str(strategy_type or "").strip().lower()

    if not config_dir:
        issues.append(
            StrategyValidationIssue(
                strategy_name=strategy_name,
                code="missing_config_dir",
                message="missing config directory",
            )
        )
        return issues
    if not config_dir.exists():
        issues.append(
            StrategyValidationIssue(
                strategy_name=strategy_name,
                code="config_dir_not_found",
                message=f"config directory does not exist: {config_dir}",
                path=str(config_dir),
            )
        )
        return issues

    required_files = ["features.yaml"]
    required_files.extend(
        resolve_strategy_profile_path(config_dir, p).relative_to(config_dir).as_posix()
        for p in required_profiles
    )
    if st in _MULTILEG_TYPES:
        required_files.extend(
            [
                "archetypes/regime.yaml",
                "archetypes/prefilter.yaml",
                "archetypes/execution.yaml",
            ]
        )
        scfg = strategy_cfg or {}
        if bool(scfg.get("has_prefilter", False)):
            required_files.append("features_prefilter.yaml")

    for rel in required_files:
        target = config_dir / rel
        if not target.exists():
            issues.append(
                StrategyValidationIssue(
                    strategy_name=strategy_name,
                    code="missing_required_file",
                    message=f"missing required file: {target}",
                    path=str(target),
                )
            )
    return issues


def validate_pipeline_strategy_packages(
    *,
    pipeline_cfg: Dict[str, Any],
    project_root: Path,
    allow_strategy_types: Optional[Iterable[str]] = None,
    required_profiles: Sequence[str] = PACKAGED_POLICY_REQUIRED_PROFILE_STEMS,
) -> List[StrategyValidationIssue]:
    issues: List[StrategyValidationIssue] = []
    allowed = {str(t).strip().lower() for t in (allow_strategy_types or []) if str(t).strip()}
    for name, scfg in (pipeline_cfg.get("strategies") or {}).items():
        st = strategy_type_from_entry(scfg)
        if allowed and st not in allowed:
            issues.append(
                StrategyValidationIssue(
                    strategy_name=str(name),
                    code="unsupported_strategy_type",
                    message=f"unsupported strategy_type={st!r}",
                )
            )
            continue
        config_dir = resolve_strategy_config_dir(scfg or {}, project_root)
        issues.extend(
            validate_strategy_package(
                strategy_name=str(name),
                strategy_type=st,
                strategy_cfg=scfg if isinstance(scfg, dict) else {},
                config_dir=config_dir,
                required_profiles=required_profiles,
            )
        )
    return issues
