"""Strategy directory layout and default pipeline resolution (ADR §3.2)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

# Never copy these when cloning a strategy package (research / rolling / feature search).
STRATEGY_PACKAGE_SKIP_DIR_NAMES = frozenset(
    {
        "data",
        "results",
        "docker",
        "feature_store",
        "live",
        "vendor",
        "cache",
        ".git",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
    }
)

STRATEGY_PACKAGE_SKIP_FILE_SUFFIXES = (
    ".zip",
    ".whl",
    ".parquet",
    ".db",
    ".sqlite",
    ".pkl",
    ".pickle",
)

# 全局默认（无 --strategy）：多策略 PCM 编排；历史遗留单体研究包仍可用 ``pipelines/research_pipeline.yaml`` 显式传入。
DEFAULT_PCM_ORCHESTRATE_REL = Path("config/pipelines/pcm_orchestrate_2h.yaml")
LEGACY_RESEARCH_PIPELINE_REL = Path("config/pipelines/research_pipeline.yaml")

# Canonical packaged research filenames (purpose-first stem + dotted suffix).
PIPELINE_CALIBRATE_ROLL_MARKER = "calibrate_roll.default.yaml"
PIPELINE_RESEARCH_ROLL_MARKER = "research_roll.features_on.yaml"
PIPELINE_VALIDATE_STATIC_MARKER = "validate_static.full_study.yaml"

RESEARCH_PIPELINE_PROBE_NAMES = (
    PIPELINE_CALIBRATE_ROLL_MARKER,
    PIPELINE_RESEARCH_ROLL_MARKER,
    "pipeline.yaml",
)

# Default ``research/<stem>.yaml`` when resolving a bare strategy directory (CLI / multileg).
PACKAGED_PROFILE_DEFAULT_STEM = "calibrate_roll.default"

# Required packaged policy files checked by ``validate_strategy_package``.
PACKAGED_POLICY_REQUIRED_PROFILE_STEMS: Tuple[str, ...] = (
    PACKAGED_PROFILE_DEFAULT_STEM,
    "research_roll.features_on",
    "validate_static.full_study",
)


def packaged_research_yaml_name(profile: str | None = None) -> str:
    """Return basename ``*.yaml`` under ``research/`` for a dotted stem or full filename."""
    raw = str(profile).strip() if profile is not None else ""
    stem = Path(raw.replace("-", "_")).name if raw else ""
    if not stem:
        return PIPELINE_CALIBRATE_ROLL_MARKER
    return stem if stem.endswith(".yaml") else f"{stem}.yaml"


def packaged_profile_yaml_is_validate_static(fname: str) -> bool:
    """True when ``research/<fname>`` is a static validation pack (runs default ``full``)."""
    s = str(fname or "").strip()
    return s.startswith("validate_static.") and s.endswith(".yaml")

# Filenames treated as rolling-style profiles (these forbid naive ``full`` unless overridden).
RESEARCH_STAGE_FULL_BLOCKED_LEAVES = frozenset(
    {
        PIPELINE_CALIBRATE_ROLL_MARKER,
        PIPELINE_RESEARCH_ROLL_MARKER,
    }
)

def load_yaml_dict(path: Path, *, strict: bool = True) -> Dict[str, Any]:
    """Load a YAML dict from disk.

    - ``strict=True``: missing file / parse errors raise ``ValueError``.
    - ``strict=False``: returns ``{}`` on missing file / parse errors.
    """
    if not path.exists():
        if strict:
            raise ValueError(f"配置文件不存在: {path}")
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        if strict:
            raise ValueError(f"读取配置失败: {path}") from exc
        return {}
    if not isinstance(raw, dict):
        if strict:
            raise ValueError(f"配置文件格式错误(期望 dict): {path}")
        return {}
    return raw


def load_yaml_extends_chain(path: Path, *, strict: bool = True) -> Dict[str, Any]:
    """Load YAML with ``extends`` chain (child overlays parent)."""
    if not path.exists() and not strict:
        return {}
    chain: List[Dict[str, Any]] = []
    cur = path.resolve()
    visited: Set[Path] = set()
    for _ in range(64):
        if cur in visited:
            raise ValueError(f"extends 循环引用: {cur}")
        visited.add(cur)
        raw = load_yaml_dict(cur, strict=strict)
        ext = raw.pop("extends", None)
        chain.append(raw)
        if not ext:
            break
        nxt = (cur.parent / str(ext).strip()).resolve()
        if not nxt.is_file():
            raise ValueError(f"extends 指向的文件不存在: {ext!r}（自 {cur}）")
        cur = nxt
    merged: Dict[str, Any] = {}
    for layer in reversed(chain):
        merged = deep_merge_dicts(merged, layer)
    return merged


def resolve_strategy_profile_path(
    config_dir: Path, profile: str | None = None
) -> Path:
    """Resolve ``config_dir/research/<packaged profile>.yaml`` (dotted stems allowed)."""
    return config_dir / "research" / packaged_research_yaml_name(profile)


# Renamed packages: old slug → current directory name (experiments / history).
_STRATEGY_PACKAGE_SLUG_ALIASES: dict[str, str] = {
    "chop_largebar_fade": "largebar_fade",
    "fade": "largebar_fade",
}


def resolve_strategy_package_under_root(
    strategies_parent: Path,
    strategy_slug: str,
    *,
    allow_bad_candidates: bool = True,
) -> Path:
    """Resolve ``strategies_parent/<slug>`` using optional fallbacks.

    Research tree (repo ``config/strategies``): unless disabled, prefers
    ``<parent>/bad-candidates/<slug>/`` after the canonical path so archived slugs
    stay addressable via ``--strategy <slug>``.

    Live deployment tree (``live/highcap/config/strategies``): pass
    ``allow_bad_candidates=False`` so ``bad-candidates/`` is never loaded on
    ticker paths—those bundles are research-only archives.

    After primary (and optionally bad-candidates), legacy ``me-long/`` resolves
    the ``me`` slug when present. ``chop_largebar_fade`` / ``fade`` resolve to ``largebar_fade/``.
    """

    parent = strategies_parent.expanduser().resolve()
    slug = str(strategy_slug).strip()
    primary = (parent / slug).resolve()
    if primary.is_dir():
        return primary
    alias = _STRATEGY_PACKAGE_SLUG_ALIASES.get(slug.lower() if slug else "")
    if alias:
        aliased = (parent / alias).resolve()
        if aliased.is_dir():
            return aliased
    if allow_bad_candidates:
        cand = parent / "bad-candidates" / slug
        if cand.is_dir():
            return cand.resolve()
        if alias:
            cand_alias = parent / "bad-candidates" / alias
            if cand_alias.is_dir():
                return cand_alias.resolve()
    if slug == "me":
        legacy_long = parent / "me-long"
        if legacy_long.is_dir():
            return legacy_long.resolve()
    return primary


def packaged_strategy_rel_path(project_root: Path, strategy_slug: str) -> str:
    """Repo-relative posix path under ``project_root`` to the packaged strategy directory."""
    p = strategy_packaged_root(project_root, strategy_slug)
    root = project_root.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_strategy_config_input(
    path: Path, *, default_profile: str | None = None
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Resolve ``config_dir/profile_path/engine_path`` from a user input path.

    Supports:
    - strategy directory
    - research profile yaml (``.../research/*.yaml``)
    - other yaml paths under the strategy tree (treated as profile-like for loaders)
    """
    if path.is_dir():
        cfg_dir = path
        dp = PACKAGED_PROFILE_DEFAULT_STEM if default_profile is None else default_profile
        prof_path = resolve_strategy_profile_path(cfg_dir, dp)
        if not prof_path.exists():
            meta_path = cfg_dir / "meta.yaml"
            if meta_path.exists():
                prof_path = meta_path
        return cfg_dir, prof_path if prof_path.exists() else None, None
    if path.parent.name == "research":
        return path.parent.parent, path, None
    return path.parent, path, None


def strategy_packaged_root(project_root: Path, strategy_slug: str) -> Path:
    """Packaged strategy tree under ``config/strategies/<slug>/`` (with bad-candidates fallback)."""
    base = project_root / "config" / "strategies"
    return resolve_strategy_package_under_root(base, strategy_slug)


def resolve_default_pipeline_config(
    project_root: Path,
    strategy_slug: Optional[str],
    explicit_config: Optional[Path],
) -> Tuple[Path, List[str]]:
    """Resolve pipeline YAML path for ``mlbot pipeline`` when ``--config`` is omitted.

    Order (per ADR §3.2): ``research/calibrate_roll.default.yaml`` →
    ``research/research_roll.features_on.yaml`` → ``research/pipeline.yaml``;
    each may be a thin pointer via top-level ``extends: <relative path>``.

    Returns ``(absolute_path, warning_messages)``.
    """
    warnings: List[str] = []
    if explicit_config is not None:
        p = explicit_config if explicit_config.is_absolute() else (project_root / explicit_config)
        return p.resolve(), warnings
    if not strategy_slug or not str(strategy_slug).strip():
        fb = (project_root / DEFAULT_PCM_ORCHESTRATE_REL).resolve()
        warnings.append(
            f"No --strategy/--config; using default {fb.relative_to(project_root)}"
        )
        return fb, warnings

    slug = str(strategy_slug).strip()
    research = strategy_packaged_root(project_root, slug) / "research"
    for name in RESEARCH_PIPELINE_PROBE_NAMES:
        candidate = research / name
        if not candidate.is_file():
            continue
        resolved, w = _resolve_research_pipeline_marker(candidate, project_root)
        warnings.extend(w)
        return resolved, warnings

    fb = (project_root / DEFAULT_PCM_ORCHESTRATE_REL).resolve()
    warnings.append(
        f"No {', '.join(RESEARCH_PIPELINE_PROBE_NAMES)} under "
        f"{research.relative_to(project_root)}; falling back to "
        f"{fb.relative_to(project_root)}"
    )
    return fb, warnings


def _resolve_research_pipeline_marker(
    marker: Path, project_root: Path
) -> Tuple[Path, List[str]]:
    raw = load_yaml_dict(marker, strict=False)
    ext = raw.get("extends")
    if isinstance(ext, str) and ext.strip():
        target = (marker.parent / ext.strip()).resolve()
        if target.is_file():
            rel_m = marker.relative_to(project_root)
            rel_t = target.relative_to(project_root)
            return target, [f"Resolved pipeline via {rel_m} → {rel_t}"]
        fb = (project_root / DEFAULT_PCM_ORCHESTRATE_REL).resolve()
        return fb, [
            f"{marker.relative_to(project_root)}: extends {ext!r} missing; "
            f"falling back to {fb.relative_to(project_root)}"
        ]
    return marker.resolve(), []


def is_research_turbo_or_slow_yaml(config_path: Path) -> bool:
    """True for rolling packaged profiles whose default CLI stage is never ``full``."""
    try:
        parts = config_path.resolve().parts
    except Exception:
        parts = config_path.parts
    if len(parts) < 2 or parts[-2] != "research":
        return False
    leaf = parts[-1]
    return leaf in RESEARCH_STAGE_FULL_BLOCKED_LEAVES


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dicts (override wins); lists and scalars are replaced."""
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def strategy_package_copy_ignore(_directory: str, names: List[str]) -> Set[str]:
    """``copytree`` ignore callback: skip repo-scale dirs and bulky artifacts."""
    ignored: Set[str] = set()
    for name in names:
        if name in STRATEGY_PACKAGE_SKIP_DIR_NAMES:
            ignored.add(name)
            continue
        lower = str(name).lower()
        if any(lower.endswith(suffix) for suffix in STRATEGY_PACKAGE_SKIP_FILE_SUFFIXES):
            ignored.add(name)
    return ignored


def copy_strategy_package(
    src: Path,
    dst: Path,
    *,
    dirs_exist_ok: bool = False,
    ignore: Optional[Callable[[str, List[str]], Set[str]]] = None,
) -> None:
    """Copy a strategy directory without dragging data/results/docker artifacts."""
    src_path = Path(src).resolve()
    dst_path = Path(dst).resolve()
    if not src_path.is_dir():
        raise FileNotFoundError(f"strategy package not found: {src_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and not dirs_exist_ok:
        shutil.rmtree(dst_path)
    ignore_cb = ignore or strategy_package_copy_ignore
    shutil.copytree(
        src_path,
        dst_path,
        ignore=ignore_cb,
        dirs_exist_ok=dirs_exist_ok,
        ignore_dangling_symlinks=True,
    )

