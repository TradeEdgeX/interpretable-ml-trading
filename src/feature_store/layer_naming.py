from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional


def default_layer_from_config(
    config_dir: str | Path,
    *,
    prefix: str = "features",
    features_file: str = "features.yaml",
) -> str:
    """
    AUTO FeatureStore layer id derived from config content.

    Intended use:
    - FeatureStore is a materialized "wide table" per (symbol, timeframe, month).
    - The layer is the dataset id. AUTO keeps it stable as long as feature definitions don't change.

    Format:  ``{prefix}_{archetype}_{timeframe}_{hash10}``
    Example: ``features_me_60T_696d36d50f``

    - *archetype*: config directory name (e.g. ``me``, ``bpc``)
    - *timeframe*: from ``meta.yaml`` strategy.timeframe (e.g. ``60T``, ``240T``)
    - *hash*: SHA-1 of config files (stable as long as feature definitions don't change)

    We hash:
    - {config_dir}/features.yaml (main driver)
    - {config_dir}/meta.yaml (optional)
    - {config_dir}/feature_contract.yaml (optional)
    - repo-level config/feature_dependencies.yaml (feature DAG / compute mapping)
    """
    cfg = Path(config_dir)
    if cfg.is_file():
        cfg = cfg.parent

    # --- Extract archetype + timeframe for human-readable prefix ---
    archetype = cfg.name  # directory name = archetype id (me / bpc / fer / lv)
    if archetype == "_shared":
        archetype = "tree_full" if features_file == "features_all.yaml" else "tree_core"

    timeframe: str | None = None
    meta_path = cfg / "meta.yaml"
    meta_bytes: bytes | None = None
    if meta_path.exists():
        meta_bytes = meta_path.read_bytes()
        # Quick regex extraction avoids yaml dependency in this module
        tf_match = re.search(rb'timeframe:\s*["\']?([\w]+)["\']?', meta_bytes)
        if tf_match:
            timeframe = tf_match.group(1).decode("utf-8")

    # --- Compute content hash ---
    parts: list[bytes] = []
    features_path = cfg / features_file
    if not features_path.exists() and features_file != "features.yaml":
        raise FileNotFoundError(f"Feature manifest not found: {features_path}")
    if features_path.exists():
        parts.append(features_path.read_bytes())
    elif (cfg / "features.yaml").exists():
        parts.append((cfg / "features.yaml").read_bytes())
    if meta_bytes is not None:
        parts.append(meta_bytes)
    for fname in ("feature_contract.yaml",):
        p = cfg / fname
        if p.exists():
            parts.append(p.read_bytes())

    # repo-level feature registry (for invalidation when DAG changes)
    repo_root = cfg
    for _ in range(8):
        if (repo_root / "config").exists():
            break
        if repo_root.parent == repo_root:
            break
        repo_root = repo_root.parent
    global_deps = repo_root / "config" / "feature_dependencies.yaml"
    if global_deps.exists():
        parts.append(global_deps.read_bytes())

    if not parts:
        parts = [str(cfg).encode("utf-8")]

    h = hashlib.sha1(b"\n---\n".join(parts)).hexdigest()[:10]

    # --- Assemble layer name ---
    name_parts = [prefix]
    if archetype:
        name_parts.append(archetype)
    if timeframe:
        name_parts.append(timeframe)
    name_parts.append(h)
    return "_".join(name_parts)


def resolve_layer_name(
    layer: str | None,
    config_dir: str | Path,
    *,
    features_file: str = "features.yaml",
) -> str:
    """
    Resolve layer name: auto-generate from config if not specified.
    
    This is a unified helper function used by all scripts to handle layer name resolution.
    It ensures consistent behavior whether scripts are called directly or through CLI.
    
    Args:
        layer: Layer name (None or empty string = auto-generate, otherwise use as-is)
        config_dir: Config directory path (used when auto-generating)
        
    Returns:
        Resolved layer name (always returns a string, never None)
        
    Example:
        >>> resolve_layer_name(None, "config/nnmultihead/path_primitives_4h_80h_min")
        'features_291404fba6'
        >>> resolve_layer_name("", "config/nnmultihead/path_primitives_4h_80h_min")
        'features_291404fba6'
        >>> resolve_layer_name("heavy_v6", "config/...")
        'heavy_v6'
    """
    if layer is None or (isinstance(layer, str) and layer.strip() == ""):
        return default_layer_from_config(config_dir, features_file=features_file)
    return str(layer)


def detect_layer_for_strategy(
    strategy: Optional[str] = None,
    features_store_root: str = "feature_store",
    timeframe: Optional[str] = None,
) -> Optional[str]:
    """
    Auto-detect the latest feature store layer matching a strategy (and optionally timeframe).

    Scans ``features_store_root`` for directories with ``.meta.json`` files
    whose ``config_dir`` contains ``/strategies/{strategy}``.
    When *timeframe* is given the candidates are further narrowed to layers
    whose meta ``timeframe`` field matches.
    Returns the most recently modified matching layer name, or ``None``
    if nothing matches.

    This is the shared implementation used by both the CLI
    (``mlbot gate apply-archetype``) and standalone scripts
    (e.g. ``backtest_execution_layer.py``).
    """
    fs_root = Path(features_store_root)
    if not fs_root.exists():
        return None

    layers = [d for d in fs_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not layers:
        return None

    matching_layers = []
    for layer_dir in layers:
        meta_file = fs_root / f"{layer_dir.name}.meta.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                config_dir = meta.get("config_dir", "")
                meta_tf = meta.get("timeframe", "")
                # strategy filter — exact match to avoid me-short matching me-short-240T
                if strategy:
                    _markers = (
                        f"/strategies/{strategy}/",
                        f"/strategies/{strategy}",
                        f"/bad-candidates/{strategy}/",
                        f"/bad-candidates/{strategy}",
                    )
                    ok = False
                    for _marker in _markers:
                        _idx = config_dir.find(_marker)
                        if _idx < 0:
                            continue
                        _after = config_dir[_idx + len(_marker) :]
                        if _after and not _after.startswith("/"):
                            continue
                        ok = True
                        break
                    if not ok:
                        continue
                # timeframe filter
                if timeframe and meta_tf and meta_tf != timeframe:
                    continue
                matching_layers.append(layer_dir)
            except Exception:
                pass

    if matching_layers:
        latest = max(matching_layers, key=lambda p: p.stat().st_mtime)
        return latest.name
    elif layers and not timeframe:
        # Fallback: return the most recent layer regardless of strategy
        latest = max(layers, key=lambda p: p.stat().st_mtime)
        return latest.name

    return None


def detect_layer_timeframe(
    layer: str,
    features_store_root: str = "feature_store",
) -> Optional[str]:
    """Read the *timeframe* field from a layer's .meta.json, or None."""
    meta_file = Path(features_store_root) / f"{layer}.meta.json"
    if not meta_file.exists():
        return None
    try:
        with open(meta_file) as f:
            return json.load(f).get("timeframe")
    except Exception:
        return None

