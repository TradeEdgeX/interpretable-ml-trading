"""
Feature Registry with Decorator-based Registration.

This module provides a centralized registry for feature computation functions,
allowing automatic registration via decorators instead of maintaining a
hand-written mapping dictionary.

Usage:
    # In feature module (e.g., baseline_features.py):
    from src.features.registry import register_feature

    @register_feature("compute_rsi")
    def compute_rsi_from_series(close: pd.Series, period: int = 14) -> pd.Series:
        ...

    # In feature loader:
    from src.features.registry import get_feature_func, ensure_features_registered

    ensure_features_registered()  # Import all feature modules
    func = get_feature_func("compute_rsi")
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable, Dict, List, Optional, Any, Set
from functools import wraps

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Central registry for feature computation functions.

    Provides:
    - Decorator-based registration (@register_feature)
    - Automatic discovery of feature modules
    - Thread-safe singleton pattern
    """

    _instance: Optional["FeatureRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "FeatureRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._functions: Dict[str, Callable] = {}
            cls._instance._metadata: Dict[str, Dict[str, Any]] = {}
            cls._instance._imported_modules: Set[str] = set()
        return cls._instance

    def register(
        self,
        name: str,
        func: Callable,
        *,
        category: str = "default",
        description: str = "",
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a feature function.

        Args:
            name: Unique name for the feature function
            func: The callable to register
            category: Category for grouping (e.g., "baseline", "orderflow")
            description: Human-readable description
            inputs: Expected input column names
            outputs: Output column names
            overwrite: If True, overwrite existing registration
        """
        if name in self._functions and not overwrite:
            existing = self._functions[name]
            if existing is not func:
                logger.warning(
                    f"Feature '{name}' already registered. "
                    f"Use overwrite=True to replace. Keeping existing."
                )
                return

        self._functions[name] = func
        self._metadata[name] = {
            "category": category,
            "description": description,
            "inputs": inputs or [],
            "outputs": outputs or [],
            "module": getattr(func, "__module__", "unknown"),
        }

    def get(self, name: str) -> Optional[Callable]:
        """Get a registered function by name."""
        return self._functions.get(name)

    def get_or_raise(self, name: str) -> Callable:
        """Get a registered function, raise if not found."""
        func = self._functions.get(name)
        if func is None:
            available = list(self._functions.keys())[:20]
            raise ValueError(
                f"Unknown feature function: '{name}'. "
                f"Available (first 20): {available}"
            )
        return func

    def list_features(self, category: Optional[str] = None) -> List[str]:
        """List all registered feature names, optionally filtered by category."""
        if category:
            return [
                name
                for name, meta in self._metadata.items()
                if meta.get("category") == category
            ]
        return list(self._functions.keys())

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a registered feature."""
        return self._metadata.get(name)

    def import_module(self, module_path: str) -> None:
        """Import a module to trigger its @register_feature decorators.

        Handles both 'src.features.*' and 'features.*' paths for compatibility
        with different PYTHONPATH configurations (Docker vs local).
        """
        if module_path in self._imported_modules:
            return

        # Try original path first
        try:
            importlib.import_module(module_path)
            self._imported_modules.add(module_path)
            logger.debug(f"Imported feature module: {module_path}")
            return
        except ImportError:
            pass

        # If original path starts with 'src.', try without it
        # (for Docker where PYTHONPATH=/workspace/src)
        if module_path.startswith("src."):
            alt_path = module_path[4:]  # Remove 'src.' prefix
            try:
                importlib.import_module(alt_path)
                self._imported_modules.add(module_path)
                logger.debug(f"Imported feature module: {alt_path} (alt path)")
                return
            except ImportError:
                pass

        # If path doesn't start with 'src.', try with it
        # (for local development)
        else:
            alt_path = f"src.{module_path}"
            try:
                importlib.import_module(alt_path)
                self._imported_modules.add(module_path)
                logger.debug(f"Imported feature module: {alt_path} (alt path)")
                return
            except ImportError:
                pass

        logger.warning(
            f"Failed to import feature module {module_path}: No module named '{module_path}'"
        )

    def clear(self) -> None:
        """Clear all registrations (mainly for testing)."""
        self._functions.clear()
        self._metadata.clear()
        self._imported_modules.clear()

    @property
    def count(self) -> int:
        """Number of registered features."""
        return len(self._functions)


# Global registry instance
_registry = FeatureRegistry()


def register_feature(
    name: str,
    *,
    category: str = "default",
    description: str = "",
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator to register a feature computation function.

    Usage:
        @register_feature("compute_rsi", category="baseline")
        def compute_rsi_from_series(close: pd.Series, period: int = 14) -> pd.Series:
            ...

        # Or with more metadata:
        @register_feature(
            "compute_vpin",
            category="orderflow",
            description="Volume-Synchronized Probability of Informed Trading",
            inputs=["ticks"],
            outputs=["vpin", "vpin_ma_20"],
        )
        def compute_vpin_from_ticks(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        _registry.register(
            name=name,
            func=func,
            category=category,
            description=description,
            inputs=inputs,
            outputs=outputs,
        )
        # Attach registration info to the function for introspection
        func._feature_name = name
        func._feature_category = category
        return func

    return decorator


def get_feature_func(name: str) -> Callable:
    """
    Get a feature function by name.

    Automatically calls ensure_features_registered() if needed.

    Args:
        name: Feature function name

    Returns:
        The feature function

    Raises:
        ValueError: If function not found
    """
    # Auto-register features if not done yet
    ensure_features_registered()

    func = _registry.get(name)
    if func is not None:
        return func

    raise ValueError(
        f"Unknown feature function: '{name}'. "
        f"Not found in registry ({_registry.count} registered)."
    )


# Alias for backward compatibility
get_compute_func = get_feature_func


def list_registered_features(category: Optional[str] = None) -> List[str]:
    """List all registered feature names."""
    return _registry.list_features(category)


def get_registry() -> FeatureRegistry:
    """Get the global feature registry instance."""
    return _registry


# =============================================================================
# Auto-discovery of feature modules
# =============================================================================

# List of modules to import for feature registration
FEATURE_MODULES = [
    # Baseline features
    "src.features.time_series.baseline_features",
    # Market-cap / cross-sectional normalization features
    "src.features.time_series.market_cap_features",
    # Funding rate features
    "src.features.time_series.funding_rate_features",
    # Open Interest features
    "src.features.time_series.open_interest_features",
    # Order flow features
    "src.features.time_series.utils_order_flow_features",
    # Volatility features
    "src.features.time_series.utils_volatility_features",
    # Liquidity features
    "src.features.time_series.utils_liquidity_features",
    # Interaction features
    "src.features.time_series.utils_interaction_features",
    # WPT features
    "src.features.time_series.utils_wpt_features",
    # Hilbert features
    "src.features.time_series.utils_hilbert_features",
    # Hurst features
    "src.features.time_series.utils_hurst_features",
    # Spectrum features
    "src.features.time_series.utils_spectrum_features",
    # DTW features
    "src.features.time_series.utils_dtw_features",
    "src.features.time_series.utils_dtw_individual",
    # GARCH features
    "src.features.time_series.utils_garch_features",
    # EVT features
    "src.features.time_series.utils_evt_features",
    # Volume profile
    "src.features.time_series.utils_volume_profile",
    # Reflexivity features
    "src.features.time_series.reflexivity_features",
    # Feature wrappers
    "src.features.loader.feature_wrappers",
    # interaction_feature_wrappers.py removed - functions already in utils_interaction_features.py
    # common_derived_feature_wrappers.py removed - functions already in utils_interaction_features.py
    "src.features.loader.talib_feature_wrappers",
    # dl_feature_wrappers.py removed - functions moved to dl_sequence_features.py
    "src.features.time_series.dl_sequence_features",
    # Selector utils
    "src.features.loader.selector_utils",
    # Alpha (TS-adapted subset)
    "src.features.time_series.alpha_factors.alpha_ts_wrappers",
    # Box-structure (consolidation detector) features
    "src.features.time_series.box_structure_features",
    # Session & Microstructure features
    "src.features.time_series.session_features",
    # T5 orderbook wall + liquidation proxy
    "src.features.time_series.wall_features",
    "src.features.time_series.t5_liquidation_proxy_features",
]

_features_registered = False


def ensure_features_registered() -> None:
    """
    Ensure all feature modules are imported, triggering @register_feature decorators.

    This function is idempotent - calling it multiple times has no effect after
    the first call.

    Usage:
        from src.features.registry import ensure_features_registered, get_feature_func

        ensure_features_registered()
        func = get_feature_func("compute_rsi")
    """
    global _features_registered
    if _features_registered:
        return

    for module_path in FEATURE_MODULES:
        _registry.import_module(module_path)

    _features_registered = True
    logger.info(f"Feature registry initialized with {_registry.count} features")


def _ensure_features_registered(force: bool = False) -> None:
    """
    Alias for ensure_features_registered with force option.

    Args:
        force: If True, re-import all modules even if already registered
    """
    global _features_registered
    if force:
        _features_registered = False
        _registry.clear()  # This also clears _imported_modules

        # Force re-import by reloading modules
        import importlib

        for module_path in FEATURE_MODULES:
            try:
                mod = importlib.import_module(module_path)
                importlib.reload(mod)
            except Exception:
                pass
        _features_registered = True
    else:
        ensure_features_registered()
