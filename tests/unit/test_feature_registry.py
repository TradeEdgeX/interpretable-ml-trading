"""
Tests for the feature registry module.
"""

import pytest
import pandas as pd
import numpy as np

from src.features.registry import (
    FeatureRegistry,
    register_feature,
    get_feature_func,
    get_compute_func,
    list_registered_features,
    get_registry,
    ensure_features_registered,
    _ensure_features_registered,
)

# 在模块加载时就注册所有特征
ensure_features_registered()


class TestFeatureRegistry:
    @pytest.fixture
    def clean_registry(self):
        """Provide a clean registry for testing."""
        registry = get_registry()
        registry.clear()
        yield registry
        registry.clear()

    def test_register_and_get(self, clean_registry):
        """Test basic registration and retrieval."""

        def my_feature_func(x):
            return x * 2

        clean_registry.register("my_feature", my_feature_func)
        retrieved = clean_registry.get("my_feature")
        assert retrieved is my_feature_func

    def test_get_nonexistent(self, clean_registry):
        """Test getting a nonexistent feature returns None."""
        result = clean_registry.get("nonexistent")
        assert result is None

    def test_get_or_raise(self, clean_registry):
        """Test get_or_raise raises on missing feature."""

        def my_func(x):
            return x

        clean_registry.register("exists", my_func)
        assert clean_registry.get_or_raise("exists") is my_func

        with pytest.raises(ValueError, match="Unknown feature function"):
            clean_registry.get_or_raise("does_not_exist")

    def test_list_features(self, clean_registry):
        """Test listing registered features."""

        def func1():
            pass

        def func2():
            pass

        def func3():
            pass

        clean_registry.register("feat1", func1, category="cat_a")
        clean_registry.register("feat2", func2, category="cat_a")
        clean_registry.register("feat3", func3, category="cat_b")

        all_features = clean_registry.list_features()
        assert set(all_features) == {"feat1", "feat2", "feat3"}

        cat_a = clean_registry.list_features(category="cat_a")
        assert set(cat_a) == {"feat1", "feat2"}

    def test_metadata(self, clean_registry):
        """Test metadata storage."""

        def my_func():
            pass

        clean_registry.register(
            "my_feat",
            my_func,
            category="test",
            description="A test feature",
            inputs=["close"],
            outputs=["result"],
        )

        meta = clean_registry.get_metadata("my_feat")
        assert meta["category"] == "test"
        assert meta["description"] == "A test feature"
        assert meta["inputs"] == ["close"]
        assert meta["outputs"] == ["result"]

    def test_no_overwrite_by_default(self, clean_registry):
        """Test that re-registering without overwrite keeps existing."""

        def func1():
            return 1

        def func2():
            return 2

        clean_registry.register("same_name", func1)
        clean_registry.register("same_name", func2)  # Should be ignored

        retrieved = clean_registry.get("same_name")
        assert retrieved is func1

    def test_overwrite_explicit(self, clean_registry):
        """Test that overwrite=True replaces existing."""

        def func1():
            return 1

        def func2():
            return 2

        clean_registry.register("same_name", func1)
        clean_registry.register("same_name", func2, overwrite=True)

        retrieved = clean_registry.get("same_name")
        assert retrieved is func2


class TestRegisterDecorator:
    @pytest.fixture
    def clean_registry(self):
        registry = get_registry()
        registry.clear()
        yield registry
        registry.clear()

    def test_decorator_registration(self, clean_registry):
        """Test that @register_feature decorator works."""

        @register_feature("decorated_func", category="test")
        def my_decorated_function(x):
            return x * 2

        # Should be registered
        func = clean_registry.get("decorated_func")
        assert func is my_decorated_function

        # Should have metadata attached
        assert my_decorated_function._feature_name == "decorated_func"
        assert my_decorated_function._feature_category == "test"

    def test_decorator_preserves_function(self, clean_registry):
        """Test that decorator preserves original function behavior."""

        @register_feature("multiply_func")
        def multiply(x: int, factor: int = 2) -> int:
            return x * factor

        assert multiply(5) == 10
        assert multiply(5, 3) == 15


class TestGetFeatureFunc:
    @pytest.fixture
    def clean_registry(self):
        registry = get_registry()
        registry.clear()
        yield registry
        registry.clear()

    def test_get_from_registry(self, clean_registry):
        """Test getting function from registry."""

        @register_feature("in_registry")
        def my_func():
            pass

        func = get_feature_func("in_registry")
        assert func is my_func

    def test_not_found_raises(self, clean_registry):
        """Test that missing function raises ValueError."""
        with pytest.raises(ValueError, match="Unknown feature function"):
            get_feature_func("definitely_does_not_exist_12345")

    def test_get_compute_func_alias(self, clean_registry):
        """Test that get_compute_func is an alias for get_feature_func."""

        @register_feature("aliased_func")
        def my_func():
            pass

        func = get_compute_func("aliased_func")
        assert func is my_func


class TestAllFeaturesRegistration:
    """Test that all features are correctly registered via decorators."""

    def setup_method(self):
        """Ensure all feature modules are imported before each test."""
        _ensure_features_registered(force=True)

    def test_feature_count_at_least_200(self):
        """Verify we have at least 200 features registered."""
        registry = get_registry()

        assert (
            registry.count >= 200
        ), f"Expected at least 200 features, got {registry.count}"

    def test_key_features_exist(self):
        """Verify key feature functions can be retrieved."""
        key_features = [
            # Baseline
            "compute_atr",
            "compute_rsi",
            "compute_macd",
            "compute_roc_5_from_series",
            "compute_bb_width_features_from_series",
            # Order flow
            "extract_order_flow_features",
            "compute_vpin_derived_features_from_base",
            "compute_trade_cluster_derived_features_from_base",
            # Volatility
            "extract_extended_volatility_features",
            "compute_vol_raw_features_from_series",
            # Liquidity
            "extract_liquidity_features",
            "compute_liquidity_void_features_from_series",
            # WPT
            "extract_wpt_features",
            # Interaction
            "compute_atr_ratio_from_series",
            "compute_compression_score_from_series",
            # TA-Lib
            "compute_talib_indicator",
            "compute_talib_sma",
        ]

        registry = get_registry()
        missing = []
        for feat in key_features:
            func = registry.get(feat)
            if func is None:
                missing.append(feat)

        assert len(missing) == 0, f"Missing key features: {missing}"

    def test_all_functions_are_callable(self):
        """Verify all registered functions are callable."""
        registry = get_registry()

        failed = []
        for func_name in registry.list_features():
            func = registry.get(func_name)
            if not callable(func):
                failed.append(func_name)

        assert len(failed) == 0, f"Non-callable functions: {failed}"
