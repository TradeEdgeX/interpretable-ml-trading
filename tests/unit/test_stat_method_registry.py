import numpy as np

from scripts.stat_method_registry import (
    canonicalize_method_name,
    evaluate_rr_split_method,
    normalize_fallback_methods,
    normalize_method,
    standardize_method_list,
)


def test_normalize_method_aliases():
    assert normalize_method("mean_effect") == "mean_effect"
    assert normalize_method("distribution_ks") == "distribution_ks"
    assert canonicalize_method_name("welch_ttest") == "welch_ttest"
    assert canonicalize_method_name("tail_bad_rate_ratio") == "tail_bad_rate_ratio"


def test_normalize_fallback_methods_dedup_and_order():
    methods = [
        "mean_effect",
        "distribution_ks",
        "mean_effect",
        "welch_ttest",
        "distribution_ks",
        "upside_positive_rate_ratio",
    ]
    out = normalize_fallback_methods(methods, default=["distribution_ks"])
    assert out == [
        "mean_effect",
        "distribution_ks",
        "welch_ttest",
        "upside_positive_rate_ratio",
    ]
    out2 = standardize_method_list(methods, default=["ks"])
    assert out2 == out


def test_evaluate_effect_method_passes_with_positive_effect():
    rr_pass = np.array([1.2, 0.8, 0.9, 1.0])
    rr_reject = np.array([0.1, 0.2, 0.3, 0.0])
    passed, score, extra = evaluate_rr_split_method("mean_effect", rr_pass, rr_reject)
    assert passed is True
    assert score > 0
    assert extra["effect"] > 0


def test_evaluate_positive_rr_method():
    rr_pass = np.array([1.1, 1.0, 0.95, 0.2, 0.3])
    rr_reject = np.array([0.1, 0.2, 0.4, 0.5, 0.6])
    passed, score, extra = evaluate_rr_split_method(
        "upside_positive_rate_ratio",
        rr_pass,
        rr_reject,
        thresholds={"positive_rr_threshold": 0.8, "min_positive_lift": 1.05},
    )
    assert passed is True
    assert score >= 1.05
    assert extra["positive_lift"] >= 1.05
