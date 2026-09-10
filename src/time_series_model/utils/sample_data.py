"""Utility helpers for generating synthetic factor datasets."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def create_sample_data(
    n_samples: int = 1000,
    n_factors: int = 60,
    seed: int = 42,
    return_dataframe: bool = False,
) -> (
    Tuple[np.ndarray, np.ndarray, List[str]]
    | Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]
):
    """Generate synthetic factor data for demos and tests."""

    rng = np.random.default_rng(seed)

    factor_names: List[str] = []
    categories = ["momentum", "volatility", "mean_reversion", "trend", "volume"]
    for i in range(n_factors):
        category = categories[i % len(categories)]
        factor_names.append(f"{category}_{i+1}")

    X = rng.standard_normal((n_samples, n_factors))

    momentum_idx = [i for i, name in enumerate(factor_names) if "momentum" in name]
    volatility_idx = [i for i, name in enumerate(factor_names) if "volatility" in name]

    y = (
        X[:, momentum_idx].mean(axis=1) * 0.3
        + X[:, volatility_idx].mean(axis=1) * -0.2
        + rng.standard_normal(n_samples) * 0.1
    )

    if not return_dataframe:
        return X, y, factor_names

    df = pd.DataFrame(X, columns=factor_names)
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=n_samples, freq="5T")
    df["synthetic_future_return"] = y

    return X, y, factor_names, df


__all__ = ["create_sample_data"]
