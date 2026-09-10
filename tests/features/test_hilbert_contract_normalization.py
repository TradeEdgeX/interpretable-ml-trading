import numpy as np
import pandas as pd

from src.features.time_series.utils_hilbert_features import extract_hilbert_features


def test_hilbert_replace_env_with_qnorm_produces_bounded_outputs():
    n = 400
    t = np.arange(n)
    price_fluc = np.sin(2 * np.pi * t / 40) + 0.2 * np.random.default_rng(1).normal(
        size=n
    )
    cvd_fluc = 0.7 * price_fluc + 0.2 * np.random.default_rng(2).normal(size=n)
    volume = 1000 + 200 * np.abs(np.random.default_rng(3).normal(size=n))
    close = 100 + np.cumsum(price_fluc * 0.1)

    df = pd.DataFrame(
        {
            "wpt_price_fluctuation": price_fluc,
            "wpt_cvd_fluctuation": cvd_fluc,
            "volume": volume,
            "close": close,
        }
    )

    out = extract_hilbert_features(
        df,
        price_fluctuation_col="wpt_price_fluctuation",
        cvd_fluctuation_col="wpt_cvd_fluctuation",
        volume_col="volume",
        use_quantile_normalize=True,
        quantile_window=120,
        use_volume_fusion=True,
        replace_env_with_qnorm=True,
    )

    # Rank-normalized envelopes should be within [0,1] (allow tiny numeric slack) once available
    for c in ["hilbert_price_env", "hilbert_cvd_env", "hilbert_volume_env"]:
        vals = pd.to_numeric(out[c], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        assert np.isfinite(vals.to_numpy()).all()
        assert vals.min() >= -1e-6
        assert vals.max() <= 1.0 + 1e-6

    # Slopes should be bounded due to diff of [0,1] + explicit clipping
    for c in ["hilbert_price_env_slope", "hilbert_cvd_env_slope"]:
        vals = pd.to_numeric(out[c], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        assert np.isfinite(vals.to_numpy()).all()
        assert vals.min() >= -1.0 - 1e-6
        assert vals.max() <= 1.0 + 1e-6

    # Divergence is binary-ish
    div = pd.to_numeric(out["hilbert_triple_divergence"], errors="coerce").fillna(0.0)
    assert ((div >= 0.0) & (div <= 1.0)).all()
