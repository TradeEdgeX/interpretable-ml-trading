from src.time_series_model.diagnostics.ood_config import (
    compute_size_cap_multiplier,
    load_ood_config,
)


def test_load_ood_config_has_expected_defaults() -> None:
    # Uses in-code default when config/ood was removed; dashboard keys for snapshot contract.
    cfg = load_ood_config("config/ood/ood_config.yaml")
    assert cfg.version == 1
    assert cfg.ood_horizon_bars > 0
    assert cfg.survival_horizon_bars > 0
    assert "ood_score" in cfg.dashboard_keys


def test_size_cap_power_formula_monotone() -> None:
    cfg = load_ood_config("config/ood/ood_config.yaml")

    cap_good = compute_size_cap_multiplier(cfg=cfg, ood_score=0.1, survival_prob=0.9)
    cap_bad_ood = compute_size_cap_multiplier(cfg=cfg, ood_score=0.9, survival_prob=0.9)
    cap_bad_surv = compute_size_cap_multiplier(
        cfg=cfg, ood_score=0.1, survival_prob=0.1
    )

    assert 0.0 <= cap_good <= 1.0
    assert 0.0 <= cap_bad_ood <= 1.0
    assert 0.0 <= cap_bad_surv <= 1.0
    assert cap_good > cap_bad_ood
    assert cap_good > cap_bad_surv
