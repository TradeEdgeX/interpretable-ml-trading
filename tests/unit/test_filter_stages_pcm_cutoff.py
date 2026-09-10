from scripts.pipeline_steps.filter_stages import (
    _pcm_ef_cutoff,
    _pcm_ef_val_segment_end,
)


def test_pcm_ef_cutoff_prefers_explicit():
    assert (
        _pcm_ef_cutoff(
            pcm_cutoff_date="2024-05-31",
            test_start="2024-04-01",
            holdout_start="2024-01-01",
        )
        == "2024-05-31"
    )


def test_pcm_ef_cutoff_fallback_to_test_start():
    assert (
        _pcm_ef_cutoff(
            pcm_cutoff_date=None,
            test_start="2024-04-01",
            holdout_start="2024-01-01",
        )
        == "2024-04-01"
    )


def test_pcm_ef_val_segment_end():
    assert (
        _pcm_ef_val_segment_end(
            pcm_cutoff_date="2024-05-31",
            test_start="2024-04-01",
            holdout_start="2024-01-01",
            end_date="2026-03-31",
        )
        == "2024-05-31"
    )
    assert (
        _pcm_ef_val_segment_end(
            pcm_cutoff_date=None,
            test_start="2024-04-01",
            holdout_start="2024-01-01",
            end_date="2026-03-31",
        )
        == "2024-04-01"
    )
