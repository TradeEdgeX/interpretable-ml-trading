from src.research.stat_kernels.z_test import two_proportion_z


def test_two_proportion_z_significant():
    z = two_proportion_z(0.6, 100, 0.4, 100)
    assert z > 2.0


def test_two_proportion_z_small_sample():
    assert two_proportion_z(0.9, 2, 0.1, 2) == 0.0
