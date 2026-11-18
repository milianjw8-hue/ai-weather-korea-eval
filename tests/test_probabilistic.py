import numpy as np
import pytest

from kwxeval.probabilistic import (
    brier,
    crps_ensemble,
    crps_gaussian,
    exceedance_probability,
    rank_flatness,
    rank_histogram,
    spread_skill,
)


def crps_bruteforce(y, x):
    x = np.asarray(x, float)
    return np.abs(x - y).mean() - 0.5 * np.abs(x[:, None] - x[None, :]).mean()


def test_crps_matches_definition():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y = rng.normal(size=7), rng.normal()
        assert crps_ensemble(y, x) == pytest.approx(crps_bruteforce(y, x))


def test_crps_single_member_is_absolute_error():
    assert crps_ensemble(3.0, [5.0]) == pytest.approx(2.0)


def test_fair_crps_unbiased_for_small_ensembles():
    # members and observation drawn from the same N(0,1): fair CRPS averages to the Gaussian value
    rng = np.random.default_rng(1)
    y = rng.normal(size=40000)
    ens = rng.normal(size=(40000, 5))
    target = crps_gaussian(y, 0.0, 1.0).mean()
    assert crps_ensemble(y, ens, fair=True).mean() == pytest.approx(target, rel=0.02)
    assert crps_ensemble(y, ens).mean() > target * 1.05  # plain CRPS penalises small m


def test_crps_axis_and_nan():
    ens = np.array([[1.0, 2.0, 3.0], [np.nan, 1.0, 2.0]]).T  # members on axis 0
    out = crps_ensemble([2.0, 1.0], ens, axis=0)
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_spread_skill_consistent_ensemble():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=20000)
    ens = rng.normal(size=(20000, 10))
    s = spread_skill(truth, ens)
    assert s["spread_skill_ratio"] == pytest.approx(1.0, abs=0.03)
    s2 = spread_skill(truth, ens * 0.5)
    assert s2["spread_skill_ratio"] < 0.7


def test_rank_histogram_flat_vs_underdispersive():
    rng = np.random.default_rng(3)
    y = rng.normal(size=30000)
    flat = rank_histogram(y, rng.normal(size=(30000, 9)))
    narrow = rank_histogram(y, 0.3 * rng.normal(size=(30000, 9)))
    assert flat.sum() == 30000 and len(flat) == 10
    assert rank_flatness(flat)["delta"] < 2
    f = rank_flatness(narrow)
    assert f["delta"] > 50 and f["outside_fraction"] > 0.5


def test_brier_perfect_and_climatology():
    obs = np.array([0, 20, 0, 20, 0, 0, 0, 20.0])
    perfect = brier(obs, (obs >= 14).astype(float), 14)
    assert perfect["brier"] == 0 and perfect["bss_clim"] == 1
    clim = brier(obs, np.full(obs.size, 3 / 8), 14)
    assert clim["bss_clim"] == pytest.approx(0, abs=1e-12)
    assert clim["brier"] == pytest.approx(clim["reliability"] - clim["resolution"] + clim["uncertainty"])


def test_exceedance_probability():
    assert exceedance_probability([[10, 15, 20, 5]], 14)[0] == 0.5
