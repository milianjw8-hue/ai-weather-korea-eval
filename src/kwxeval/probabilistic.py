"""Probabilistic (ensemble) verification scores.

Pure NumPy. The ensemble member axis is the LAST axis of `ens` unless `axis` is given.
Shared with ai-downscale-verify (dsverify.probabilistic); keep the two copies identical.

References
----------
Hersbach (2000) Wea. Forecasting 15, 559-570                 CRPS decomposition and ensemble form
Gneiting & Raftery (2007) JASA 102, 359-378                   CRPS as a proper score; Gaussian closed form
Ferro (2014) QJRMS 140, 1917-1923                             fair (ensemble-size-adjusted) CRPS
Fortin et al. (2014) J. Hydrometeor. 15, 1708-1713            spread correction (m+1)/m
Hamill (2001) Mon. Wea. Rev. 129, 550-560                     rank histograms
Candille & Talagrand (2005) QJRMS 131, 2131-2150              rank-histogram flatness (delta)
Brier (1950); Murphy (1973) J. Appl. Meteor. 12, 595-600      Brier score and reliability
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _move(ens, axis):
    return np.moveaxis(np.asarray(ens, dtype=float), axis, -1)


def crps_ensemble(obs, ens, axis: int = -1, fair: bool = False):
    """CRPS of an ensemble for each case (same shape as `obs`).

    CRPS = E|X - y| - 0.5 E|X - X'|, computed in O(m log m) from sorted members.
    fair=True uses the unbiased m(m-1) normalisation (Ferro 2014), which removes the
    advantage that larger ensembles get from sampling alone.
    """
    x = np.sort(_move(ens, axis), axis=-1)
    y = np.asarray(obs, dtype=float)[..., None]
    m = x.shape[-1]
    term1 = np.abs(x - y).mean(axis=-1)
    i = np.arange(1, m + 1)
    s = ((2 * i - m - 1) * x).sum(axis=-1)  # = 0.5 * sum_ij |x_i - x_j|
    denom = m * (m - 1) if fair else m * m
    if fair and m < 2:
        raise ValueError("fair CRPS needs at least 2 members")
    out = term1 - s / denom
    return np.where(np.isfinite(y[..., 0]) & np.isfinite(x).all(axis=-1), out, np.nan)


def crps_gaussian(obs, mu, sigma):
    """Closed-form CRPS for a normal predictive distribution N(mu, sigma^2)."""
    sigma = np.asarray(sigma, dtype=float)
    z = (np.asarray(obs, dtype=float) - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def skill_score(score, reference):
    """Generic skill score 1 - score/reference using means over all finite cases."""
    s, r = np.asarray(score, float), np.asarray(reference, float)
    ok = np.isfinite(s) & np.isfinite(r)
    return float(1 - s[ok].mean() / r[ok].mean()) if ok.any() and r[ok].mean() > 0 else np.nan


def spread_skill(obs, ens, axis: int = -1, weights=None) -> dict:
    """Ensemble spread vs RMSE of the ensemble mean.

    spread = sqrt(mean variance * (m+1)/m) so that a statistically consistent ensemble has
    spread/RMSE close to 1 for any ensemble size (Fortin et al. 2014). Ratio < 1: under-dispersive.
    """
    x = _move(ens, axis)
    y = np.asarray(obs, dtype=float)
    m = x.shape[-1]
    mean = x.mean(axis=-1)
    var = x.var(axis=-1, ddof=1)
    ok = np.isfinite(y) & np.isfinite(mean) & np.isfinite(var)
    w = np.ones_like(y) if weights is None else np.broadcast_to(np.asarray(weights, float), y.shape)
    w = np.where(ok, w, 0)
    if w.sum() == 0:
        return {"n": 0}
    mse = (w * np.where(ok, (mean - y) ** 2, 0)).sum() / w.sum()
    mvar = (w * np.where(ok, var, 0)).sum() / w.sum()
    spread = np.sqrt(mvar * (m + 1) / m)
    rmse = np.sqrt(mse)
    return {"n": int(ok.sum()), "members": m, "rmse_mean": float(rmse), "spread": float(spread),
            "spread_skill_ratio": float(spread / rmse) if rmse > 0 else np.nan}


def rank_histogram(obs, ens, axis: int = -1, seed: int = 0):
    """Counts of the observation's rank among members (m+1 bins). Ties are broken at random."""
    x = _move(ens, axis).reshape(-1, np.asarray(ens).shape[axis])
    y = np.asarray(obs, dtype=float).reshape(-1)
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[ok], y[ok]
    m = x.shape[1]
    below = (x < y[:, None]).sum(axis=1)
    ties = (x == y[:, None]).sum(axis=1)
    rng = np.random.default_rng(seed)
    rank = below + np.floor(rng.random(len(y)) * (ties + 1)).astype(int)
    return np.bincount(rank, minlength=m + 1)


def rank_flatness(counts) -> dict:
    """Candille & Talagrand delta: 1 for a flat histogram on average, >1 not flat.
    Also returns the fraction of observations outside the ensemble range."""
    c = np.asarray(counts, float)
    n, k = c.sum(), len(c)
    if n == 0:
        return {"delta": np.nan, "outside_fraction": np.nan}
    delta = ((c - n / k) ** 2).sum() / (n * (k - 1) / k)
    return {"delta": float(delta), "outside_fraction": float((c[0] + c[-1]) / n),
            "expected_outside": 2 / k}


def exceedance_probability(ens, threshold, axis: int = -1):
    """Fraction of members >= threshold."""
    return (_move(ens, axis) >= threshold).mean(axis=-1)


def brier(obs, prob, threshold, n_bins: int = 10) -> dict:
    """Brier score for the event obs >= threshold, its Murphy decomposition, and reliability-diagram data."""
    y = np.asarray(obs, float)
    p = np.asarray(prob, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0:
        return {"n": 0}
    o = (y >= threshold).astype(float)
    bs = float(((p - o) ** 2).mean())
    base = o.mean()
    unc = base * (1 - base)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rel = res = 0.0
    table = []
    for b in range(n_bins):
        sel = bins == b
        nk = int(sel.sum())
        if nk == 0:
            continue
        pk, ok_ = p[sel].mean(), o[sel].mean()
        rel += nk * (pk - ok_) ** 2
        res += nk * (ok_ - base) ** 2
        table.append({"bin": b, "n": nk, "forecast_prob": float(pk), "observed_freq": float(ok_)})
    rel /= y.size
    res /= y.size
    return {"n": int(y.size), "threshold": threshold, "base_rate": float(base), "brier": bs,
            "reliability": float(rel), "resolution": float(res), "uncertainty": float(unc),
            "bss_clim": float(1 - bs / unc) if unc > 0 else np.nan, "reliability_table": table}
