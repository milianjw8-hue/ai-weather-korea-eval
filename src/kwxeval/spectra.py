"""Radially averaged 2-D power spectra on a regional lat-lon grid.

AI models trained with mean-squared-error losses tend to smooth small scales as lead time grows.
Comparing their spectra with the analysis shows how much small-scale variance is lost.
The regional grid is treated as locally Cartesian (dx = R cos(lat0) dlon), the field mean is
removed and a 2-D Hann window is applied before the FFT. Use for relative comparison between
models on the same grid, not for absolute spectral slopes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

EARTH_RADIUS_KM = 6371.0


def power_spectrum(field: np.ndarray, dx_km: float, dy_km: float, n_bins: int | None = None):
    """Return (wavelength_km, power) of a 2-D field (lat, lon)."""
    a = np.asarray(field, dtype=float)
    a = np.where(np.isfinite(a), a, np.nanmean(a))
    a = a - a.mean()
    ny, nx = a.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    p2 = np.abs(np.fft.fft2(a * win)) ** 2 / (win**2).sum()
    ky = np.fft.fftfreq(ny, d=dy_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    k = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    kmax = min(np.abs(kx).max(), np.abs(ky).max())
    n_bins = n_bins or max(4, min(nx, ny) // 2)
    edges = np.linspace(0, kmax, n_bins + 1)
    idx = np.digitize(k.ravel(), edges) - 1
    ok = (idx >= 1) & (idx < n_bins)  # skip the mean (k=0 bin)
    power = np.bincount(idx[ok], weights=p2.ravel()[ok], minlength=n_bins) / np.maximum(
        np.bincount(idx[ok], minlength=n_bins), 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    keep = (np.arange(n_bins) >= 1) & (power > 0)
    return 1.0 / centres[keep], power[keep]


def grid_spacing_km(lat, lon):
    lat0 = np.deg2rad(np.mean(lat))
    dy = EARTH_RADIUS_KM * np.deg2rad(abs(float(np.diff(lat).mean())))
    dx = EARTH_RADIUS_KM * np.cos(lat0) * np.deg2rad(abs(float(np.diff(lon).mean())))
    return dx, dy


def spectra_table(fields: dict[str, xr.DataArray]) -> pd.DataFrame:
    """Spectra of several (lat, lon) fields on the same grid -> tidy table name, wavelength_km, power."""
    rows = []
    for name, da in fields.items():
        dx, dy = grid_spacing_km(da["lat"].values, da["lon"].values)
        wl, pw = power_spectrum(da.transpose("lat", "lon").values, dx, dy)
        rows.append(pd.DataFrame({"name": name, "wavelength_km": wl, "power": pw}))
    return pd.concat(rows, ignore_index=True)
