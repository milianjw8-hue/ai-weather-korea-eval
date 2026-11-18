"""Gridded verification of forecasts against an analysis (e.g. ERA5).

Scores are computed per variable and lead time over a region with cos(latitude) area weights:
  rmse, bias, mae, pattern_corr (centred, area-weighted spatial correlation of the fields)
Forecast and truth are first put on a common grid by bilinear interpolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .probabilistic import crps_ensemble, spread_skill
from .regions import Region

ENSEMBLE_DIM = "ensemble"  # dimension name used by Earth2Studio ensemble output


def standardize(ds: xr.Dataset) -> xr.Dataset:
    """Rename common coordinate spellings to lat/lon/time/lead_time and sort latitude ascending."""
    ren = {}
    for a, b in (("latitude", "lat"), ("longitude", "lon"), ("valid_time", "time")):
        if a in ds.dims or a in ds.coords:
            ren[a] = b
    ds = ds.rename(ren)
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    if "lead_time" in ds.coords and not np.issubdtype(ds["lead_time"].dtype, np.timedelta64):
        ds = ds.assign_coords(lead_time=pd.to_timedelta(ds["lead_time"].values, unit="h"))
    return ds


def common_grid(forecasts: dict[str, xr.Dataset], truth: xr.Dataset, region: Region, resolution: float | None = None):
    """Target lat/lon grid: the coarsest resolution among inputs (or `resolution`), inside region."""
    if resolution is None:
        res = [float(np.abs(np.diff(d["lat"].values)).min()) for d in [truth, *forecasts.values()]]
        resolution = max(res)
    lat = np.arange(region.lat_min, region.lat_max + 1e-6, resolution)
    lon = np.arange(region.lon_min, region.lon_max + 1e-6, resolution)
    return lat, lon


def to_grid(ds: xr.Dataset, lat, lon) -> xr.Dataset:
    lon_vals = ds["lon"].values
    target_lon = np.mod(lon, 360) if lon_vals.max() > 180 else lon
    out = ds.interp(lat=lat, lon=target_lon, method="linear")
    return out.assign_coords(lon=lon)


def _weights(lat):
    w = np.cos(np.radians(lat))
    return xr.DataArray(w / w.mean(), dims="lat", coords={"lat": lat})


def _scores(f: xr.DataArray, o: xr.DataArray, w: xr.DataArray) -> dict:
    ok = np.isfinite(f) & np.isfinite(o)
    f, o = f.where(ok), o.where(ok)
    ww = w.broadcast_like(f).where(ok)
    wsum = ww.sum()
    d = f - o
    bias = (ww * d).sum() / wsum
    rmse = np.sqrt((ww * d**2).sum() / wsum)
    mae = (ww * np.abs(d)).sum() / wsum
    fa = f - (ww * f).sum() / wsum
    oa = o - (ww * o).sum() / wsum
    pc = (ww * fa * oa).sum() / np.sqrt((ww * fa**2).sum() * (ww * oa**2).sum())
    return {"rmse": float(rmse), "bias": float(bias), "mae": float(mae), "pattern_corr": float(pc),
            "n_points": int(ok.sum())}


def _ensemble_scores(f: xr.DataArray, o: xr.DataArray, w: xr.DataArray) -> dict:
    """Area-weighted CRPS (plain and fair) and spread-skill for an ensemble field (ensemble, lat, lon)."""
    f = f.transpose(ENSEMBLE_DIM, ...)
    x = np.moveaxis(f.values, 0, -1)
    y = o.transpose(*f.dims[1:]).values
    wt = w.broadcast_like(o).transpose(*f.dims[1:]).values
    out = {"members": int(f.sizes[ENSEMBLE_DIM])}
    for key, fair in (("crps", False), ("crps_fair", True)):
        c = crps_ensemble(y, x, fair=fair)
        ok = np.isfinite(c)
        out[key] = float((wt[ok] * c[ok]).sum() / wt[ok].sum())
    ss = spread_skill(y, x, weights=wt)
    out.update(spread=ss["spread"], spread_skill_ratio=ss["spread_skill_ratio"])
    return out


def grid_scores(forecasts: dict[str, xr.Dataset], truth: xr.Dataset, variables: list[str], region: Region,
                resolution: float | None = None) -> pd.DataFrame:
    """Return a tidy table: model, init_time, lead_h, valid_time, variable, rmse, bias, mae, pattern_corr, crps.

    Forecasts with an `ensemble` dimension are scored on the ensemble mean for the deterministic
    scores and on all members for crps, crps_fair, spread and spread_skill_ratio. Deterministic
    forecasts get crps = mae so every model shares one proper score.
    """
    truth = standardize(truth)
    forecasts = {k: standardize(v) for k, v in forecasts.items()}
    lat, lon = common_grid(forecasts, truth, region, resolution)
    tg = to_grid(truth, lat, lon)
    w = _weights(lat)
    rows = []
    for name, fc in forecasts.items():
        fg = to_grid(fc, lat, lon)
        for t0 in fg["time"].values:
            for lt in fg["lead_time"].values:
                vt = t0 + lt
                if vt not in tg["time"].values:
                    continue
                for v in variables:
                    if v not in fg or v not in tg:
                        continue
                    f = fg[v].sel(time=t0, lead_time=lt)
                    o = tg[v].sel(time=vt)
                    if ENSEMBLE_DIM in f.dims:
                        s = _scores(f.mean(ENSEMBLE_DIM), o, w)
                        s.update(_ensemble_scores(f, o, w))
                    else:
                        s = _scores(f, o, w)
                        s.update(members=1, crps=s["mae"])  # CRPS of a point forecast = absolute error
                    rows.append({"model": name, "init_time": pd.Timestamp(t0), "lead_h": lt / np.timedelta64(1, "h"),
                                 "valid_time": pd.Timestamp(vt), "variable": v, **s})
    if not rows:
        raise ValueError("no overlapping valid times between forecasts and truth")
    return pd.DataFrame(rows)


def persistence(truth: xr.Dataset, init_times, lead_hours) -> xr.Dataset:
    """Persistence forecast built from the truth itself: field at init time held constant."""
    truth = standardize(truth)
    leads = pd.to_timedelta(np.asarray(lead_hours), unit="h")
    fields = []
    for t0 in pd.to_datetime(init_times):
        x = truth.sel(time=np.datetime64(t0)).drop_vars("time", errors="ignore")
        fields.append(xr.concat([x] * len(leads), dim=pd.Index(leads, name="lead_time")))
    return xr.concat(fields, dim=pd.Index(pd.to_datetime(init_times), name="time"))
