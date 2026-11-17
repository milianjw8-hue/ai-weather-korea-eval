"""Simple tropical-cyclone tracker and track/intensity errors.

The tracker follows the mean-sea-level-pressure minimum: starting from a first guess,
at each step it takes the MSLP minimum within `search_km` of the previous position.
This is adequate for comparing models on a well-defined typhoon; it is not a full
TC-tracking algorithm (no vorticity or warm-core checks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

EARTH_RADIUS_KM = 6371.0


def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _min_near(field: xr.DataArray, lat0, lon0, search_km):
    la, lo = np.meshgrid(field["lat"].values, field["lon"].values, indexing="ij")
    dist = great_circle_km(lat0, lon0, la, lo)
    vals = np.where(dist <= search_km, field.values, np.inf)
    if not np.isfinite(vals).any():
        return np.nan, np.nan, np.nan
    i, j = np.unravel_index(np.argmin(vals), vals.shape)
    return float(la[i, j]), float(lo[i, j]), float(field.values[i, j])


def track_minimum(msl: xr.DataArray, first_guess: tuple[float, float], search_km: float = 400.0,
                  wind: xr.DataArray | None = None, wind_radius_km: float = 300.0) -> pd.DataFrame:
    """Track the MSLP minimum along the `time` dimension of a (time, lat, lon) field.

    Returns columns time, lat, lon, msl_min (same unit as input) and, if `wind` (speed) is given,
    wind_max within `wind_radius_km` of the centre.
    """
    lat0, lon0 = first_guess
    rows = []
    for t in msl["time"].values:
        lat0, lon0, pmin = _min_near(msl.sel(time=t), lat0, lon0, search_km)
        row = {"time": pd.Timestamp(t), "lat": lat0, "lon": lon0, "msl_min": pmin}
        if wind is not None and np.isfinite(lat0):
            w = wind.sel(time=t)
            la, lo = np.meshgrid(w["lat"].values, w["lon"].values, indexing="ij")
            near = great_circle_km(lat0, lon0, la, lo) <= wind_radius_km
            row["wind_max"] = float(np.nanmax(np.where(near, w.values, np.nan)))
        rows.append(row)
        if not np.isfinite(lat0):
            break
    return pd.DataFrame(rows)


def forecast_to_valid_time(ds: xr.Dataset, init_time) -> xr.Dataset:
    """Select one initial time and relabel lead_time as valid time."""
    x = ds.sel(time=np.datetime64(pd.Timestamp(init_time)))
    vt = x["time"].values + x["lead_time"].values
    return x.drop_vars("time").rename({"lead_time": "time"}).assign_coords(time=vt)


def track_errors(fc_track: pd.DataFrame, ref_track: pd.DataFrame) -> pd.DataFrame:
    """Direct position error (km) and central-pressure error at matching times."""
    m = fc_track.merge(ref_track, on="time", suffixes=("_fc", "_ref"))
    m["position_error_km"] = great_circle_km(m["lat_fc"], m["lon_fc"], m["lat_ref"], m["lon_ref"])
    m["msl_error"] = m["msl_min_fc"] - m["msl_min_ref"]
    return m
