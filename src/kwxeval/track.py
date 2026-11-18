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


def reference_motion(ref_track: pd.DataFrame) -> pd.DataFrame:
    """Unit vector (east, north) of the reference storm motion at each fix, from centred differences
    (one-sided at the ends), and translation speed in km/h."""
    r = ref_track.sort_values("time").reset_index(drop=True)
    lat, lon = r["lat"].to_numpy(float), r["lon"].to_numpy(float)
    t = (r["time"] - r["time"].iloc[0]) / np.timedelta64(1, "h")
    n = len(r)
    ex, ny_, spd = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        a, b = max(i - 1, 0), min(i + 1, n - 1)
        if a == b:
            continue
        dx = np.deg2rad(lon[b] - lon[a]) * EARTH_RADIUS_KM * np.cos(np.deg2rad(0.5 * (lat[a] + lat[b])))
        dy = np.deg2rad(lat[b] - lat[a]) * EARTH_RADIUS_KM
        d = np.hypot(dx, dy)
        if d > 0:
            ex[i], ny_[i] = dx / d, dy / d
            spd[i] = d / (t.iloc[b] - t.iloc[a])
    return pd.DataFrame({"time": r["time"], "motion_e": ex, "motion_n": ny_, "speed_kmh": spd})


def track_errors(fc_track: pd.DataFrame, ref_track: pd.DataFrame) -> pd.DataFrame:
    """Position, along-track, cross-track and intensity errors at matching times.

    along_track_km > 0: forecast centre AHEAD of the reference (too fast);
    cross_track_km > 0: forecast centre to the RIGHT of the reference motion.
    msl_error is in the input unit (Pa for ERA5/model fields and best tracks read by kwxeval);
    msl_error_hpa is the same in hPa. wind_error compares maximum wind (m/s) when both exist.
    speed_error_kmh: forecast minus reference translation speed (> 0: forecast storm moves too fast).
    """
    ref = ref_track.merge(reference_motion(ref_track), on="time")
    fc = fc_track.dropna(subset=["lat", "lon"])
    fc_speed = reference_motion(fc)[["time", "speed_kmh"]].rename(columns={"speed_kmh": "speed_kmh_fc"})
    fc = fc.merge(fc_speed, on="time", how="left")
    m = fc.merge(ref, on="time", suffixes=("_fc", "_ref"))
    m["speed_error_kmh"] = m["speed_kmh_fc"] - m["speed_kmh"]
    m["position_error_km"] = great_circle_km(m["lat_fc"], m["lon_fc"], m["lat_ref"], m["lon_ref"])
    # local tangent plane at the reference fix
    dx = np.deg2rad(m["lon_fc"] - m["lon_ref"]) * EARTH_RADIUS_KM * np.cos(np.deg2rad(m["lat_ref"]))
    dy = np.deg2rad(m["lat_fc"] - m["lat_ref"]) * EARTH_RADIUS_KM
    m["along_track_km"] = dx * m["motion_e"] + dy * m["motion_n"]
    m["cross_track_km"] = dx * m["motion_n"] - dy * m["motion_e"]
    if "msl_min_fc" in m and "msl_min_ref" in m:
        m["msl_error"] = m["msl_min_fc"] - m["msl_min_ref"]
        m["msl_error_hpa"] = m["msl_error"] / 100.0
    if "wind_max_fc" in m and "wind_max_ref" in m:
        m["wind_error"] = m["wind_max_fc"] - m["wind_max_ref"]
    return m


def summarize_track_errors(err_tables: dict[str, pd.DataFrame], init_time) -> pd.DataFrame:
    """One row per model and lead time with the error components."""
    t0 = pd.Timestamp(init_time)
    rows = []
    cols = ["position_error_km", "along_track_km", "cross_track_km", "speed_error_kmh", "msl_error_hpa",
            "wind_error"]
    for name, e in err_tables.items():
        x = e.copy()
        x["lead_h"] = (pd.to_datetime(x["time"]) - t0) / np.timedelta64(1, "h")
        x = x[[c for c in ["lead_h", *cols] if c in x]]
        x.insert(0, "model", name)
        x.insert(1, "init_time", t0)
        rows.append(x)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


LEAD_BINS = (0, 24, 48, 72, 120, 240)


def track_bias(summary: pd.DataFrame, lead_bins=LEAD_BINS) -> pd.DataFrame:
    """Signed (systematic) track and intensity errors per model and lead-time window.

    along_track_km      mean > 0: storms run ahead (too fast); < 0: too slow
    frac_ahead          share of fixes where the forecast is ahead of the reference
    speed_error_kmh     mean forecast-minus-reference translation speed
    cross_track_km      mean > 0: forecast to the right of the reference motion
    msl_error_hpa       mean > 0: central pressure too high (storm too weak)
    wind_error          mean < 0: maximum wind too weak
    Lead 0 is excluded (it measures the analysis, not the forecast).
    """
    x = summary[summary["lead_h"] > 0].copy()
    if x.empty:
        return pd.DataFrame()
    x["lead_window"] = pd.cut(x["lead_h"], bins=list(lead_bins), right=True,
                              labels=[f"{a + 1 if a else 1}-{b}h" for a, b in zip(lead_bins[:-1], lead_bins[1:])])
    x["ahead"] = (x["along_track_km"] > 0).astype(float)
    agg = {"n": ("lead_h", "size"), "along_track_km": ("along_track_km", "mean"), "frac_ahead": ("ahead", "mean"),
           "cross_track_km": ("cross_track_km", "mean")}
    for c in ("speed_error_kmh", "msl_error_hpa", "wind_error"):
        if c in x:
            agg[c] = (c, "mean")
    out = x.groupby(["model", "lead_window"], observed=True).agg(**agg).reset_index()
    return out.round(2)


def ensemble_mean_track(member_tracks: pd.DataFrame) -> pd.DataFrame:
    """Mean position, central pressure and wind of member tracks at each time, plus the track spread
    (mean great-circle distance of members from the mean position, km)."""
    g = member_tracks.dropna(subset=["lat", "lon"]).groupby("time")
    agg = {"lat": "mean", "lon": "mean", "msl_min": "mean"}
    if "wind_max" in member_tracks:
        agg["wind_max"] = "mean"
    mean = g.agg(agg).reset_index()
    mean["n_members"] = g.size().values
    sp = member_tracks.merge(mean[["time", "lat", "lon"]], on="time", suffixes=("", "_mean"))
    sp["d"] = great_circle_km(sp["lat"], sp["lon"], sp["lat_mean"], sp["lon_mean"])
    mean["track_spread_km"] = sp.groupby("time")["d"].mean().reindex(mean["time"]).values
    return mean
