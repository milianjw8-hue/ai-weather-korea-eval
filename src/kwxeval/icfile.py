"""Turn an external analysis (e.g. KIM) into an Earth2Studio initial-condition file.

Earth2Studio's ``DataArrayFile`` source reads one DataArray with dims (time, variable, lat, lon) on the
global 0.25 deg grid (lat 90 -> -90, lon 0 -> 359.75) using Earth2Studio variable names
(msl, u10m, t2m, z500, ...). Many models also need the state 6 h before the initial time,
so the file must hold both times.

The variable list a model needs is only known with Earth2Studio installed: use
:func:`model_input_variables` on the GPU machine, then :func:`check_ic_file`.

Units must already match Earth2Studio (Pa for msl/sp, m2 s-2 for geopotential z*, K, m s-1).
This module does not convert units; :func:`check_ic_file` flags typical mistakes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LAT = np.linspace(90.0, -90.0, 721)
LON = np.arange(0.0, 360.0, 0.25)

# rough physical ranges in Earth2Studio units, to catch unit errors (hPa vs Pa, m vs m2 s-2)
PLAUSIBLE = {
    "msl": (85_000.0, 110_000.0),
    "sp": (45_000.0, 110_000.0),
    "t2m": (180.0, 340.0),
    "u10m": (-80.0, 80.0),
    "v10m": (-80.0, 80.0),
    "z500": (44_000.0, 60_000.0),
    "z850": (8_000.0, 17_000.0),
    "tcwv": (0.0, 100.0),
}


def required_times(init_times, history_hours=(6,)) -> list[pd.Timestamp]:
    """Initial times plus the earlier states most models need (default: -6 h)."""
    out = set()
    for t in pd.to_datetime(list(init_times)):
        out.add(t)
        for h in history_hours:
            out.add(t - pd.Timedelta(hours=h))
    return sorted(out)


def _standard_lon(ds: xr.Dataset, lon: str) -> xr.Dataset:
    ds = ds.assign_coords({lon: ds[lon] % 360}).sortby(lon)
    # wrap one column on each side so interpolation works across 0/360
    first = ds.isel({lon: 0}).assign_coords({lon: ds[lon].values[0] + 360})
    last = ds.isel({lon: -1}).assign_coords({lon: ds[lon].values[-1] - 360})
    return xr.concat([last, ds, first], dim=lon)


def prepare_ic_file(ds: xr.Dataset, out_path: str | Path, rename: dict[str, str] | None = None,
                    lat: str = "lat", lon: str = "lon", time: str = "time",
                    variables: list[str] | None = None) -> Path:
    """Regrid a (time, lat, lon) dataset to the Earth2Studio 0.25 deg grid and save one DataArray.

    rename maps source variable names to Earth2Studio names, e.g. {"PSL": "msl", "U10": "u10m"}.
    The source grid must be global (a regional grid would leave NaN that models cannot use).
    """
    ds = ds.rename(rename or {})
    ds = ds.rename({k: v for k, v in {lat: "lat", lon: "lon", time: "time"}.items() if k != v})
    variables = variables or [v for v in ds.data_vars if {"lat", "lon"} <= set(ds[v].dims)]
    if not variables:
        raise ValueError("no (lat, lon) variables found")
    lat_span = float(ds.lat.max() - ds.lat.min())
    lon_span = float((ds.lon % 360).max() - (ds.lon % 360).min())
    if lat_span < 170 or lon_span < 350:
        raise ValueError(f"source grid is not global (lat span {lat_span:.1f}, lon span {lon_span:.1f} deg)")
    ds = _standard_lon(ds[variables], "lon").sortby("lat")
    # poles: copy the outermost source rows to +-90 so linear interpolation leaves no NaN
    parts = [ds]
    if float(ds.lat[0]) > -90.0:
        parts.insert(0, ds.isel(lat=[0]).assign_coords(lat=[-90.0]))
    if float(ds.lat[-1]) < 90.0:
        parts.append(ds.isel(lat=[-1]).assign_coords(lat=[90.0]))
    ds = xr.concat(parts, dim="lat") if len(parts) > 1 else ds
    reg = ds.interp(lat=np.sort(LAT), lon=LON, method="linear").sortby("lat", ascending=False)
    da = reg.to_array("variable").transpose("time", "variable", "lat", "lon").astype("float32")
    da.name = "fields"
    da.attrs["note"] = "Earth2Studio DataArrayFile initial conditions, made by kwxeval.icfile"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(out_path)
    return out_path


def check_ic_file(path: str | Path, init_times=None, variables: list[str] | None = None) -> list[str]:
    """Return a list of problems (empty if the file looks usable)."""
    problems = []
    da = xr.open_dataarray(path)
    if tuple(da.dims) != ("time", "variable", "lat", "lon"):
        problems.append(f"dims are {da.dims}, expected (time, variable, lat, lon)")
    if da.sizes.get("lat") != 721 or da.sizes.get("lon") != 1440:
        problems.append(f"grid is {da.sizes.get('lat')}x{da.sizes.get('lon')}, expected 721x1440")
    elif float(da.lat[0]) != 90.0 or float(da.lon[0]) != 0.0:
        problems.append("lat must run 90 -> -90 and lon 0 -> 359.75")
    have = [str(v) for v in da["variable"].values]
    if variables:
        missing = sorted(set(variables) - set(have))
        if missing:
            problems.append(f"missing variables: {', '.join(missing)}")
    if init_times is not None:
        times = set(pd.to_datetime(da.time.values))
        missing_t = [str(t) for t in required_times(init_times) if t not in times]
        if missing_t:
            problems.append(f"missing times (incl. -6 h states): {', '.join(missing_t)}")
    for v in have:
        x = da.sel(variable=v).values
        if np.isnan(x).any():
            problems.append(f"{v}: contains NaN")
            continue
        if v in PLAUSIBLE:
            lo, hi = PLAUSIBLE[v]
            med = float(np.median(x))
            if not lo <= med <= hi:
                problems.append(f"{v}: median {med:.4g} outside {lo:g}..{hi:g} (unit error?)")
    da.close()
    return problems


def model_input_variables(model_name: str) -> list[str]:
    """Variables a model needs as input (needs Earth2Studio; downloads the checkpoint on first use)."""
    from earth2studio.models import px

    from .run import MODELS

    cls = getattr(px, MODELS[model_name].cls)
    model = cls.load_model(cls.load_default_package())
    return [str(v) for v in model.input_coords()["variable"]]
