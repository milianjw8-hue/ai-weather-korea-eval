"""Create a SYNTHETIC typhoon case to test `kwxeval verify` without a GPU or internet.

Writes runs/<case>/era5.nc (truth), truth_ifs.nc (a second analysis), and fake "model" forecasts:
  model_a  small position offset growing with lead time, slight weakening
  model_b  larger offset and stronger weakening (smoother field)
  model_a@gfs  model_a from another initial condition (more offset)
Nothing here is real data.

Usage: python scripts/make_synthetic_case.py --case synthetic_typhoon --out runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LAT = np.arange(50.0, 19.99, -0.25)  # descending, like ERA5
LON = np.arange(110.0, 150.01, 0.25)


def vortex(lat0, lon0, dp, r0=2.0):
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    r2 = (la - lat0) ** 2 + ((lo - lon0) * np.cos(np.radians(lat0))) ** 2
    msl = 101300.0 - dp * np.exp(-r2 / (2 * r0**2))
    # cyclonic geostrophic-like wind proportional to pressure gradient (shape only)
    dpdy, dpdx = np.gradient(msl, LAT, LON)
    u, v = 2e-3 * dpdy, -2e-3 * dpdx / np.cos(np.radians(la))
    t2m = 300.0 - 0.6 * (la - 20)
    z500 = 57500.0 - 60.0 * (la - 20) - dp * 0.3 * np.exp(-r2 / (2 * (2 * r0) ** 2))
    return msl, u, v, t2m, z500


def centre(t_hours):
    return 25.0 + 0.18 * t_hours / 6 * 1.5, 126.0 + 0.10 * t_hours / 6 * 1.5


def fields(times, offset_km_per_day=0.0, weaken=0.0):
    out = {k: [] for k in ("msl", "u10m", "v10m", "t2m", "z500")}
    t0 = times[0]
    for t in times:
        h = (t - t0) / np.timedelta64(1, "h")
        lat0, lon0 = centre(h)
        lon0 += offset_km_per_day * h / 24 / 100.0
        msl, u, v, t2m, z500 = vortex(lat0, lon0, 4500.0 * (1 - weaken * h / 96))
        for k, x in zip(out, (msl, u, v, t2m, z500)):
            out[k].append(x.astype("float32"))
    return {k: np.stack(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="synthetic_typhoon")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    d = Path(args.out) / args.case
    d.mkdir(parents=True, exist_ok=True)

    init = np.datetime64("2030-08-01T00:00")
    leads = np.arange(0, 97, 6).astype("timedelta64[h]")
    vt = init + leads
    tr = fields(vt)
    xr.Dataset({k: (("time", "lat", "lon"), v) for k, v in tr.items()},
               coords={"time": vt, "lat": LAT, "lon": LON},
               attrs={"title": "SYNTHETIC truth (not real data)"}).to_netcdf(d / "era5.nc")

    # second verifying analysis ("IFS"): truth plus small analysis noise, to show reference sensitivity
    rng0 = np.random.default_rng(7)
    noise = {"msl": 60.0, "u10m": 0.8, "v10m": 0.8, "t2m": 0.5, "z500": 15.0}
    xr.Dataset({k: (("time", "lat", "lon"), v + rng0.normal(0, noise[k], v.shape).astype("float32"))
                for k, v in tr.items()},
               coords={"time": vt, "lat": LAT, "lon": LON},
               attrs={"title": "SYNTHETIC second analysis (not real data)", "reference": "ifs"}
               ).to_netcdf(d / "truth_ifs.nc")

    # model_a@gfs: the same model started from a different (worse) initial state
    for name, off, wk in (("model_a", 60.0, 0.1), ("model_b", 180.0, 0.35), ("model_a@gfs", 110.0, 0.15)):
        f = fields(vt, off, wk)
        ds = xr.Dataset({k: (("time", "lead_time", "lat", "lon"), v[None]) for k, v in f.items()},
                        coords={"time": [init], "lead_time": leads.astype(int), "lat": LAT, "lon": LON},
                        attrs={"title": f"SYNTHETIC forecast {name} (not real data)"})
        # integer hours: readable by every xarray version (standardize() converts to timedelta)
        ds["lead_time"].attrs["long_name"] = "lead time in hours"
        ds.to_netcdf(d / f"{name}.nc")
    # 6-member ensemble: each member has its own track offset, so spread grows with lead time
    rng = np.random.default_rng(1)
    members = [fields(vt, float(rng.normal(60.0, 90.0)), 0.1) for _ in range(6)]
    ens = xr.Dataset({k: (("ensemble", "time", "lead_time", "lat", "lon"),
                          np.stack([m[k] for m in members])[:, None]) for k in members[0]},
                     coords={"ensemble": np.arange(6), "time": [init], "lead_time": leads.astype(int),
                             "lat": LAT, "lon": LON},
                     attrs={"title": "SYNTHETIC ensemble forecast model_ens (not real data)"})
    ens.to_netcdf(d / "model_ens.nc")
    # IBTrACS-style best track of the synthetic storm (exact centres, header + units row like v04r01)
    rows = ["SID,SEASON,NUMBER,BASIN,SUBBASIN,NAME,ISO_TIME,NATURE,LAT,LON,WMO_WIND,WMO_PRES,WMO_AGENCY",
            " ,Year, , , , , , ,degrees_north,degrees_east,kts,mb, "]
    for h in range(0, 97, 6):
        la, lo = centre(h)
        t = (pd.Timestamp(init) + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(f"2030213N25126,2030,1,WP,MM,SYNTH,{t},TS,{la:.4f},{lo:.4f},80,955,tokyo")
    (d / "ibtracs_synthetic.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"synthetic case written to {d}")


if __name__ == "__main__":
    main()
