"""Best-track readers.

IBTrACS v04r01 CSV (NOAA NCEI): one row per storm fix; the second line of the file holds units.
Position and intensity are taken from one source agency:

  "WMO"   LAT/LON with WMO_WIND/WMO_PRES (responsible agency; RSMC Tokyo in the western North Pacific)
  "TOKYO" TOKYO_LAT/LON/WIND/PRES (10-min mean wind)
  "KMA"   KMA_LAT/LON/WIND/PRES   (KMA best track; IBTrACS v04r01 lists it for 2015-2022)
  "CMA", "HKO", "USA"             likewise (USA winds are 1-min means)

Wind is in knots and pressure in mb in IBTrACS; outputs are m/s and Pa (msl_min) plus hPa.
Wind averaging periods differ between agencies and are NOT adjusted here.
Data: https://www.ncei.noaa.gov/products/international-best-track-archive
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KNOT = 0.514444  # m/s
AGENCIES = ("WMO", "TOKYO", "KMA", "CMA", "HKO", "USA")


def _cols(agency: str):
    a = agency.upper()
    if a == "WMO":
        return "LAT", "LON", "WMO_WIND", "WMO_PRES"
    return f"{a}_LAT", f"{a}_LON", f"{a}_WIND", f"{a}_PRES"


def read_ibtracs(path, sid: str | None = None, name: str | None = None, season: int | None = None,
                 agency: str = "WMO", synoptic_only: bool = True) -> pd.DataFrame:
    """Return one storm's track: time (UTC), lat, lon, msl_min (Pa), msl_hpa, wind_max (m/s), wind_kt.

    Select by `sid` (e.g. '2023210N14148') or by `name` + `season`. With synoptic_only=True
    only 00/06/12/18 UTC fixes are kept (the times AI models are verified at).
    """
    if agency.upper() not in AGENCIES:
        raise ValueError(f"agency must be one of {AGENCIES}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    iso = pd.to_datetime(df["ISO_TIME"].str.strip(), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.assign(_t=iso)[iso.notna()]  # drops the units row
    if sid:
        df = df[df["SID"] == sid]
    else:
        if not (name and season):
            raise ValueError("give sid, or name and season")
        df = df[(df["NAME"].str.upper() == name.upper()) & (df["SEASON"] == str(season))]
    if df.empty:
        raise ValueError("storm not found in the best-track file")
    if df["SID"].nunique() > 1:
        raise ValueError(f"several storms match: {sorted(df['SID'].unique())}; pass sid=")
    la, lo, wi, pr = _cols(agency)
    num = {c: pd.to_numeric(df[c].str.strip().replace("", np.nan), errors="coerce") if c in df else np.nan
           for c in (la, lo, wi, pr)}
    out = pd.DataFrame({
        "time": df["_t"].values,
        "lat": num[la], "lon": num[lo], "wind_kt": num[wi], "msl_hpa": num[pr],
    }).reset_index(drop=True)
    out["wind_max"] = out["wind_kt"] * KNOT
    out["msl_min"] = out["msl_hpa"] * 100.0
    out = out.dropna(subset=["lat", "lon"])
    if synoptic_only:
        out = out[(out["time"].dt.hour % 6 == 0) & (out["time"].dt.minute == 0)]
    out.attrs.update(sid=df["SID"].iloc[0], name=df["NAME"].iloc[0], agency=agency.upper())
    return out.reset_index(drop=True)


def read_track_csv(path) -> pd.DataFrame:
    """Simple track CSV with columns time, lat, lon and optionally msl_hpa, wind_ms (e.g. a KMA best track)."""
    df = pd.read_csv(path)
    out = pd.DataFrame({"time": pd.to_datetime(df["time"]), "lat": df["lat"], "lon": df["lon"]})
    if "msl_hpa" in df:
        out["msl_hpa"] = df["msl_hpa"]
        out["msl_min"] = df["msl_hpa"] * 100.0
    if "wind_ms" in df:
        out["wind_max"] = df["wind_ms"]
    return out


def load_best_track(cfg: dict, root: Path | None = None) -> pd.DataFrame:
    """Load the `best_track` block of a case: {path, format: ibtracs|csv, sid | name+season, agency}."""
    path = Path(cfg["path"])
    if root is not None and not path.is_absolute():
        path = root / path
    if cfg.get("format", "ibtracs") == "csv":
        return read_track_csv(path)
    return read_ibtracs(path, sid=cfg.get("sid"), name=cfg.get("name"), season=cfg.get("season"),
                        agency=cfg.get("agency", "WMO"))
