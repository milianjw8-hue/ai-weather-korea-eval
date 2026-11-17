"""Verification regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Region:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def crop(self, ds: xr.Dataset | xr.DataArray, lat: str = "lat", lon: str = "lon"):
        """Crop a lat/lon dataset (any latitude order, 0-360 or -180-180 longitudes)."""
        lons = ds[lon].values
        lo_min, lo_max = self.lon_min, self.lon_max
        if lons.max() > 180 and lo_min < 0:
            lo_min, lo_max = lo_min % 360, lo_max % 360
        lat_vals = ds[lat].values
        lat_sel = slice(self.lat_max, self.lat_min) if lat_vals[0] > lat_vals[-1] else slice(self.lat_min, self.lat_max)
        return ds.sel({lat: lat_sel, lon: slice(lo_min, lo_max)})

    def contains(self, lat, lon) -> np.ndarray:
        lat, lon = np.asarray(lat), np.asarray(lon)
        return (lat >= self.lat_min) & (lat <= self.lat_max) & (lon >= self.lon_min) & (lon <= self.lon_max)


KOREA = Region("korea", 30.0, 45.0, 120.0, 135.0)  # Korean Peninsula and surrounding seas
KOREA_WIDE = Region("korea_wide", 20.0, 50.0, 110.0, 150.0)  # for typhoon tracks approaching Korea
REGIONS = {r.name: r for r in (KOREA, KOREA_WIDE)}
