import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kwxeval.cli import main
from kwxeval.regions import KOREA, Region
from kwxeval.track import great_circle_km, track_errors, track_minimum
from kwxeval.verify import grid_scores, persistence

ROOT = Path(__file__).resolve().parents[1]
LAT = np.arange(45.0, 29.99, -0.5)
LON = np.arange(120.0, 135.01, 0.5)
TIMES = pd.date_range("2030-01-01", periods=5, freq="6h").values
LEADS = np.arange(0, 25, 6).astype("timedelta64[h]")


def truth_ds(value=1.0):
    data = np.full((len(TIMES), len(LAT), len(LON)), value)
    return xr.Dataset({"t2m": (("time", "lat", "lon"), data)}, coords={"time": TIMES, "lat": LAT, "lon": LON})


def forecast_ds(offset):
    data = np.full((1, len(LEADS), len(LAT), len(LON)), 1.0 + offset)
    return xr.Dataset({"t2m": (("time", "lead_time", "lat", "lon"), data)},
                      coords={"time": TIMES[:1], "lead_time": LEADS, "lat": LAT, "lon": LON})


def test_constant_offset_scores():
    s = grid_scores({"m": forecast_ds(2.0)}, truth_ds(), ["t2m"], KOREA)
    assert len(s) == len(LEADS)
    assert np.allclose(s["bias"], 2.0)
    assert np.allclose(s["rmse"], 2.0)


def test_valid_time_alignment_and_pattern_corr():
    rng = np.random.default_rng(0)
    tr = truth_ds()
    tr["t2m"].values[:] = rng.normal(size=tr["t2m"].shape)
    fc = forecast_ds(0.0)
    for k, lt in enumerate(LEADS):
        fc["t2m"].values[0, k] = tr["t2m"].sel(time=TIMES[0] + lt).values
    s = grid_scores({"perfect": fc}, tr, ["t2m"], KOREA)
    assert np.allclose(s["rmse"], 0, atol=1e-9)
    assert np.allclose(s["pattern_corr"], 1)


def test_persistence_shape():
    p = persistence(truth_ds(), [TIMES[0]], [0, 6, 12])
    assert p["t2m"].dims == ("time", "lead_time", "lat", "lon")
    assert p.sizes["lead_time"] == 3


def test_region_crop_descending_lat():
    r = Region("t", 33, 38, 124, 130)
    c = r.crop(truth_ds())
    assert float(c.lat.min()) >= 33 and float(c.lat.max()) <= 38


def test_tracker_follows_moving_low():
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    centres = [(32.0 + i, 125.0 + i) for i in range(len(TIMES))]
    msl = np.stack([101300 - 3000 * np.exp(-((la - a) ** 2 + (lo - b) ** 2) / 2) for a, b in centres])
    da = xr.DataArray(msl, dims=("time", "lat", "lon"), coords={"time": TIMES, "lat": LAT, "lon": LON})
    tr = track_minimum(da, (31.5, 124.5), search_km=300)
    assert np.allclose(tr["lat"], [c[0] for c in centres])
    assert np.allclose(tr["lon"], [c[1] for c in centres])
    err = track_errors(tr, tr)
    assert np.allclose(err["position_error_km"], 0)


def test_great_circle_one_degree_latitude():
    assert great_circle_km(33, 126, 34, 126) == pytest.approx(111.2, abs=0.2)


def test_cli_verify_synthetic(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_case.py"), "--out", str(tmp_path / "runs")],
                   check=True)
    main(["verify", "--case", "synthetic_typhoon", "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "res"),
          "--persistence"])
    out = tmp_path / "res" / "synthetic_typhoon"
    s = pd.read_csv(out / "grid_scores.csv")
    msl = s[(s.variable == "msl") & (s.lead_h == 96)].set_index("model")["rmse"]
    assert msl["model_a"] < msl["model_b"]  # model_a is built with the smaller error
    ea = pd.read_csv(out / "track_error_model_a_2030080100.csv")["position_error_km"].iloc[-1]
    eb = pd.read_csv(out / "track_error_model_b_2030080100.csv")["position_error_km"].iloc[-1]
    assert ea < eb
    assert (out / "tracks_2030080100.png").exists()
