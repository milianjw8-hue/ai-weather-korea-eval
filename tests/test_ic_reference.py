"""Initial-condition options, IC files, translation-speed bias and reference sensitivity (no GPU, no internet)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kwxeval.cli import main
from kwxeval.icfile import check_ic_file, prepare_ic_file, required_times
from kwxeval.run import MODELS, check_ic_times, forecast_label, model_table, split_label
from kwxeval.track import summarize_track_errors, track_bias, track_errors

ROOT = Path(__file__).resolve().parents[1]


def test_labels_roundtrip():
    assert forecast_label("pangu6") == "pangu6"
    assert forecast_label("pangu6", "gfs") == "pangu6@gfs"
    assert split_label("pangu6@gfs") == ("pangu6", "gfs")
    assert split_label("pangu6") == ("pangu6", "era5")


def test_model_table_has_new_models():
    t = model_table()
    for name in ("aifs_ens", "fcn3", "aurora", "graphcast_op"):
        assert name in set(t["name"])
    assert MODELS["fcn3"].gpu_gb == 80 and MODELS["fcn3"].kind == "ens"


def test_ic_time_limits():
    check_ic_times("era5", ["2023-08-08T00:00"])
    check_ic_times("gfs", ["2023-08-08T00:00"])
    with pytest.raises(ValueError, match="2024-03-01"):
        check_ic_times("ifs", ["2023-08-08T00:00"])
    with pytest.raises(ValueError):  # -6 h state falls before the archive start
        check_ic_times("ifs", ["2024-03-01T00:00"])
    check_ic_times("ifs", ["2025-04-11T12:00"])
    with pytest.raises(ValueError):
        check_ic_times("kim", ["2025-04-11T12:00"])


def _fake_global(hpa=False):
    lat = np.arange(-89.5, 89.6, 1.0)  # like many model grids: no pole rows
    lon = np.arange(-180, 180, 1.0)
    times = pd.to_datetime(["2025-04-11T06:00", "2025-04-11T12:00"])
    shape = (len(times), lat.size, lon.size)
    msl = np.full(shape, 101_300.0) - 500 * np.cos(np.deg2rad(lat))[None, :, None]
    return xr.Dataset({"PSL": (("time", "lat", "lon"), msl / (100 if hpa else 1)),
                       "U10": (("time", "lat", "lon"), np.full(shape, 3.0)),
                       "T2": (("time", "lat", "lon"), np.full(shape, 285.0))},
                      coords={"time": times, "lat": lat, "lon": lon})


def test_prepare_and_check_ic_file(tmp_path):
    ren = {"PSL": "msl", "U10": "u10m", "T2": "t2m"}
    p = prepare_ic_file(_fake_global(), tmp_path / "ic.nc", rename=ren)
    da = xr.open_dataarray(p)
    assert da.dims == ("time", "variable", "lat", "lon")
    assert da.sizes["lat"] == 721 and da.sizes["lon"] == 1440
    assert float(da.lat[0]) == 90.0 and float(da.lon[0]) == 0.0
    assert not np.isnan(da.values).any()
    # smooth field survives regridding
    assert abs(float(da.sel(variable="msl", lat=0.0, lon=0.0).isel(time=0)) - 100_800.0) < 1.0
    assert abs(float(da.sel(variable='msl', lat=90.0, lon=0.0).isel(time=0)) - 101_296.0) < 1.0
    da.close()
    assert check_ic_file(p, ["2025-04-11T12:00"], ["msl", "u10m", "t2m"]) == []
    probs = check_ic_file(p, ["2025-04-11T18:00"], ["msl", "z500"])
    assert any("z500" in x for x in probs) and any("missing times" in x for x in probs)
    # hPa instead of Pa is flagged
    p2 = prepare_ic_file(_fake_global(hpa=True), tmp_path / "ic_hpa.nc", rename=ren)
    assert any("unit" in x for x in check_ic_file(p2))


def test_regional_source_rejected(tmp_path):
    ds = _fake_global().sel(lat=slice(20, 50), lon=slice(110, 150))
    with pytest.raises(ValueError, match="not global"):
        prepare_ic_file(ds, tmp_path / "x.nc", rename={"PSL": "msl"})


def test_required_times():
    t = required_times(["2025-04-11T12:00"])
    assert [str(x) for x in t] == ["2025-04-11 06:00:00", "2025-04-11 12:00:00"]


def test_speed_bias_detected():
    times = pd.date_range("2030-08-01", periods=9, freq="6h")
    ref = pd.DataFrame({"time": times, "lat": 20 + 0.5 * np.arange(9), "lon": 130.0,
                        "msl_min": 95_000.0})
    # forecast moves 30 % faster and is 8 hPa too weak
    fc = pd.DataFrame({"time": times, "lat": 20 + 0.65 * np.arange(9), "lon": 130.0,
                       "msl_min": 95_800.0})
    e = track_errors(fc, ref)
    assert (e["speed_error_kmh"].dropna() > 0).all()
    b = track_bias(summarize_track_errors({"m": e}, times[0]))
    assert (b["along_track_km"] > 0).all() and (b["frac_ahead"] == 1).all()
    assert np.allclose(b["msl_error_hpa"], 8.0)
    assert set(b["lead_window"].astype(str)) == {"1-24h", "25-48h"}


def test_verify_with_ic_variant_and_second_reference(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "results"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_case.py"), "--out", str(runs)],
                   check=True)
    bt = runs / "synthetic_typhoon" / "ibtracs_synthetic.csv"
    main(["verify", "--case", "synthetic_typhoon", "--runs", str(runs), "--out", str(out), "--best-track", str(bt)])
    main(["verify", "--case", "synthetic_typhoon", "--runs", str(runs), "--out", str(out), "--truth", "ifs"])
    g = pd.read_csv(out / "synthetic_typhoon" / "grid_scores.csv")
    assert set(g["reference"]) == {"era5"}
    assert {"era5", "gfs"} <= set(g["ic"])
    assert "model_a@gfs" in set(g["model"]) and "truth_ifs" not in set(g["model"])
    gi = pd.read_csv(out / "synthetic_typhoon" / "ref_ifs" / "grid_scores.csv")
    assert set(gi["reference"]) == {"ifs"}
    # the worse initial state shows up as larger error than the same model from ERA5
    r = g[(g.variable == "msl") & (g.lead_h == 48)].set_index("model")["rmse"]
    assert r["model_a@gfs"] > r["model_a"]
    # analysis noise in the second reference raises lead-0 error of ERA5-started runs
    a0 = g[(g.variable == "u10m") & (g.lead_h == 0) & (g.model == "model_a")]["rmse"].iloc[0]
    b0 = gi[(gi.variable == "u10m") & (gi.lead_h == 0) & (gi.model == "model_a")]["rmse"].iloc[0]
    assert b0 > a0
    bias = pd.read_csv(out / "synthetic_typhoon" / "track_bias.csv")
    assert {"model_a", "model_b", "model_a@gfs"} <= set(bias["model"])


def test_cli_models(capsys):
    main(["models"])
    assert "fcn3" in capsys.readouterr().out
