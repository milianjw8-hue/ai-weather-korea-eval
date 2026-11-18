import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kwxeval.cli import main
from kwxeval.regions import KOREA
from kwxeval.spectra import power_spectrum
from kwxeval.track import ensemble_mean_track
from kwxeval.verify import grid_scores

ROOT = Path(__file__).resolve().parents[1]
LAT = np.arange(45.0, 29.99, -0.5)
LON = np.arange(120.0, 135.01, 0.5)
TIMES = pd.date_range("2030-01-01", periods=3, freq="6h").values
LEADS = np.array([0, 6, 12]).astype("timedelta64[h]")


def _truth(rng):
    return xr.Dataset({"t2m": (("time", "lat", "lon"), rng.normal(size=(3, len(LAT), len(LON))))},
                      coords={"time": TIMES, "lat": LAT, "lon": LON})


def _ens(values):
    return xr.Dataset({"t2m": (("ensemble", "time", "lead_time", "lat", "lon"), values)},
                      coords={"ensemble": np.arange(values.shape[0]), "time": TIMES[:1], "lead_time": LEADS,
                              "lat": LAT, "lon": LON})


def test_perfect_ensemble_scores_zero():
    rng = np.random.default_rng(0)
    tr = _truth(rng)
    vals = np.stack([tr["t2m"].values] * 4)[:, None]  # every member = truth
    s = grid_scores({"ens": _ens(vals)}, tr, ["t2m"], KOREA)
    assert np.allclose(s["crps"], 0, atol=1e-12)
    assert np.allclose(s["rmse"], 0, atol=1e-12)
    assert (s["members"] == 4).all()


def test_consistent_ensemble_spread_skill_near_one():
    rng = np.random.default_rng(1)
    tr = _truth(rng)
    # truth is drawn like a member around the ensemble centre -> statistically consistent ensemble
    centre = tr["t2m"].values + 0.0
    vals = centre[None, None] + rng.normal(size=(20, 1, 3, len(LAT), len(LON)))
    tr2 = tr.copy()
    tr2["t2m"].values[:] = centre + rng.normal(size=centre.shape)
    s = grid_scores({"ens": _ens(vals)}, tr2, ["t2m"], KOREA)
    assert s["spread_skill_ratio"].mean() == pytest.approx(1.0, abs=0.1)
    assert (s["crps_fair"] < s["crps"]).all()


def test_deterministic_crps_equals_mae():
    rng = np.random.default_rng(2)
    tr = _truth(rng)
    det = xr.Dataset({"t2m": (("time", "lead_time", "lat", "lon"),
                              tr["t2m"].values[None] + 1.0)},
                     coords={"time": TIMES[:1], "lead_time": LEADS, "lat": LAT, "lon": LON})
    s = grid_scores({"det": det}, tr, ["t2m"], KOREA)
    assert np.allclose(s["crps"], s["mae"]) and np.allclose(s["crps"], 1.0)


def test_spectrum_peak_and_smoothing():
    ny, nx, dx = 128, 128, 10.0
    x = np.arange(nx) * dx
    wave = np.tile(np.sin(2 * np.pi * x / 200.0), (ny, 1))  # 200 km wavelength
    wl, pw = power_spectrum(wave, dx, dx)
    assert wl[np.argmax(pw)] == pytest.approx(200, rel=0.15)
    rng = np.random.default_rng(3)
    noise = rng.normal(size=(ny, nx))
    smooth = (noise + np.roll(noise, 1, 0) + np.roll(noise, 1, 1) + np.roll(noise, -1, 0) + np.roll(noise, -1, 1)) / 5
    _, p_noise = power_spectrum(noise, dx, dx)
    _, p_smooth = power_spectrum(smooth, dx, dx)
    assert p_smooth[-1] / p_noise[-1] < 0.2  # shortest wavelengths damped


def test_ensemble_mean_track():
    t = pd.Timestamp("2030-01-01")
    mt = pd.DataFrame({"time": [t, t], "lat": [30.0, 31.0], "lon": [125.0, 125.0], "msl_min": [98000, 99000],
                       "member": [0, 1]})
    m = ensemble_mean_track(mt)
    assert m.loc[0, "lat"] == 30.5 and m.loc[0, "msl_min"] == 98500
    assert m.loc[0, "track_spread_km"] == pytest.approx(55.6, abs=0.5)


def test_cli_synthetic_with_ensemble(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_case.py"), "--out", str(tmp_path / "runs")],
                   check=True)
    main(["verify", "--case", "synthetic_typhoon", "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "res"),
          "--spectrum-variable", "msl"])
    out = tmp_path / "res" / "synthetic_typhoon"
    s = pd.read_csv(out / "grid_scores.csv")
    ens = s[s.model == "model_ens"]
    assert (ens["members"] == 6).all() and ens["crps_fair"].notna().all()
    late = s[(s.variable == "msl") & (s.lead_h == 96)].set_index("model")
    assert late.loc["model_ens", "crps"] < late.loc["model_b", "crps"]
    for f in ("crps_vs_lead.png", "spread_skill_vs_lead.png", "spectra.png", "spectra.csv",
              "track_model_ens_mean_2030080100.csv", "track_model_ens_members_2030080100.csv"):
        assert (out / f).exists(), f
    tr = pd.read_csv(out / "track_model_ens_mean_2030080100.csv")
    assert tr["track_spread_km"].iloc[-1] > tr["track_spread_km"].iloc[1]
