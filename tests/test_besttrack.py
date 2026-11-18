import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kwxeval.besttrack import KNOT, read_ibtracs
from kwxeval.cli import main
from kwxeval.track import track_errors

ROOT = Path(__file__).resolve().parents[1]

IBTRACS = """SID,SEASON,NUMBER,BASIN,SUBBASIN,NAME,ISO_TIME,NATURE,LAT,LON,WMO_WIND,WMO_PRES,WMO_AGENCY,KMA_LAT,KMA_LON,KMA_WIND,KMA_PRES
 ,Year, , , , , , ,degrees_north,degrees_east,kts,mb, ,degrees_north,degrees_east,kts,mb
2023210N14148,2023,60,WP,MM,KHANUN,2023-08-09 00:00:00,TS,30.0,129.0,70,960,tokyo,30.1,129.1,75,958
2023210N14148,2023,60,WP,MM,KHANUN,2023-08-09 03:00:00,TS,30.5,129.0,70,960,tokyo,,,,
2023210N14148,2023,60,WP,MM,KHANUN,2023-08-09 06:00:00,TS,31.0,129.0,65,965,tokyo,31.1,129.1,70,962
2017200N10140,2017,40,WP,MM,KHANUN,2017-10-12 00:00:00,TS,18.0,118.0,50,985,tokyo,,,,
"""


@pytest.fixture
def ib(tmp_path):
    p = tmp_path / "ib.csv"
    p.write_text(IBTRACS, encoding="utf-8")
    return p


def test_read_ibtracs_units_and_selection(ib):
    t = read_ibtracs(ib, name="khanun", season=2023)
    assert len(t) == 2  # units row dropped, 03 UTC non-synoptic fix dropped
    assert t.loc[0, "msl_min"] == 96000 and t.loc[0, "wind_max"] == pytest.approx(70 * KNOT)
    assert len(read_ibtracs(ib, name="khanun", season=2023, synoptic_only=False)) == 3
    k = read_ibtracs(ib, sid="2023210N14148", agency="KMA")
    assert k.loc[0, "lat"] == 30.1 and k.loc[0, "msl_hpa"] == 958
    with pytest.raises(ValueError):
        read_ibtracs(ib, name="khanun", season=1999)


def _track(lats, lons, start="2030-01-01"):
    t = pd.date_range(start, periods=len(lats), freq="6h")
    return pd.DataFrame({"time": t, "lat": lats, "lon": lons, "msl_min": 96000.0})


def test_along_and_cross_track_signs():
    ref = _track([20.0, 21.0, 22.0], [130.0, 130.0, 130.0])  # moving due north
    ahead = _track([20.5, 21.5, 22.5], [130.0, 130.0, 130.0])
    right = _track([20.0, 21.0, 22.0], [130.5, 130.5, 130.5])  # east of a northward track = right
    a = track_errors(ahead, ref)
    r = track_errors(right, ref)
    assert (a["along_track_km"] > 50).all() and np.allclose(a["cross_track_km"], 0, atol=1e-6)
    assert (r["cross_track_km"] > 40).all() and np.allclose(r["along_track_km"], 0, atol=1e-6)
    assert np.allclose(np.hypot(r["along_track_km"], r["cross_track_km"]), r["position_error_km"], rtol=0.01)


def test_cli_against_best_track(tmp_path):
    runs = tmp_path / "runs"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_case.py"), "--out", str(runs)], check=True)
    main(["verify", "--case", "synthetic_typhoon", "--runs", str(runs), "--out", str(tmp_path / "res"),
          "--best-track", str(runs / "synthetic_typhoon" / "ibtracs_synthetic.csv")])
    out = tmp_path / "res" / "synthetic_typhoon"
    s = pd.read_csv(out / "track_summary_by_lead.csv")
    last = s[s.lead_h == 96].set_index("model")
    assert last.loc["ERA5", "position_error_km"] < 30  # analysis close to best track (grid spacing)
    assert last.loc["model_a", "position_error_km"] < last.loc["model_b", "position_error_km"]
    assert last.loc["model_b", "cross_track_km"] > 0  # model_b is displaced east of a NNE-moving storm
    assert (out / "track_best.csv").exists() and (out / "track_components_2030080100.png").exists()


def test_missing_best_track_falls_back_to_era5(tmp_path, capsys):
    runs = tmp_path / "runs"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_case.py"), "--out", str(runs)], check=True)
    main(["verify", "--case", "synthetic_typhoon", "--runs", str(runs), "--out", str(tmp_path / "res"),
          "--best-track", str(tmp_path / "nope.csv")])
    assert "verifying tracks against ERA5" in capsys.readouterr().out
