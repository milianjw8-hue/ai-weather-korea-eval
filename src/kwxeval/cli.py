"""Command line interface.

  kwxeval run    --case khanun_2023 --models fcn pangu6 --out runs/     (GPU, internet)
  kwxeval truth  --case khanun_2023 --out runs/                         (internet)
  kwxeval verify --case khanun_2023 --runs runs/ --out results/         (CPU)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from . import plots
from .regions import REGIONS
from .track import ensemble_mean_track, forecast_to_valid_time, track_errors, track_minimum
from .spectra import spectra_table
from .verify import ENSEMBLE_DIM, common_grid, grid_scores, persistence, standardize, to_grid

ROOT_CASES = Path(__file__).resolve().parents[2] / "cases" / "cases.yaml"


def load_case(name: str, path: str | None = None) -> dict:
    cases = yaml.safe_load(Path(path or ROOT_CASES).read_text(encoding="utf-8"))
    case = cases[name]
    case["name"] = name
    case["region_obj"] = REGIONS[case.get("region", "korea")]
    return case


def valid_times(case) -> list[pd.Timestamp]:
    t = set()
    for t0 in pd.to_datetime(case["init_times"]):
        for h in range(0, case["hours"] + 1, 6):
            t.add(t0 + pd.Timedelta(hours=h))
    return sorted(t)


def cmd_run(args):
    from .run import run_forecast

    case = load_case(args.case, args.cases)
    for m in args.models:
        out = Path(args.out) / case["name"] / f"{m}.nc"
        run_forecast(m, case["init_times"], case["hours"], case["region_obj"], out, device=args.device)
        print(f"saved {out}")


def cmd_truth(args):
    from .run import fetch_truth

    case = load_case(args.case, args.cases)
    out = Path(args.out) / case["name"] / "era5.nc"
    fetch_truth(valid_times(case), case["region_obj"], out)
    print(f"saved {out}")


def _safe(name: str) -> str:
    return name.replace(" (mean)", "_mean").replace(" ", "_")


def _spectra_at_last_lead(forecasts, truth, case, variable):
    """Spectra of truth and each model at the longest lead of the first initial time (common grid)."""
    if variable not in truth:
        return None
    t0 = np.datetime64(pd.Timestamp(case["init_times"][0]))
    fcs = {k: standardize(v) for k, v in forecasts.items() if variable in v}
    lat, lon = common_grid(fcs, truth, case["region_obj"])
    fields, lead = {}, None
    for name, fc in fcs.items():
        if t0 not in fc["time"].values:
            continue
        leads = [lt for lt in fc["lead_time"].values if t0 + lt in truth["time"].values]
        if not leads:
            continue
        lead = max(leads) if lead is None else min(lead, max(leads))
    if lead is None:
        return None
    fields["ERA5"] = to_grid(truth[[variable]], lat, lon)[variable].sel(time=t0 + lead)
    for name, fc in fcs.items():
        f = to_grid(fc[[variable]], lat, lon)[variable].sel(time=t0, lead_time=lead)
        if ENSEMBLE_DIM in f.dims:
            fields[f"{name} (member 0)"] = f.isel({ENSEMBLE_DIM: 0})
            fields[f"{name} (mean)"] = f.mean(ENSEMBLE_DIM)
        else:
            fields[name] = f
    title = f"{variable}, lead {int(lead / np.timedelta64(1, 'h'))} h from {pd.Timestamp(t0):%Y-%m-%d %H} UTC"
    return spectra_table(fields), title


def cmd_verify(args):
    case = load_case(args.case, args.cases)
    rundir = Path(args.runs) / case["name"]
    out = Path(args.out) / case["name"]
    out.mkdir(parents=True, exist_ok=True)

    truth = standardize(xr.open_dataset(rundir / "era5.nc"))
    forecasts = {p.stem: xr.open_dataset(p) for p in sorted(rundir.glob("*.nc")) if p.stem != "era5"}
    if not forecasts:
        raise SystemExit(f"no forecast files in {rundir}")
    if args.persistence:
        forecasts["persistence"] = persistence(truth, case["init_times"], range(0, case["hours"] + 1, 6))
    variables = args.variables or [v for v in truth.data_vars]

    scores = grid_scores(forecasts, truth, variables, case["region_obj"])
    scores.to_csv(out / "grid_scores.csv", index=False)
    plots.score_vs_lead(scores, out / "rmse_vs_lead.png", "rmse")
    plots.score_vs_lead(scores, out / "pattern_corr_vs_lead.png", "pattern_corr")
    plots.score_vs_lead(scores, out / "crps_vs_lead.png", "crps")
    plots.spread_skill_vs_lead(scores, out / "spread_skill_vs_lead.png")
    spec = _spectra_at_last_lead(forecasts, truth, case, args.spectrum_variable or variables[0])
    if spec is not None:
        table, title = spec
        table.to_csv(out / "spectra.csv", index=False)
        plots.spectra(table, out / "spectra.png", title)
    summary = {"case": case["name"], "models": list(forecasts), "variables": variables}

    if case.get("track"):
        ws_truth = np.hypot(truth["u10m"], truth["v10m"]) if "u10m" in truth else None
        ref = track_minimum(truth["msl"], tuple(case["first_guess"]), case.get("search_km", 400), ws_truth)
        ref.to_csv(out / "track_era5.csv", index=False)
        for t0 in case["init_times"]:
            t0s = pd.Timestamp(t0).strftime("%Y%m%d%H")
            ref_t = ref[ref.time >= pd.Timestamp(t0)].reset_index(drop=True)
            if ref_t.empty:
                continue
            fg = (ref_t.loc[0, "lat"], ref_t.loc[0, "lon"])
            tracks, errs = {"ERA5": ref_t}, {}
            for name, fc in forecasts.items():
                if name == "persistence":
                    continue
                f = forecast_to_valid_time(standardize(fc), t0)
                ws = np.hypot(f["u10m"], f["v10m"]) if "u10m" in f else None
                if ENSEMBLE_DIM in f["msl"].dims:
                    member_tracks = [
                        track_minimum(f["msl"].isel({ENSEMBLE_DIM: k}), fg, case.get("search_km", 400),
                                      None if ws is None else ws.isel({ENSEMBLE_DIM: k})).assign(member=k)
                        for k in range(f.sizes[ENSEMBLE_DIM])]
                    mt = pd.concat(member_tracks, ignore_index=True)
                    mt.to_csv(out / f"track_{name}_members_{t0s}.csv", index=False)
                    tr = ensemble_mean_track(mt)
                    name = f"{name} (mean)"
                else:
                    tr = track_minimum(f["msl"], fg, case.get("search_km", 400), ws)
                tr.to_csv(out / f"track_{_safe(name)}_{t0s}.csv", index=False)
                tracks[name] = tr
                errs[name] = track_errors(tr, ref_t)
                errs[name].to_csv(out / f"track_error_{_safe(name)}_{t0s}.csv", index=False)
            r = case["region_obj"]
            plots.tracks(tracks, out / f"tracks_{t0s}.png", (r.lon_min, r.lon_max, r.lat_min, r.lat_max))
            plots.track_error_vs_lead(errs, t0, out / f"track_error_{t0s}.png")
        summary["track"] = True

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    mean = scores.groupby(["variable", "model", "lead_h"])[["rmse", "pattern_corr", "crps"]].mean().round(3)
    print(mean.to_string())
    print(f"results in {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kwxeval")
    p.add_argument("--cases", help="cases YAML (default: cases/cases.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--case", required=True)
    r.add_argument("--models", nargs="+", default=["fcn"])
    r.add_argument("--out", default="runs")
    r.add_argument("--device", default="cuda")
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("truth")
    t.add_argument("--case", required=True)
    t.add_argument("--out", default="runs")
    t.set_defaults(func=cmd_truth)
    v = sub.add_parser("verify")
    v.add_argument("--case", required=True)
    v.add_argument("--runs", default="runs")
    v.add_argument("--out", default="results")
    v.add_argument("--variables", nargs="+")
    v.add_argument("--persistence", action="store_true", help="add a persistence baseline")
    v.add_argument("--spectrum-variable", help="variable for the power-spectrum figure (default: first)")
    v.set_defaults(func=cmd_verify)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
