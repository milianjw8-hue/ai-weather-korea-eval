"""Command line interface.

  kwxeval models                                                        list models and GPU memory
  kwxeval run    --case khanun_2023 --models fcn pangu6 --out runs/     (GPU, internet)
  kwxeval run    --case jeju_wind_2025 --models aifs_ens --ic ifs --members 8
  kwxeval truth  --case khanun_2023 --out runs/ [--source ifs]          (internet)
  kwxeval verify --case khanun_2023 --runs runs/ --out results/ [--truth ifs]   (CPU)
  kwxeval ic-check kim_ic.nc --case jeju_wind_2025 --variables msl u10m ...
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
from .besttrack import load_best_track
from .track import (ensemble_mean_track, forecast_to_valid_time, summarize_track_errors, track_bias, track_errors,
                    track_minimum)
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


def truth_file(rundir: Path, source: str) -> Path:
    return rundir / ("era5.nc" if source == "era5" else f"truth_{source}.nc")


def is_truth_file(p: Path) -> bool:
    return p.stem == "era5" or p.stem.startswith("truth_")


def cmd_models(args):
    from .run import model_table

    print(model_table().to_string(index=False))


def cmd_run(args):
    from .run import check_ic_times, forecast_label, run_forecast

    case = load_case(args.case, args.cases)
    check_ic_times(args.ic, case["init_times"])  # before any download
    for m in args.models:
        out = Path(args.out) / case["name"] / f"{forecast_label(m, args.ic)}.nc"
        run_forecast(m, case["init_times"], case["hours"], case["region_obj"], out, device=args.device,
                     ic=args.ic, ic_path=args.ic_path, members=args.members)
        print(f"saved {out}")


def cmd_truth(args):
    from .run import fetch_truth

    case = load_case(args.case, args.cases)
    out = truth_file(Path(args.out) / case["name"], args.source)
    fetch_truth(valid_times(case), case["region_obj"], out, source=args.source)
    print(f"saved {out}")


def cmd_ic_check(args):
    from .icfile import check_ic_file

    init = load_case(args.case, args.cases)["init_times"] if args.case else None
    problems = check_ic_file(args.path, init, args.variables)
    if problems:
        print("problems:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print("ok")


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


def _verify_tracks(case, truth, forecasts, out, args):
    """Track every model (and ERA5) and score them against the best track, or against the ERA5 track
    when no best track is configured."""
    search = case.get("search_km", 400)
    bt_cfg = case.get("best_track")
    if args.best_track:
        bt_cfg = {**(bt_cfg or {}), "path": args.best_track}
    best = None
    if bt_cfg:
        try:
            best = load_best_track(bt_cfg, Path.cwd())
        except FileNotFoundError as e:
            print(f"warning: best track not found ({e.filename}); verifying tracks against ERA5 instead")
    if best is not None:
        best.to_csv(out / "track_best.csv", index=False)
    ref_name = "best track" if best is not None else "ERA5"
    ws_truth = np.hypot(truth["u10m"], truth["v10m"]) if "u10m" in truth else None
    all_summaries = []
    for t0 in case["init_times"]:
        t0s = pd.Timestamp(t0).strftime("%Y%m%d%H")
        t0p = pd.Timestamp(t0)
        if best is not None:
            b = best[best.time >= t0p].reset_index(drop=True)
            if b.empty:
                continue
            fg = (b.loc[0, "lat"], b.loc[0, "lon"])
        else:
            fg = tuple(case["first_guess"])
        era5 = track_minimum(truth["msl"].sel(time=truth["time"] >= np.datetime64(t0p)), fg, search,
                             None if ws_truth is None else ws_truth.sel(time=truth["time"] >= np.datetime64(t0p)))
        era5.to_csv(out / f"track_era5_{t0s}.csv", index=False)
        if era5.empty:
            continue
        if best is None:
            ref_t, tracks, errs = era5, {"ERA5": era5}, {}
        else:
            ref_t = best[(best.time >= t0p) & (best.time <= era5.time.max())].reset_index(drop=True)
            tracks = {"best track": ref_t, "ERA5": era5}
            errs = {"ERA5": track_errors(era5, ref_t)}  # analysis-vs-best-track baseline
        fg = (era5.loc[0, "lat"], era5.loc[0, "lon"]) if best is None else fg
        for name, fc in forecasts.items():
            if name == "persistence":
                continue
            f = forecast_to_valid_time(standardize(fc), t0)
            ws = np.hypot(f["u10m"], f["v10m"]) if "u10m" in f else None
            if ENSEMBLE_DIM in f["msl"].dims:
                member_tracks = [
                    track_minimum(f["msl"].isel({ENSEMBLE_DIM: k}), fg, search,
                                  None if ws is None else ws.isel({ENSEMBLE_DIM: k})).assign(member=k)
                    for k in range(f.sizes[ENSEMBLE_DIM])]
                mt = pd.concat(member_tracks, ignore_index=True)
                mt.to_csv(out / f"track_{name}_members_{t0s}.csv", index=False)
                tr = ensemble_mean_track(mt)
                name = f"{name} (mean)"
            else:
                tr = track_minimum(f["msl"], fg, search, ws)
            tr.to_csv(out / f"track_{_safe(name)}_{t0s}.csv", index=False)
            tracks[name] = tr
            errs[name] = track_errors(tr, ref_t)
        for name, e in errs.items():
            e.to_csv(out / f"track_error_{_safe(name)}_{t0s}.csv", index=False)
        r = case["region_obj"]
        plots.tracks(tracks, out / f"tracks_{t0s}.png", (r.lon_min, r.lon_max, r.lat_min, r.lat_max), ref_name)
        plots.track_error_vs_lead(errs, t0, out / f"track_error_{t0s}.png")
        summ = summarize_track_errors(errs, t0)
        plots.track_components(summ, out / f"track_components_{t0s}.png", ref_name)
        all_summaries.append(summ)
    if all_summaries:
        summ = pd.concat(all_summaries, ignore_index=True)
        summ = add_label_columns(summ, "best_track" if best is not None else "era5")
        summ.to_csv(out / "track_summary.csv", index=False)
        bias = track_bias(summ)
        if not bias.empty:
            bias.to_csv(out / "track_bias.csv", index=False)
            print(f"\nsystematic track/intensity errors against {ref_name} "
                  "(along>0 too fast, msl>0 too weak):")
            print(bias.to_string(index=False))
        by_lead = summ.groupby(["model", "lead_h"]).mean(numeric_only=True).round(1)
        by_lead.to_csv(out / "track_summary_by_lead.csv")
        print(f"\ntrack errors against {ref_name} (mean over initial times):")
        print(by_lead.to_string())


def add_label_columns(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Split 'model@ic' labels into base_model and ic, and record the verifying reference."""
    from .run import split_label

    if df.empty or "model" not in df:
        return df
    base, ic = zip(*(split_label(str(m).replace(" (mean)", "")) for m in df["model"]))
    df = df.copy()
    df["base_model"], df["ic"], df["reference"] = list(base), list(ic), reference
    return df


def cmd_verify(args):
    case = load_case(args.case, args.cases)
    rundir = Path(args.runs) / case["name"]
    out = Path(args.out) / case["name"]
    out.mkdir(parents=True, exist_ok=True)

    ref = args.truth
    if ref != "era5":
        out = out / f"ref_{ref}"
        out.mkdir(parents=True, exist_ok=True)
    tpath = truth_file(rundir, ref)
    if not tpath.exists():
        raise SystemExit(f"{tpath} not found: run `kwxeval truth --case {case['name']} --source {ref}` first")
    truth = standardize(xr.open_dataset(tpath))
    forecasts = {p.stem: xr.open_dataset(p) for p in sorted(rundir.glob("*.nc")) if not is_truth_file(p)}
    if not forecasts:
        raise SystemExit(f"no forecast files in {rundir}")
    if args.persistence:
        forecasts["persistence"] = persistence(truth, case["init_times"], range(0, case["hours"] + 1, 6))
    variables = args.variables or [v for v in truth.data_vars]

    scores = grid_scores(forecasts, truth, variables, case["region_obj"])
    scores = add_label_columns(scores, ref)
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
    summary = {"case": case["name"], "models": list(forecasts), "variables": variables, "reference": ref}

    if case.get("track"):
        _verify_tracks(case, truth, forecasts, out, args)
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
    r.add_argument("--ic", default="era5", choices=["era5", "ifs", "gfs", "file"],
                   help="initial conditions (ifs: from 2024-03-01, gfs: from 2021-01-01)")
    r.add_argument("--ic-path", help="NetCDF from kwxeval.icfile.prepare_ic_file (with --ic file)")
    r.add_argument("--members", type=int, default=1, help="members for stochastic (ens) models")
    r.set_defaults(func=cmd_run)
    ms = sub.add_parser("models", help="list registered models")
    ms.set_defaults(func=cmd_models)
    t = sub.add_parser("truth")
    t.add_argument("--case", required=True)
    t.add_argument("--out", default="runs")
    t.add_argument("--source", default="era5", choices=["era5", "ifs"])
    t.set_defaults(func=cmd_truth)
    ic = sub.add_parser("ic-check", help="check an initial-condition file made with kwxeval.icfile")
    ic.add_argument("path")
    ic.add_argument("--case")
    ic.add_argument("--variables", nargs="+")
    ic.set_defaults(func=cmd_ic_check)
    v = sub.add_parser("verify")
    v.add_argument("--case", required=True)
    v.add_argument("--runs", default="runs")
    v.add_argument("--out", default="results")
    v.add_argument("--variables", nargs="+")
    v.add_argument("--persistence", action="store_true", help="add a persistence baseline")
    v.add_argument("--spectrum-variable", help="variable for the power-spectrum figure (default: first)")
    v.add_argument("--best-track", help="IBTrACS CSV (overrides the case's best_track.path)")
    v.add_argument("--truth", default="era5", choices=["era5", "ifs"],
                   help="verifying analysis; non-ERA5 results go to <out>/<case>/ref_<truth>/")
    v.set_defaults(func=cmd_verify)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
