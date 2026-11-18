# Changelog

## 0.4.0 (2026-11-18)

- Initial conditions: `kwxeval run --ic era5|gfs|ifs|file --ic-path`; output named `<model>@<ic>.nc`
  (ERA5 runs keep the bare name). Archive limits checked before download (IFS open data from 2024-03-01,
  GFS from 2021-01-01, including the -6 h state)
- `kwxeval.icfile`: regrid a global analysis (e.g. KIM) to the Earth2Studio 0.25 deg grid,
  `check_ic_file` (dims, grid, times, NaN, unit sanity), `kwxeval ic-check`
- Model registry with kind and GPU memory: adds GraphCast operational, FuXi, FengWu, AIFS, AIFS ENS,
  AIFS2, AIFS2 ENS, FCN3, Aurora, Aurora 1.5 ensemble, WeatherNext 2 Cyclones; `kwxeval models`
- Stochastic models run as ensembles (`--members`) with zero initial perturbation
- Second verifying analysis: `kwxeval truth --source ifs`, `kwxeval verify --truth ifs` (-> `ref_ifs/`);
  `base_model`, `ic`, `reference` columns in grid and track tables
- Typhoon: translation-speed error and `track_bias.csv` (signed mean along/cross-track, share ahead,
  speed, central pressure and maximum wind errors by lead window)
- Synthetic case adds `model_a@gfs` and `truth_ifs.nc`; docs/EXPERIMENT_DESIGN.md

## 0.3.0 (2026-11-18)

- Best-track verification: IBTrACS v04r01 CSV reader (`kwxeval.besttrack`) with agency choice
  (WMO, TOKYO, KMA, CMA, HKO, USA), knots -> m/s and mb -> Pa, synoptic-time filter
- Case option `best_track` and CLI `--best-track`; falls back to the ERA5 track if the file is missing
- Along-track (+ = ahead) and cross-track (+ = right of motion) error decomposition,
  central-pressure error (hPa) and maximum-wind error against the best track
- ERA5 analysis track scored against the best track as a baseline
- New outputs: track_best.csv, track_summary.csv, track_summary_by_lead.csv, track_components_*.png

## 0.2.0 (2026-11-18)

- Ensemble forecasts (dimension `ensemble`, as written by Earth2Studio): area-weighted CRPS, fair CRPS,
  spread-skill ratio; deterministic models get CRPS = MAE
- Typhoon tracking per member, ensemble-mean track and track spread (km)
- Radially averaged power spectra of truth and models at the last common lead (`spectra.png`)
- New figures: crps_vs_lead, spread_skill_vs_lead; `--spectrum-variable` option
- Synthetic case now includes a 6-member ensemble
- `kwxeval.probabilistic` shared with ai-downscale-verify

## 0.1.0 (2026-11-18)

- Earth2Studio runner, ERA5 truth download, gridded verification, persistence baseline, typhoon tracker
