# Changelog

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
