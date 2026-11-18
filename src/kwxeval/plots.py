"""Figures: score vs lead time, and typhoon tracks."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#7f7f7f"]
UNITS = {"msl": "Pa", "u10m": "m s$^{-1}$", "v10m": "m s$^{-1}$", "t2m": "K", "z500": "m$^2$ s$^{-2}$"}


def score_vs_lead(scores: pd.DataFrame, path, score: str = "rmse"):
    variables = list(dict.fromkeys(scores["variable"]))
    models = list(dict.fromkeys(scores["model"]))
    fig, axes = plt.subplots(1, len(variables), figsize=(3.4 * len(variables), 3.2), squeeze=False)
    for ax, v in zip(axes[0], variables):
        for i, m in enumerate(models):
            s = scores[(scores.variable == v) & (scores.model == m)].groupby("lead_h")[score].mean()
            ax.plot(s.index, s.values, "o-", ms=3, color=COLORS[i % len(COLORS)], label=m)
        ax.set_title(v)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel(f"{score} ({UNITS.get(v, '')})" if score in ("rmse", "bias", "mae") else score)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def tracks(track_tables: dict[str, pd.DataFrame], path, extent=(115, 140, 20, 45)):
    """Plot tracks; uses cartopy coastlines when cartopy is installed."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(5.2, 5.2))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, lw=0.6)
        ax.gridlines(draw_labels=True, lw=0.3, alpha=0.5)
        kw = {"transform": ccrs.PlateCarree()}
    except ImportError:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("longitude (°E)")
        ax.set_ylabel("latitude (°N)")
        ax.grid(alpha=0.3)
        kw = {}
    for i, (name, t) in enumerate(track_tables.items()):
        is_ref = name.lower() in ("era5", "reference", "best track")
        color = "k" if is_ref else COLORS[i % len(COLORS)]
        ax.plot(t["lon"], t["lat"], "-", color=color, lw=1.8, marker="o", ms=3, label=name, **kw)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def track_error_vs_lead(err_tables: dict[str, pd.DataFrame], init_time, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    t0 = pd.Timestamp(init_time)
    for i, (name, e) in enumerate(err_tables.items()):
        lead = (pd.to_datetime(e["time"]) - t0) / np.timedelta64(1, "h")
        ax.plot(lead, e["position_error_km"], "o-", ms=3, color=COLORS[i % len(COLORS)], label=name)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("track error (km)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def spectra(table: pd.DataFrame, path, title: str = ""):
    """Power spectra (log-log); the analysis is drawn in black."""
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for i, (name, g) in enumerate(table.groupby("name", sort=False)):
        is_ref = name.lower().startswith("era5")
        ax.loglog(g["wavelength_km"], g["power"], "-", lw=2 if is_ref else 1.4,
                  color="k" if is_ref else COLORS[i % len(COLORS)], label=name)
    ax.invert_xaxis()
    ax.set_xlabel("wavelength (km)")
    ax.set_ylabel("power (relative)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def spread_skill_vs_lead(scores: pd.DataFrame, path):
    ens = scores[scores["members"] > 1] if "members" in scores else scores.iloc[0:0]
    if ens.empty:
        return
    variables = list(dict.fromkeys(ens["variable"]))
    fig, axes = plt.subplots(1, len(variables), figsize=(3.4 * len(variables), 3.0), squeeze=False)
    for ax, v in zip(axes[0], variables):
        for i, m in enumerate(dict.fromkeys(ens["model"])):
            s = ens[(ens.variable == v) & (ens.model == m)].groupby("lead_h")["spread_skill_ratio"].mean()
            ax.plot(s.index, s.values, "o-", ms=3, color=COLORS[i % len(COLORS)], label=m)
        ax.axhline(1, color="k", ls="--", lw=0.8)
        ax.set_title(v)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel("spread / RMSE")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
