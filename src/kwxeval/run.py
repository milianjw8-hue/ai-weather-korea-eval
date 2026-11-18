"""Run open AI weather models with NVIDIA Earth2Studio and save a regional subset.

Requires a CUDA GPU and `pip install "kwxeval[run]"` plus the model extras of Earth2Studio
(see the Earth2Studio install guide). Written against the Earth2Studio 0.20 API.

Initial conditions (``ic``):

- ``era5``  ARCO ERA5 on Google Cloud. Reproducible; what most models were trained on.
- ``ifs``   ECMWF IFS open data (AWS mirror). Operational-like. Earth2Studio serves it only
            from 2024-03-01 onward.
- ``gfs``   NOAA GFS analysis (AWS). Operational-like. From 2021-01-01 onward.
- ``file``  A local NetCDF prepared with :func:`kwxeval.icfile.prepare_ic_file`, e.g. from KIM.

Running the same model from several initial conditions separates model error from
initial-condition error; see docs/EXPERIMENT_DESIGN.md.
"""

from __future__ import annotations

import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .regions import Region


@dataclass(frozen=True)
class ModelSpec:
    cls: str            # class name in earth2studio.models.px
    step_h: int         # hours per model call
    kind: str           # "det" (deterministic) or "ens" (stochastic: members from model noise)
    gpu_gb: int         # GPU memory listed in the Earth2Studio model badges
    note: str = ""


# GPU memory: Earth2Studio 0.20 model badges. Check each checkpoint's licence before publishing results.
MODELS: dict[str, ModelSpec] = {
    "fcn": ModelSpec("FCN", 6, "det", 40),
    "sfno": ModelSpec("SFNO", 6, "det", 40),
    "pangu6": ModelSpec("Pangu6", 6, "det", 40),
    "pangu24": ModelSpec("Pangu24", 24, "det", 40),
    "graphcast_small": ModelSpec("GraphCastSmall", 6, "det", 40, "1.0 deg"),
    "graphcast_op": ModelSpec("GraphCastOperational", 6, "det", 40, "fine-tuned on HRES"),
    "fuxi": ModelSpec("FuXi", 6, "det", 40),
    "fengwu": ModelSpec("FengWu", 6, "det", 40),
    "aifs": ModelSpec("AIFS", 6, "det", 40),
    "aifs_ens": ModelSpec("AIFSENS", 6, "ens", 48),
    "aifs2": ModelSpec("AIFS2", 6, "det", 40),
    "aifs2_ens": ModelSpec("AIFS2ENS", 6, "ens", 40),
    "fcn3": ModelSpec("FCN3", 6, "ens", 80),
    "aurora": ModelSpec("Aurora", 6, "det", 48),
    "aurora15_ens": ModelSpec("Aurora1p5Ensemble_6h", 6, "ens", 48),
    "wn2_cyclones_mini": ModelSpec("WeatherNext2CyclonesMini", 6, "ens", 40, "1.0 deg, check licence"),
    "wn2_cyclones": ModelSpec("WeatherNext2Cyclones", 6, "ens", 80, "check licence"),
}

IC_SOURCES = ("era5", "ifs", "gfs", "file")
IC_EARLIEST = {"ifs": pd.Timestamp("2024-03-01"), "gfs": pd.Timestamp("2021-01-01")}

DEFAULT_VARIABLES = ["msl", "u10m", "v10m", "t2m", "z500"]


def model_table() -> pd.DataFrame:
    return pd.DataFrame([{"name": k, "class": s.cls, "step_h": s.step_h, "kind": s.kind,
                          "gpu_gb": s.gpu_gb, "note": s.note} for k, s in MODELS.items()])


def forecast_label(model: str, ic: str = "era5") -> str:
    """File stem / model label. ERA5 runs keep the bare model name (backward compatible)."""
    return model if ic == "era5" else f"{model}@{ic}"


def split_label(label: str) -> tuple[str, str]:
    """'pangu6@gfs' -> ('pangu6', 'gfs'); 'pangu6' -> ('pangu6', 'era5')."""
    model, _, ic = label.partition("@")
    return model, (ic or "era5")


def check_ic_times(ic: str, init_times) -> None:
    """Fail early (before downloading weights) if an initial time is outside the source's archive."""
    if ic not in IC_SOURCES:
        raise ValueError(f"unknown ic '{ic}', choose from {IC_SOURCES}")
    first = IC_EARLIEST.get(ic)
    if first is None:
        return
    # models with a -6 h input state also need the earlier time
    bad = [str(t) for t in pd.to_datetime(init_times) if t - pd.Timedelta(hours=6) < first]
    if bad:
        raise ValueError(f"ic={ic} is available from {first:%Y-%m-%d} only (Earth2Studio 0.20); "
                         f"initial time(s) too early: {', '.join(bad)}. Use ic=era5 or gfs for older cases.")


def ic_source(ic: str, ic_path: str | Path | None = None):
    """Return an Earth2Studio data source for the initial conditions."""
    from earth2studio import data

    if ic == "era5":
        return data.ARCO_ERA5()
    if ic == "ifs":
        return data.IFS(source="aws")
    if ic == "gfs":
        return data.GFS(source="aws")
    if ic == "file":
        if not ic_path:
            raise ValueError("ic=file needs ic_path (see kwxeval.icfile.prepare_ic_file)")
        return data.DataArrayFile(str(ic_path))
    raise ValueError(f"unknown ic '{ic}'")


def load_model(name: str):
    from earth2studio.models import px

    cls = getattr(px, MODELS[name].cls)
    return cls.load_model(cls.load_default_package())


def run_forecast(model_name: str, init_times: list[str], hours: int, region: Region, out_path: str | Path,
                 variables: list[str] | None = None, device: str = "cuda", ic: str = "era5",
                 ic_path: str | Path | None = None, members: int = 1) -> Path:
    """Run one model and write a region-cropped NetCDF (time, lead_time, [ensemble,] lat, lon).

    Stochastic models (kind "ens") run `members` members with no initial perturbation: the spread comes
    from the model's own noise. Deterministic models ignore `members`.
    """
    import torch
    from earth2studio.io import NetCDF4Backend
    from earth2studio.run import deterministic, ensemble

    spec = MODELS[model_name]
    check_ic_times(ic, init_times)
    variables = variables or DEFAULT_VARIABLES
    nsteps = int(np.ceil(hours / spec.step_h))
    source = ic_source(ic, ic_path)
    model = load_model(model_name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords = OrderedDict({"variable": np.array(variables)})

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.nc"
        io = NetCDF4Backend(str(raw), backend_kwargs={"mode": "w"})
        if spec.kind == "ens" and members > 1:
            from earth2studio.perturbation import Zero

            ensemble(init_times, nsteps, members, model, source, io, Zero(),
                     batch_size=1, output_coords=coords, device=torch.device(device))
        else:
            deterministic(init_times, nsteps, model, source, io, output_coords=coords,
                          device=torch.device(device))
        io.close()
        with xr.open_dataset(raw) as ds:
            sub = region.crop(ds).load()
    sub.attrs.update({"model": model_name, "initial_conditions": ic, "region": region.name,
                      "members": members if spec.kind == "ens" else 1})
    sub.to_netcdf(out_path)
    return out_path


TRUTH_SOURCES = ("era5", "ifs")


def fetch_truth(valid_times, region: Region, out_path: str | Path, variables: list[str] | None = None,
                source: str = "era5") -> Path:
    """Save verifying analysis fields at valid times (time, lat, lon): ERA5 or IFS analysis."""
    from earth2studio import data

    variables = variables or DEFAULT_VARIABLES
    if source == "era5":
        ds_src, label = data.ARCO_ERA5(), "ARCO ERA5 (Google Cloud)"
    elif source == "ifs":
        first = pd.to_datetime(list(valid_times)).min()
        if first < IC_EARLIEST["ifs"]:
            raise ValueError(f"IFS analysis is available from {IC_EARLIEST['ifs']:%Y-%m-%d} only")
        ds_src, label = data.IFS(source="aws"), "ECMWF IFS open data (AWS)"
    else:
        raise ValueError(f"unknown truth source '{source}', choose from {TRUTH_SOURCES}")
    da = ds_src(list(valid_times), variables)  # (time, variable, lat, lon)
    da = region.crop(da)
    ds = da.to_dataset(dim="variable").load()
    ds.attrs.update({"source": label, "reference": source, "region": region.name})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    return out_path
