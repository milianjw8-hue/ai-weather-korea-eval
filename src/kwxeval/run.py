"""Run open AI weather models with NVIDIA Earth2Studio and save a regional subset.

Requires a CUDA GPU and `pip install "kwxeval[run]"` plus the model extras of Earth2Studio
(see the Earth2Studio install guide). Initial conditions come from ARCO ERA5 on Google Cloud
(public, no account needed). Earth2Studio lists ~40 GB of GPU memory for most models; smaller
GPUs may fail with out-of-memory errors.

Tested against the Earth2Studio 0.20 API.
"""

from __future__ import annotations

import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xarray as xr

from .regions import Region

# name -> (module attribute in earth2studio.models.px, model time step in hours)
MODELS = {
    "fcn": ("FCN", 6),
    "sfno": ("SFNO", 6),
    "pangu6": ("Pangu6", 6),
    "pangu24": ("Pangu24", 24),
    "graphcast_small": ("GraphCastSmall", 6),
}

DEFAULT_VARIABLES = ["msl", "u10m", "v10m", "t2m", "z500"]


def load_model(name: str):
    from earth2studio.models import px

    cls_name, _ = MODELS[name]
    cls = getattr(px, cls_name)
    return cls.load_model(cls.load_default_package())


def run_forecast(model_name: str, init_times: list[str], hours: int, region: Region, out_path: str | Path,
                 variables: list[str] | None = None, device: str = "cuda") -> Path:
    """Run a deterministic forecast and write a region-cropped NetCDF (time, lead_time, lat, lon)."""
    import torch
    from earth2studio.data import ARCO_ERA5
    from earth2studio.io import NetCDF4Backend
    from earth2studio.run import deterministic

    variables = variables or DEFAULT_VARIABLES
    step_h = MODELS[model_name][1]
    nsteps = int(np.ceil(hours / step_h))
    model = load_model(model_name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.nc"
        io = NetCDF4Backend(str(raw), backend_kwargs={"mode": "w"})
        deterministic(init_times, nsteps, model, ARCO_ERA5(), io,
                      output_coords=OrderedDict({"variable": np.array(variables)}),
                      device=torch.device(device))
        io.close()
        with xr.open_dataset(raw) as ds:
            sub = region.crop(ds).load()
    sub.attrs.update({"model": model_name, "initial_conditions": "ARCO ERA5", "region": region.name})
    sub.to_netcdf(out_path)
    return out_path


def fetch_truth(valid_times, region: Region, out_path: str | Path, variables: list[str] | None = None) -> Path:
    """Download ERA5 analysis fields at valid times from ARCO ERA5 and save the region (time, lat, lon)."""
    from earth2studio.data import ARCO_ERA5

    variables = variables or DEFAULT_VARIABLES
    da = ARCO_ERA5()(list(valid_times), variables)  # (time, variable, lat, lon)
    da = region.crop(da)
    ds = da.to_dataset(dim="variable").load()
    ds.attrs.update({"source": "ARCO ERA5 (Google Cloud)", "region": region.name})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    return out_path
