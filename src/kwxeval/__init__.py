"""kwxeval: evaluate open AI weather prediction models on Korean high-impact weather cases.

Forecast files are NetCDF with dims (time, lead_time, lat, lon) and one variable per field,
as written by Earth2Studio's NetCDF4 backend (cropped to the region by `kwxeval.run`).
Truth files are NetCDF with dims (time, lat, lon).
"""

__version__ = "0.1.0"

from .regions import KOREA, Region  # noqa: F401
from .verify import grid_scores  # noqa: F401
from .track import track_minimum, track_errors  # noqa: F401
