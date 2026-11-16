"""CAMELS-US access for the audit: precipitation (the same NLDAS-extended forcing the LSTM
reads), observed discharge, and static attributes. All loaders delegate to the vendored
LSTM data utilities so the audit sees exactly the data the model sees."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from hydrokg.lstm.datautils import load_attributes, load_discharge, load_forcing

PRECIP_COLUMN = "PRCP(mm/day)"
STRATIFICATION_ATTRIBUTES = ["aridity", "pet_mean", "p_mean", "frac_snow", "dom_land_cover", "huc"]


@lru_cache(maxsize=4096)
def _forcing(camels_root: str, basin: str) -> tuple[pd.DataFrame, int]:
    return load_forcing(Path(camels_root), basin)


def precipitation(camels_root: str | Path, basin: str) -> pd.Series:
    df, _ = _forcing(str(camels_root), basin)
    return df[PRECIP_COLUMN].rename("p")


def observed_discharge(camels_root: str | Path, basin: str, missing_below: float = 0.0) -> pd.Series:
    df, area = _forcing(str(camels_root), basin)
    q = load_discharge(Path(camels_root), basin, area)
    q = q.where(q >= missing_below)        # -999 cfs codes -> NaN
    return q.rename("qobs")


def basin_period_frame(camels_root: str | Path, basin: str, period: tuple[pd.Timestamp, pd.Timestamp],
                       missing_below: float = 0.0, with_obs: bool = True) -> pd.DataFrame:
    """Daily P (and Q_obs) for one basin and period."""
    p = precipitation(camels_root, basin)
    frame = p.to_frame()
    if with_obs:
        frame = frame.join(observed_discharge(camels_root, basin, missing_below), how="left")
    idx = pd.date_range(period[0], period[1])
    return frame.reindex(idx)


def static_attributes(db_path: str | Path, basins: Iterable[str],
                      features: Optional[list[str]] = None) -> pd.DataFrame:
    """CAMELS attributes from the run's sqlite database (built by add_camels_attributes)."""
    features = features or STRATIFICATION_ATTRIBUTES
    df = load_attributes(str(db_path), list(basins), drop_lat_lon=False, keep_features=features)
    return df


def model_attributes(db_path: str | Path, basins: Iterable[str]) -> pd.DataFrame:
    """The 27 standardizable attributes the LSTM consumes (for attribute-kNN analogs)."""
    return load_attributes(str(db_path), list(basins), drop_lat_lon=True).astype(float)


def latlon(db_path: str | Path, basins: Iterable[str]) -> pd.DataFrame:
    return load_attributes(str(db_path), list(basins), drop_lat_lon=False,
                           keep_features=["gauge_lat", "gauge_lon"])
