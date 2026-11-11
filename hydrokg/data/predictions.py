"""Prediction files: {basin_id: DataFrame(qobs, qsim)} pickles as written by the LSTM code."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_predictions(path: str | Path) -> dict[str, pd.DataFrame]:
    with open(path, "rb") as fp:
        preds = pickle.load(fp)
    return {str(b).zfill(8): df for b, df in preds.items()}


def save_predictions(preds: dict[str, pd.DataFrame], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(preds, fp)


def mask_missing(preds: dict[str, pd.DataFrame], missing_below: float = 0.0) -> dict[str, pd.DataFrame]:
    """Set CAMELS missing-discharge codes (negative Q_obs) to NaN. Returns new frames."""
    out = {}
    for basin, df in preds.items():
        df = df.copy()
        df["qobs"] = df["qobs"].where(df["qobs"] >= missing_below)
        out[basin] = df
    return out


def missing_report(preds: dict[str, pd.DataFrame], missing_below: float = 0.0) -> pd.DataFrame:
    rows = []
    for basin, df in preds.items():
        neg = int((df["qobs"] < missing_below).sum())
        nan = int(df["qobs"].isna().sum())
        if neg or nan:
            rows.append({"basin_id": basin, "n_missing_code": neg, "n_nan": nan,
                         "min_qobs": float(df["qobs"].min())})
    return pd.DataFrame(rows, columns=["basin_id", "n_missing_code", "n_nan", "min_qobs"])


def attach_series(preds: dict[str, pd.DataFrame], series: dict[str, pd.Series], name: str) -> dict[str, pd.DataFrame]:
    out = {}
    for basin, df in preds.items():
        df = df.copy()
        s = series.get(basin)
        df[name] = s.reindex(df.index) if s is not None else np.nan
        out[basin] = df
    return out
