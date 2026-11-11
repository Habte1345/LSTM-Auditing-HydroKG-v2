"""Basin splits. Writes the {split: {'train': [...], 'test': [...]}} pickle format of the
LSTM code's `create_splits`, so the same file drives training and evaluation."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def temporal_split(basins: list[str]) -> dict:
    return {0: {"train": list(basins), "test": list(basins)}}


def stratified_holdout(basins: list[str], strata: pd.Series, fraction: float, seed: int) -> dict:
    """Hold out `fraction` of basins, stratified by `strata` (e.g. aridity class).
    Strata with a single basin are kept in training."""
    strata = strata.reindex(basins).fillna("unknown")
    counts = strata.value_counts()
    rare = strata[strata.map(counts) < 2].index.tolist()
    pool = [b for b in basins if b not in rare]
    train, test = train_test_split(pool, test_size=fraction, random_state=seed,
                                   stratify=strata.loc[pool].to_numpy())
    return {0: {"train": sorted(train + rare), "test": sorted(test)}}


def save_split(split: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(split, fp)


def load_split(path: str | Path) -> dict:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def describe_split(split: dict, strata: pd.Series) -> pd.DataFrame:
    s = split[0]
    rows = []
    for part in ("train", "test"):
        counts = strata.reindex(s[part]).value_counts()
        for cls, n in counts.items():
            rows.append({"part": part, "stratum": cls, "n_basins": int(n)})
    return pd.DataFrame(rows)
