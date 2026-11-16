"""Feedback signals derived from audits: curriculum sampling weights and the violation
profile feature. Both are computed only from audits of the TRAINING period; any attempt to
build them from an audit that touches the test period raises.

Curriculum weight of basin b (online rules R0-R3, rates from the latest epoch):
    s_b = mean_r rho_{b,r}
    w_b = max(s_b, f * max_b' s_b')^(1/T)          (floor f, temperature T)
Samples are drawn with probability proportional to the weight of their basin.

Violation profile of basin b: the vector (rho_{b,r})_r over the selected rules, with
non-applicable rules set to 0, appended as static attributes (z-scored by the LSTM data
pipeline like every other attribute).
"""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from hydrokg.rules.registry import ONLINE_RULES, OBSERVATION_FREE, RULE_IDS, rules_in_family

PROFILE_PREFIX = "violation_rate_"


class LeakageError(RuntimeError):
    pass


def require_training_period(period: tuple, train_period: tuple, what: str) -> None:
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if start < pd.Timestamp(train_period[0]) or end > pd.Timestamp(train_period[1]):
        raise LeakageError(f"{what} must be derived from the training period {train_period}, got {period}.")


def curriculum_weights(online_rates: pd.DataFrame, basins: list[str], temperature: float,
                       floor: float) -> pd.Series:
    """Basin weights (mean 1) from per-basin online rule rates."""
    rates = online_rates.reindex(basins)[[r for r in ONLINE_RULES if r in online_rates.columns]]
    s = rates.mean(axis=1, skipna=True).fillna(0.0).to_numpy(dtype=float)
    smax = s.max() if len(s) else 0.0
    if smax <= 0:
        return pd.Series(1.0, index=basins)
    w = np.power(np.maximum(s, floor * smax), 1.0 / temperature)
    return pd.Series(w / w.mean(), index=basins)


def profile_rules(mode: str) -> list[str]:
    return list(RULE_IDS) if mode == "all" else rules_in_family(OBSERVATION_FREE)


def profile_matrix(rates: pd.DataFrame, basins: list[str], rules: list[str]) -> pd.DataFrame:
    prof = rates.reindex(index=basins, columns=rules).astype(float).fillna(0.0)
    prof.columns = [f"{PROFILE_PREFIX}{r}" for r in rules]
    return prof


def write_profile_db(source_db: str | Path, target_db: str | Path, profile: pd.DataFrame) -> Path:
    """Copy the run's attributes.db and add/overwrite the profile columns. Basins without a
    profile row get 0. Constant columns receive a 1e-6 ramp so the z-score is defined."""
    target_db = Path(target_db)
    target_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_db, target_db)
    with closing(sqlite3.connect(target_db)) as conn, conn:
        table = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col="gauge_id")
        for col in profile.columns:
            values = profile[col].reindex(table.index).fillna(0.0)
            if values.std() == 0:
                values = values + np.arange(len(values)) * 1e-6
            table[col] = values
        table.to_sql("basin_attributes", conn, if_exists="replace", index=True, index_label="gauge_id")
    return target_db
