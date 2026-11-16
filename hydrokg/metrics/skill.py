"""Skill metrics on days where both observation and simulation are finite."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _valid(obs, sim):
    obs = np.asarray(obs, dtype=float).ravel()
    sim = np.asarray(sim, dtype=float).ravel()
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def kge_components(obs, sim) -> dict:
    """KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2),
    r = Pearson correlation, alpha = sigma_s/sigma_o, beta = mu_s/mu_o (Gupta et al., 2009)."""
    o, s = _valid(obs, sim)
    nan = {"r": np.nan, "alpha": np.nan, "beta": np.nan, "kge": np.nan, "n": len(o)}
    if len(o) < 2 or np.std(o) == 0 or np.mean(o) == 0 or np.std(s) == 0:
        return nan
    r = float(np.corrcoef(o, s)[0, 1])
    alpha = float(np.std(s) / np.std(o))
    beta = float(np.mean(s) / np.mean(o))
    kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {"r": r, "alpha": alpha, "beta": beta, "kge": kge, "n": len(o)}


def kge(obs, sim) -> float:
    return kge_components(obs, sim)["kge"]


def nse(obs, sim) -> float:
    o, s = _valid(obs, sim)
    if len(o) < 2 or np.var(o) == 0:
        return float("nan")
    return float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))


def skill_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for basin, df in preds.items():
        comp = kge_components(df["qobs"], df["qsim"])
        comp["nse"] = nse(df["qobs"], df["qsim"])
        comp["basin_id"] = basin
        rows.append(comp)
    return pd.DataFrame(rows).set_index("basin_id")
