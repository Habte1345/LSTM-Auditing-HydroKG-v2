"""Statistics for the skill-trust analysis and the enhancement comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def spearman(x: pd.Series, y: pd.Series) -> dict:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 4:
        return {"n": len(d), "rho": np.nan, "p": np.nan}
    rho, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return {"n": len(d), "rho": float(rho), "p": float(p)}


def partial_spearman(x: pd.Series, y: pd.Series, covariates: pd.DataFrame) -> dict:
    """Rank-based partial correlation: Pearson correlation of the residuals of rank(x) and
    rank(y) after linear regression on the ranks of the covariates. p-value from the t
    distribution with n - 2 - k degrees of freedom."""
    d = pd.concat([x.rename("x"), y.rename("y"), covariates], axis=1).dropna()
    k = covariates.shape[1]
    n = len(d)
    if n < k + 4:
        return {"n": n, "rho": np.nan, "p": np.nan}
    ranks = d.rank()
    z = np.column_stack([np.ones(n), ranks[covariates.columns].to_numpy()])

    def resid(v):
        beta, *_ = np.linalg.lstsq(z, v, rcond=None)
        return v - z @ beta

    rx, ry = resid(ranks["x"].to_numpy()), resid(ranks["y"].to_numpy())
    rho = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - 2 - k
    t = rho * np.sqrt(dof / max(1e-12, 1 - rho ** 2))
    p = float(2 * stats.t.sf(abs(t), dof))
    return {"n": n, "rho": rho, "p": p, "dof": dof}


def dispersion_within_bins(skill: pd.Series, burden: pd.Series, edges: list[float]) -> pd.DataFrame:
    """Spread of the burden among basins of similar skill: quantiles of V in fixed KGE bins."""
    d = pd.concat([skill.rename("kge"), burden.rename("V")], axis=1).dropna()
    d["bin"] = pd.cut(d["kge"], edges)
    g = d.groupby("bin", observed=True)["V"]
    out = pd.DataFrame({
        "n": g.size(), "V_p10": g.quantile(0.10), "V_p50": g.quantile(0.50), "V_p90": g.quantile(0.90),
    })
    out["p90_over_p10"] = out["V_p90"] / out["V_p10"].replace(0, np.nan)
    return out


def quadrant_table(d_kge: pd.Series, d_v: pd.Series, tol: float = 0.0) -> pd.DataFrame:
    """Shares of basins in each (Delta KGE, Delta V) quadrant. Delta V > 0 means fewer
    violations. |Delta| <= tol counts as unchanged."""
    d = pd.concat([d_kge.rename("dk"), d_v.rename("dv")], axis=1).dropna()

    def sign(v):
        return np.where(v > tol, "improved", np.where(v < -tol, "worse", "unchanged"))

    d["skill"], d["consistency"] = sign(d["dk"]), sign(d["dv"])
    table = pd.crosstab(d["skill"], d["consistency"], normalize=False)
    table = table.reindex(index=["improved", "unchanged", "worse"],
                          columns=["improved", "unchanged", "worse"], fill_value=0)
    share = table / len(d)
    return pd.concat({"n_basins": table, "share": share}, axis=1)


def bootstrap_mean(values, n_boot: int = 5000, ci: float = 0.95, seed: int = 0) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan}
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    a = (1 - ci) / 2
    return {"n": len(v), "mean": float(v.mean()), "lo": float(np.quantile(boots, a)),
            "hi": float(np.quantile(boots, 1 - a))}


def paired_test(before: pd.Series, after: pd.Series) -> dict:
    """Wilcoxon signed-rank test of after - before."""
    d = (after - before).dropna()
    d = d[d != 0]
    if len(d) < 10:
        return {"n": len(d), "median_diff": float(d.median()) if len(d) else np.nan, "p": np.nan}
    res = stats.wilcoxon(d)
    return {"n": len(d), "median_diff": float(d.median()), "p": float(res.pvalue)}


def grouped_bootstrap(values: pd.Series, groups: pd.Series, **kw) -> pd.DataFrame:
    d = pd.concat([values.rename("v"), groups.rename("g")], axis=1).dropna()
    return pd.DataFrame({g: bootstrap_mean(sub["v"], **kw) for g, sub in d.groupby("g")}).T


def seed_summary(frames: dict[int, pd.DataFrame], column: str) -> pd.DataFrame:
    """Per-basin mean and standard deviation of `column` across seeds."""
    stacked = pd.concat({s: f[column] for s, f in frames.items()}, axis=1)
    return pd.DataFrame({"mean": stacked.mean(axis=1), "sd": stacked.std(axis=1, ddof=1),
                         "n_seeds": stacked.notna().sum(axis=1)})
