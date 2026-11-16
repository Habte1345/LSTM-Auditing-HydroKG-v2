"""Hydrologic signatures used as independent diagnostics. None of them is a rule or a
training signal, so improvement in them cannot come from optimizing against the audit.

    runoff_ratio     RR = mean(Q) / mean(P)
    evaporative_idx  EI = 1 - RR
    fdc_mid_slope    (ln Q_20 - ln Q_70) / (0.70 - 0.20), Q_p = flow exceeded p of the time (Yilmaz et al., 2008)
    baseflow_index   sum(Q_b)/sum(Q), Lyne-Hollick filter, alpha = 0.925, three passes (Ladson et al., 2013)
    zero_flow_freq   share of days with Q <= q_zero
    q5, q95          5th and 95th percentile flows
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

EPS = 1e-6


@njit(cache=True)
def _lh_pass(series: np.ndarray, alpha: float) -> np.ndarray:
    quick = np.zeros_like(series)
    for t in range(1, len(series)):
        f = alpha * quick[t - 1] + 0.5 * (1.0 + alpha) * (series[t] - series[t - 1])
        if series[t] <= 0.0:
            f = 0.0
        else:
            f = min(max(f, 0.0), series[t])
        quick[t] = f
    return series - quick


def lyne_hollick_baseflow(q: np.ndarray, alpha: float = 0.925, passes: int = 3) -> np.ndarray:
    """Baseflow from the one-parameter digital filter, alternating forward/backward passes.
    Quickflow f_t = alpha f_{t-1} + (1 + alpha)/2 (Q_t - Q_{t-1}), constrained to 0 <= f_t <= Q_t."""
    base = np.asarray(q, dtype=np.float64).copy()
    for k in range(passes):
        if k % 2 == 0:
            base = _lh_pass(base, alpha)
        else:
            base = _lh_pass(base[::-1].copy(), alpha)[::-1].copy()
    return base


def signatures(q: pd.Series, p: pd.Series | None, zero_flow: float) -> dict:
    q = q.astype(float)
    out = {}
    if p is not None:
        m = q.notna() & p.notna()
        pm = p[m].mean()
        out["runoff_ratio"] = float(q[m].mean() / pm) if pm > 0 else np.nan
        out["evaporative_idx"] = 1.0 - out["runoff_ratio"] if np.isfinite(out["runoff_ratio"]) else np.nan
    qv = q.dropna().to_numpy()
    if len(qv) < 30:
        return {**out, "fdc_mid_slope": np.nan, "baseflow_index": np.nan,
                "zero_flow_freq": np.nan, "q5": np.nan, "q95": np.nan}
    q20 = np.quantile(qv, 0.80)   # exceeded 20 % of the time
    q70 = np.quantile(qv, 0.30)   # exceeded 70 % of the time
    out["fdc_mid_slope"] = float((np.log(max(q20, EPS)) - np.log(max(q70, EPS))) / 0.5)
    qpos = np.clip(qv, 0.0, None)
    out["baseflow_index"] = float(lyne_hollick_baseflow(qpos).sum() / qpos.sum()) if qpos.sum() > 0 else np.nan
    out["zero_flow_freq"] = float((qv <= zero_flow).mean())
    out["q5"] = float(np.quantile(qv, 0.05))
    out["q95"] = float(np.quantile(qv, 0.95))
    return out


def signature_errors(df: pd.DataFrame, zero_flow: float) -> dict:
    """Signatures of Q_obs and Q_sim on the days where both exist, and their errors.
    Absolute errors for dimensionless signatures; relative errors for flow quantiles."""
    m = df["qobs"].notna() & df["qsim"].notna()
    p = df["p"][m] if "p" in df else None
    so = signatures(df["qobs"][m], p, zero_flow)
    ss = signatures(df["qsim"][m], p, zero_flow)
    out = {}
    for key in so:
        out[f"obs_{key}"], out[f"sim_{key}"] = so[key], ss[key]
        if key in ("q5", "q95"):
            out[f"err_{key}"] = (ss[key] - so[key]) / so[key] if so[key] and np.isfinite(so[key]) and so[key] > 0 else np.nan
        else:
            out[f"err_{key}"] = ss[key] - so[key]
    return out


def signature_table(preds: dict[str, pd.DataFrame], zero_flow: float) -> pd.DataFrame:
    return pd.DataFrame({b: signature_errors(df, zero_flow) for b, df in preds.items()}).T
