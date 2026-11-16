"""Post-hoc correction of test-period predictions, triggered ONLY by observation-free rules.

clip    : Q_s(t) <- max(Q_s(t), 0) on R0 days (benchmark requested by Reviewer 1)
analog  : on R0 days, Q_s(t) <- max(0, med_s(b) * median_{d in A_k(b), Q_s,d(t) >= 0} Q_s,d(t) / med_s(d)),
          A_k(b) = k analogs of b (hydrokg.kg.analogs), med_s = median simulated flow
R5 projection (optional): in a water year with mean Q_s > mean P, Q_s <- Q_s * mean P / mean Q_s

No observed discharge is read in this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydrokg.rules.engine import water_year


def clip_negative(preds: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    out, log = {}, []
    for b, df in preds.items():
        df = df.copy()
        neg = df["qsim"] < 0
        log.append({"basin_id": b, "n_corrected": int(neg.sum()), "method": "clip"})
        df.loc[neg, "qsim"] = 0.0
        out[b] = df
    return out, pd.DataFrame(log)


def analog_correction(preds: dict[str, pd.DataFrame], finder, method: str, k: int) -> tuple[dict, pd.DataFrame]:
    medians = {b: max(float(df["qsim"].median()), 1e-3) for b, df in preds.items()}
    pool = list(preds)
    out, log = {}, []
    for b, df in preds.items():
        df = df.copy()
        neg = df.index[df["qsim"] < 0]
        analogs = finder.find(method, b, pool, k) if len(neg) else []
        n_analog = n_fallback = 0
        for t in neg:
            ratios = [preds[a].at[t, "qsim"] / medians[a] for a in analogs
                      if t in preds[a].index and preds[a].at[t, "qsim"] >= 0]
            if ratios:
                df.at[t, "qsim"] = max(0.0, medians[b] * float(np.median(ratios)))
                n_analog += 1
            else:
                df.at[t, "qsim"] = 0.0
                n_fallback += 1
        log.append({"basin_id": b, "n_corrected": len(neg), "n_analog": n_analog,
                    "n_clip_fallback": n_fallback, "analogs": ";".join(analogs), "method": f"analog_{method}"})
        out[b] = df
    return out, pd.DataFrame(log)


def r5_projection(preds: dict[str, pd.DataFrame], min_days: int) -> tuple[dict, pd.DataFrame]:
    out, log = {}, []
    for b, df in preds.items():
        df = df.copy()
        if "p" not in df:
            out[b] = df
            continue
        wy = water_year(df.index)
        n = 0
        for year in np.unique(wy):
            m = wy == year
            g = df.loc[m, ["qsim", "p"]].dropna()
            if len(g) < min_days:
                continue
            qm, pm = g["qsim"].mean(), g["p"].mean()
            if qm > pm > 0:
                df.loc[m, "qsim"] = df.loc[m, "qsim"] * pm / qm
                n += 1
        log.append({"basin_id": b, "n_years_projected": n})
        out[b] = df
    return out, pd.DataFrame(log)


def apply_correction(preds: dict[str, pd.DataFrame], cfg: dict, finder=None) -> tuple[dict, pd.DataFrame]:
    ccfg = cfg["correction"]
    logs = []
    if ccfg["method"] == "clip":
        preds, log = clip_negative(preds)
        logs.append(log)
    elif ccfg["method"] == "analog":
        if finder is None:
            raise ValueError("analog correction needs an AnalogFinder")
        preds, log = analog_correction(preds, finder, cfg["analogs"]["method"], cfg["analogs"]["k"])
        logs.append(log)
    if ccfg["r5_projection"]:
        preds, log = r5_projection(preds, cfg["rules"]["R5"]["min_days_per_window"])
        logs.append(log)
    log = pd.concat(logs, axis=0) if logs else pd.DataFrame()
    return preds, log
