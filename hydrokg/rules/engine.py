"""Vectorized evaluation of the seven auditing rules for one basin.

Notation: Q_s = simulated, Q_o = observed discharge (mm/day, missing = NaN), P = precipitation,
q_f = basin low-flow floor, d = day, w = window. All thresholds come from the configuration.

R0  Negative flow          evaluable: Q_s finite                       violation: Q_s < 0
R1  Extreme ratio          evaluable: Q_o > q_f                        violation: Q_s/Q_o < a or > b
R2  Zero-flow collapse     evaluable: Q_o > q_f                        violation: Q_s < eps
R3  High relative error    evaluable: Q_o > q_f                        violation: |Q_s - Q_o|/Q_o > c
R4  Peak timing            evaluable: observed peak events             violation: |lag| > L
R5  Water limit            evaluable: water years with >= n days       violation: mean(Q_s) > mean(P)
R6  Budyko consistency     evaluable: multi-year windows               violation: EI_s > AI  (energy)
                                                                         or 0 <= EI_s and |EI_s - Fu(AI; omega)| > delta
    with EI_s = 1 - mean(Q_s)/mean(P), AI = PET/mean(P). EI_s < 0 is a water-limit breach and is
    recorded for R5, not for R6, so the two rules never count the same failure.

The KG engine (hydrokg.kg.shacl_engine) implements the same definitions in SPARQL + SHACL;
tests/test_engines.py asserts identical results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from hydrokg.rules.context import BasinContext, fu_budyko
from hydrokg.rules.registry import DAILY_RULES, RULE_IDS


def water_year(index: pd.DatetimeIndex) -> np.ndarray:
    return np.where(index.month >= 10, index.year + 1, index.year)


@dataclass
class AuditResult:
    basin_id: str
    counts: dict = field(default_factory=dict)
    evaluable: dict = field(default_factory=dict)
    applicable: dict = field(default_factory=dict)
    daily_flags: Optional[pd.DataFrame] = None     # bool columns R0..R3, NaN-free
    daily_evaluable: Optional[pd.DataFrame] = None
    events: Optional[pd.DataFrame] = None
    r5_windows: Optional[pd.DataFrame] = None
    r6_windows: Optional[pd.DataFrame] = None
    exclusive_counts: dict = field(default_factory=dict)

    def rate(self, rule_id: str) -> float:
        n = self.evaluable.get(rule_id, 0)
        return self.counts.get(rule_id, 0) / n if n > 0 else float("nan")

    def rates(self) -> dict:
        return {r: self.rate(r) for r in RULE_IDS}

    def summary(self) -> dict:
        row = {"basin_id": self.basin_id}
        for r in RULE_IDS:
            row[f"n_{r}"] = self.counts.get(r, 0)
            row[f"eval_{r}"] = self.evaluable.get(r, 0)
            row[f"rate_{r}"] = self.rate(r)
            row[f"applicable_{r}"] = self.applicable.get(r, False)
        for r in DAILY_RULES:
            row[f"excl_{r}"] = self.exclusive_counts.get(r, 0)
        return row


# ---------------------------------------------------------------------------------------
# Daily rules
# ---------------------------------------------------------------------------------------

def daily_rule_arrays(qobs: np.ndarray, qsim: np.ndarray, q_floor: np.ndarray | float,
                      rules_cfg: dict) -> tuple[dict, dict]:
    """Elementwise flags and evaluable masks for R0-R3 (arrays of equal length).
    `q_floor` may be a scalar or a per-element array (training batches mix basins)."""
    qobs = np.asarray(qobs, dtype=float)
    qsim = np.asarray(qsim, dtype=float)
    q_floor = np.broadcast_to(np.asarray(q_floor, dtype=float), qsim.shape)
    sim_ok = np.isfinite(qsim)
    obs_ok = np.isfinite(qobs) & np.isfinite(q_floor) & (qobs > np.where(np.isfinite(q_floor), q_floor, np.inf))
    ref = sim_ok & obs_ok
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ref, qsim / np.where(ref, qobs, 1.0), np.nan)
        rel = np.where(ref, np.abs(qsim - qobs) / np.where(ref, qobs, 1.0), np.nan)
    a, b = rules_cfg["R1"]["low_ratio"], rules_cfg["R1"]["high_ratio"]
    flags = {
        "R0": sim_ok & (qsim < 0.0),
        "R1": ref & ((ratio < a) | (ratio > b)),
        "R2": ref & (qsim < rules_cfg["R2"]["sim_zero"]),
        "R3": ref & (rel > rules_cfg["R3"]["max_relative_error"]),
    }
    evaluable = {"R0": sim_ok, "R1": ref, "R2": ref, "R3": ref}
    return flags, evaluable


def exclusive_attribution(flags: pd.DataFrame, precedence: list[str]) -> pd.Series:
    """Assign each violating day to the first rule in `precedence` that it violates
    (root-cause attribution; a negative Q_s that also trips R1/R3 counts once, as R0)."""
    assigned = pd.Series("", index=flags.index, dtype=object)
    for rule_id in precedence:
        mask = flags[rule_id].to_numpy() & (assigned.to_numpy() == "")
        assigned[mask] = rule_id
    return assigned


# ---------------------------------------------------------------------------------------
# R4: peak timing
# ---------------------------------------------------------------------------------------

def peak_events(qobs: pd.Series, qsim: pd.Series, ctx: BasinContext, r4_cfg: dict) -> pd.DataFrame:
    """Observed peaks: local maxima of Q_o with height >= h_b separated by >= D days.
    For each peak t, lag = argmax_{|k|<=W} Q_s(t+k); violation if |lag| > L. Events whose
    search window leaves the series or contains missing Q_s, or whose Q_o(t) is missing, are
    not evaluable."""
    cols = ["t_obs", "t_sim", "q_obs_peak", "q_sim_peak", "lag_days", "violation"]
    if not np.isfinite(ctx.event_height):
        return pd.DataFrame(columns=cols)
    qo = qobs.to_numpy(dtype=float)
    qs = qsim.to_numpy(dtype=float)
    detect = np.where(np.isfinite(qo), qo, -np.inf)
    # -inf is not accepted by find_peaks' prominence logic; use the minimum finite value
    finite_min = np.nanmin(qo) if np.isfinite(qo).any() else 0.0
    detect = np.where(np.isfinite(detect), detect, finite_min)
    peaks, _ = find_peaks(detect, height=ctx.event_height, distance=r4_cfg["min_separation_days"])
    w, max_lag = r4_cfg["search_window_days"], r4_cfg["max_lag_days"]
    rows = []
    for t in peaks:
        if not np.isfinite(qo[t]) or t - w < 0 or t + w >= len(qs):
            continue
        window = qs[t - w:t + w + 1]
        if not np.isfinite(window).all():
            continue
        k = int(np.argmax(window)) - w
        rows.append((qobs.index[t], qobs.index[t + k], qo[t], qs[t + k], k, abs(k) > max_lag))
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------------------
# R5 / R6: water-balance windows
# ---------------------------------------------------------------------------------------

def r5_windows(qsim: pd.Series, p: pd.Series, r5_cfg: dict) -> pd.DataFrame:
    cols = ["water_year", "start", "end", "n_days", "q_sim_mean", "p_mean", "violation"]
    frame = pd.DataFrame({"qsim": qsim, "p": p}).dropna()
    if frame.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for wy, g in frame.groupby(water_year(frame.index)):
        if len(g) < r5_cfg["min_days_per_window"]:
            continue
        qm, pm = float(g["qsim"].mean()), float(g["p"].mean())
        rows.append((int(wy), g.index.min(), g.index.max(), len(g), qm, pm, qm > pm))
    return pd.DataFrame(rows, columns=cols)


def r6_windows(qsim: pd.Series, p: pd.Series, ctx: BasinContext, r6_cfg: dict) -> pd.DataFrame:
    cols = ["start_wy", "end_wy", "n_days", "q_sim_mean", "p_mean", "ai", "ei_sim", "ei_fu",
            "deviation", "water_limit_breach", "energy_violation", "curve_violation", "violation"]
    frame = pd.DataFrame({"qsim": qsim, "p": p}).dropna()
    if frame.empty or not np.isfinite(ctx.pet_mean):
        return pd.DataFrame(columns=cols)
    wy = water_year(frame.index)
    years = np.unique(wy)
    span, step = r6_cfg["window_years"], r6_cfg["step_years"]
    rows = []
    for i in range(0, len(years) - span + 1, step):
        sel = years[i:i + span]
        if sel[-1] - sel[0] != span - 1:          # require consecutive water years
            continue
        g = frame[np.isin(wy, sel)]
        if len(g) < r6_cfg["min_days_per_window"]:
            continue
        pm, qm = float(g["p"].mean()), float(g["qsim"].mean())
        if pm <= 0:
            continue
        ai = ctx.pet_mean / pm
        ei = 1.0 - qm / pm
        ei_fu = float(fu_budyko(ai, ctx.omega)) if ctx.r6_curve_applicable else float("nan")
        dev = ei - ei_fu if np.isfinite(ei_fu) else float("nan")
        water_breach = ei < 0.0
        energy = bool(r6_cfg["energy_limit"] and ei > ai)
        curve = bool(ctx.r6_curve_applicable and not water_breach
                     and abs(dev) > r6_cfg["max_budyko_deviation"])
        rows.append((int(sel[0]), int(sel[-1]), len(g), qm, pm, ai, ei, ei_fu, dev,
                     water_breach, energy, curve, energy or curve))
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------------------
# Basin audit
# ---------------------------------------------------------------------------------------

def audit_basin(df: pd.DataFrame, ctx: BasinContext, cfg: dict,
                rules: Optional[Iterable[str]] = None) -> AuditResult:
    """Evaluate `rules` (default: all seven) on one basin.

    df : DataFrame indexed by date with qsim, qobs (missing = NaN) and optionally p.
    """
    rules = list(rules) if rules is not None else list(RULE_IDS)
    rcfg = cfg["rules"]
    res = AuditResult(basin_id=ctx.basin_id)
    has_p = "p" in df.columns
    qobs = df["qobs"] if "qobs" in df.columns else pd.Series(np.nan, index=df.index)

    daily = [r for r in rules if r in DAILY_RULES]
    if daily:
        flags, evaluable = daily_rule_arrays(qobs.to_numpy(), df["qsim"].to_numpy(), ctx.q_floor, rcfg)
        res.daily_flags = pd.DataFrame({r: flags[r] for r in DAILY_RULES}, index=df.index)
        res.daily_evaluable = pd.DataFrame({r: evaluable[r] for r in DAILY_RULES}, index=df.index)
        for r in daily:
            res.applicable[r] = True if r == "R0" else bool(np.isfinite(ctx.q_floor))
            res.counts[r] = int(flags[r].sum())
            res.evaluable[r] = int(evaluable[r].sum())
        attributed = exclusive_attribution(res.daily_flags, cfg["attribution_precedence"])
        res.exclusive_counts = {r: int((attributed == r).sum()) for r in DAILY_RULES}

    if "R4" in rules:
        res.applicable["R4"] = bool(np.isfinite(ctx.event_height))
        res.events = peak_events(qobs, df["qsim"], ctx, rcfg["R4"])
        res.counts["R4"] = int(res.events["violation"].sum()) if len(res.events) else 0
        res.evaluable["R4"] = int(len(res.events))

    if "R5" in rules:
        res.applicable["R5"] = bool(has_p and ctx.r5_applicable)
        res.r5_windows = r5_windows(df["qsim"], df["p"], rcfg["R5"]) if has_p else None
        if res.applicable["R5"] and res.r5_windows is not None:
            res.counts["R5"] = int(res.r5_windows["violation"].sum())
            res.evaluable["R5"] = int(len(res.r5_windows))
        else:
            res.counts["R5"], res.evaluable["R5"] = 0, 0

    if "R6" in rules:
        res.applicable["R6"] = bool(has_p and np.isfinite(ctx.pet_mean))
        res.r6_windows = r6_windows(df["qsim"], df["p"], ctx, rcfg["R6"]) if has_p else None
        if res.applicable["R6"] and res.r6_windows is not None:
            res.counts["R6"] = int(res.r6_windows["violation"].sum())
            res.evaluable["R6"] = int(len(res.r6_windows))
        else:
            res.counts["R6"], res.evaluable["R6"] = 0, 0
    return res


def audit_observations(df: pd.DataFrame, ctx: BasinContext, cfg: dict) -> AuditResult:
    """Rule validity check: apply the observation-free rules to OBSERVED discharge.
    A well-posed physical rule should rarely flag observations; the rate obtained here is
    the rule's false-alarm rate and is reported alongside every audit."""
    obs_as_sim = pd.DataFrame({"qsim": df["qobs"], "qobs": np.nan}, index=df.index)
    if "p" in df:
        obs_as_sim["p"] = df["p"]
    return audit_basin(obs_as_sim, ctx, cfg, rules=["R0", "R5", "R6"])


def online_batch_counts(basin_ids: np.ndarray, qobs: np.ndarray, qsim: np.ndarray,
                        floors: dict, rules_cfg: dict) -> pd.DataFrame:
    """Per-basin violation and evaluable counts of R0-R3 for one training batch (physical units).
    Returns a DataFrame indexed by basin with columns n_R*, eval_R*."""
    basin_ids = np.asarray(basin_ids)
    q_floor = np.array([floors.get(b, np.nan) for b in basin_ids], dtype=float)
    flags, evaluable = daily_rule_arrays(qobs, qsim, q_floor, rules_cfg)
    data = {"basin_id": basin_ids}
    for r in DAILY_RULES:
        data[f"n_{r}"] = flags[r].astype(int)
        data[f"eval_{r}"] = evaluable[r].astype(int)
    return pd.DataFrame(data).groupby("basin_id").sum()
