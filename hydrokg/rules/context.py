"""Basin context: every basin-specific quantity the rules need, computed from
TRAINING-period data and static CAMELS attributes only.

    q_floor_b       = max(q_abs, f * mean_train(Q_obs))                       (R1-R3 scoring)
    h_b             = quantile_p(Q_obs, train)                                (R4 event height)
    AI_b            = PET_b / mean_train(P)
    EI_obs,b        = 1 - mean_train(Q_obs) / mean_train(P)
    omega_b         : Fu(AI_b; omega_b) = EI_obs,b,  Fu(AI; w) = 1 + AI - (1 + AI^w)^(1/w)   (R6)
    R5 applicable   <=> mean_train(Q_obs) / mean_train(P) < rr_max           (forcing undercatch)

Quantities that need observed discharge are left undefined (NaN) when a basin's
observations must not be used (held-out basins in feedback computations).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def fu_budyko(ai: np.ndarray | float, omega: float) -> np.ndarray | float:
    """Fu (1981) evaporative index EI = ET/P as a function of the dryness index AI = PET/P."""
    ai = np.asarray(ai, dtype=float)
    return 1.0 + ai - np.power(1.0 + np.power(ai, omega), 1.0 / omega)


def fit_fu_omega(ai: float, ei: float, bounds: tuple[float, float]) -> float:
    """Solve Fu(ai; omega) = ei for omega. Returns NaN when ei lies outside the range the
    curve can reach inside `bounds` (Fu is strictly increasing in omega)."""
    if not (np.isfinite(ai) and np.isfinite(ei)) or ai <= 0:
        return float("nan")
    lo, hi = bounds
    f_lo = fu_budyko(ai, lo) - ei
    f_hi = fu_budyko(ai, hi) - ei
    if f_lo > 0 or f_hi < 0:
        return float("nan")
    return float(brentq(lambda w: fu_budyko(ai, w) - ei, lo, hi, xtol=1e-10))


def classify_aridity(aridity_index: float, bins: list[float], labels: list[str]) -> str:
    if aridity_index is None or not np.isfinite(aridity_index):
        return "unknown"
    return labels[int(np.searchsorted(bins, aridity_index, side="left"))]


@dataclass
class BasinContext:
    basin_id: str
    aridity_index: float = float("nan")
    aridity_class: str = "unknown"
    landcover_class: str = "unknown"
    huc02: str = "unknown"
    frac_snow: float = float("nan")
    pet_mean: float = float("nan")
    p_mean_train: float = float("nan")
    q_floor: float = float("nan")
    event_height: float = float("nan")
    q_std_train: float = float("nan")
    q_mean_train: float = float("nan")
    runoff_ratio_train: float = float("nan")
    zero_flow_fraction_train: float = float("nan")
    omega: float = float("nan")
    omega_source: str = "none"
    r5_applicable: bool = True
    r6_curve_applicable: bool = False
    snow_dominated: bool = False
    intermittent: bool = False
    observations_used: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_context(basin_id: str, attributes: pd.Series, train_df: Optional[pd.DataFrame],
                  cfg: dict, use_observations: bool = True,
                  omega_fallback: Optional[float] = None) -> BasinContext:
    """Build the context of one basin.

    Parameters
    ----------
    attributes : CAMELS attributes of the basin (needs aridity, pet_mean, frac_snow,
        dom_land_cover; huc optional).
    train_df : training-period DataFrame with columns qobs (missing already NaN) and p.
    use_observations : False for basins whose observed discharge must stay unseen.
    omega_fallback : Fu parameter to use when observations may not be used.
    """
    strat = cfg["stratification"]
    rules = cfg["rules"]
    ctx = BasinContext(basin_id=basin_id)
    ctx.aridity_index = float(attributes.get("aridity", np.nan))
    ctx.aridity_class = classify_aridity(ctx.aridity_index, strat["aridity_bins"], strat["aridity_labels"])
    ctx.landcover_class = strat["landcover_groups"].get(str(attributes.get("dom_land_cover", "")).strip(), "unknown")
    huc = attributes.get("huc", attributes.get("huc_02", None))
    ctx.huc02 = str(huc).zfill(2) if huc is not None and str(huc) != "nan" else "unknown"
    ctx.frac_snow = float(attributes.get("frac_snow", np.nan))
    ctx.pet_mean = float(attributes.get("pet_mean", np.nan))
    ctx.snow_dominated = bool(np.isfinite(ctx.frac_snow) and ctx.frac_snow >= strat["snow_dominated_fraction"])

    if train_df is not None and "p" in train_df:
        ctx.p_mean_train = float(train_df["p"].mean())

    if use_observations and train_df is not None:
        ctx.observations_used = True
        valid = train_df[["qobs", "p"]].dropna() if "p" in train_df else train_df[["qobs"]].dropna()
        q = valid["qobs"]
        if len(q) > 0:
            floor_cfg = rules["low_flow_floor"]
            ctx.q_floor = float(max(floor_cfg["absolute"], floor_cfg["fraction_of_mean"] * q.mean()))
            ctx.event_height = float(q.quantile(rules["R4"]["event_quantile"]))
            ctx.q_std_train = float(q.std())
            ctx.q_mean_train = float(q.mean())
            ctx.zero_flow_fraction_train = float((q <= strat["zero_flow"]).mean())
            ctx.intermittent = bool(ctx.zero_flow_fraction_train >= strat["intermittent_zero_fraction"])
            if "p" in valid and valid["p"].mean() > 0:
                p_mean = float(valid["p"].mean())
                ctx.runoff_ratio_train = float(q.mean() / p_mean)
                ctx.r5_applicable = bool(ctx.runoff_ratio_train < rules["R5"]["undercatch_runoff_ratio"])
                ai = ctx.pet_mean / p_mean
                ei = 1.0 - ctx.runoff_ratio_train
                ctx.omega = fit_fu_omega(ai, ei, tuple(rules["R6"]["omega_bounds"]))
                ctx.omega_source = "training_observations" if np.isfinite(ctx.omega) else "not_identifiable"
    elif omega_fallback is not None and np.isfinite(omega_fallback):
        ctx.omega = float(omega_fallback)
        ctx.omega_source = "regional_fallback"

    ctx.r6_curve_applicable = bool(np.isfinite(ctx.omega) and np.isfinite(ctx.pet_mean))
    return ctx


def regional_omega(contexts: dict[str, BasinContext]) -> dict[str, float]:
    """Median fitted omega per aridity class (fallback for basins without observations)."""
    frame = pd.DataFrame([{"aridity_class": c.aridity_class, "omega": c.omega}
                          for c in contexts.values() if c.omega_source == "training_observations"])
    if frame.empty:
        return {}
    return frame.groupby("aridity_class")["omega"].median().to_dict()
