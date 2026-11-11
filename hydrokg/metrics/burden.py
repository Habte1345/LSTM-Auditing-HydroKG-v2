"""Violation burden indices built from per-rule rates rho_{b,r} = N_viol(b,r) / N_eval(b,r).

Scalar index with explicit weights (rules that are not applicable or have no evaluable
opportunity are excluded and the weights renormalized):

    V_b = sum_r w_r rho_{b,r} / sum_r w_r

    equal_rules   : w_r = 1
    equal_classes : w_r = 1 / |rules of class(r) available in b|   (each class weighs 1)
    custom        : w_r from the configuration

Family indices (reported separately, never merged):
    V_phys,b = mean of rho over observation-free rules      (R0, R5, R6)
    V_err,b  = mean of rho over observation-referenced rules (R1, R2, R3, R4)

Class burdens: mean of rho over the rules of each class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydrokg.rules.registry import (CLASSES, DAILY_RULES, FAMILIES, OBSERVATION_FREE,
                                    OBSERVATION_REFERENCED, RULE_IDS, RULES, rules_in_class,
                                    rules_in_family)


def rate_matrix(audit_table: pd.DataFrame) -> pd.DataFrame:
    """Per-basin rates; NaN where a rule is not applicable or has no evaluable opportunity."""
    rates = pd.DataFrame(index=audit_table.index)
    for r in RULE_IDS:
        rate = audit_table[f"rate_{r}"].astype(float)
        applicable = audit_table.get(f"applicable_{r}", pd.Series(True, index=audit_table.index)).astype(bool)
        rates[r] = rate.where(applicable & (audit_table[f"eval_{r}"] > 0))
    return rates


def rule_weights(weighting: str, available: list[str], custom: dict | None = None) -> dict:
    if weighting == "equal_rules":
        return {r: 1.0 for r in available}
    if weighting == "equal_classes":
        weights = {}
        for cls in CLASSES:
            members = [r for r in rules_in_class(cls) if r in available]
            for r in members:
                weights[r] = 1.0 / len(members)
        return weights
    if weighting == "custom":
        return {r: float(custom[r]) for r in available}
    raise ValueError(weighting)


def weighted_burden(rates: pd.DataFrame, weighting: str, custom: dict | None = None) -> pd.Series:
    out = {}
    for basin, row in rates.iterrows():
        available = [r for r in RULE_IDS if np.isfinite(row[r])]
        if not available:
            out[basin] = np.nan
            continue
        w = rule_weights(weighting, available, custom)
        out[basin] = sum(w[r] * row[r] for r in available) / sum(w.values())
    return pd.Series(out, name=f"V_{weighting}")


def class_burdens(rates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({cls: rates[rules_in_class(cls)].mean(axis=1, skipna=True) for cls in CLASSES})


def family_burdens(rates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "V_phys": rates[rules_in_family(OBSERVATION_FREE)].mean(axis=1, skipna=True),
        "V_err": rates[rules_in_family(OBSERVATION_REFERENCED)].mean(axis=1, skipna=True),
    })


def burden_table(audit_table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """All indices for each basin: rates, class burdens, family burdens, V_b (all weightings)."""
    rates = rate_matrix(audit_table)
    out = rates.add_prefix("rate_")
    out = out.join(class_burdens(rates).add_prefix("class_"))
    out = out.join(family_burdens(rates))
    for weighting in ("equal_rules", "equal_classes"):
        out = out.join(weighted_burden(rates, weighting))
    if cfg["burden"]["weighting"] == "custom":
        out = out.join(weighted_burden(rates, "custom", cfg["burden"]["custom_weights"]))
    out["V_b"] = out[f"V_{cfg['burden']['weighting']}"]
    for r in DAILY_RULES:
        ev = audit_table[f"eval_{r}"].replace(0, np.nan)
        out[f"exclusive_rate_{r}"] = audit_table[f"excl_{r}"] / ev
    return out


def rule_share_of_index(rates: pd.DataFrame, weighting: str, custom: dict | None = None) -> pd.Series:
    """Average contribution of each rule to the scalar index across basins (sums to 1)."""
    contrib = []
    for _, row in rates.iterrows():
        available = [r for r in RULE_IDS if np.isfinite(row[r])]
        if not available:
            continue
        w = rule_weights(weighting, available, custom)
        total = sum(w[r] * row[r] for r in available)
        if total <= 0:
            continue
        contrib.append({r: w[r] * row[r] / total for r in available})
    return pd.DataFrame(contrib).reindex(columns=RULE_IDS).fillna(0.0).mean()


def cooccurrence(flag_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """C[i, j] = P(rule j violated on a day | rule i violated on that day), pooled over basins."""
    flags = pd.concat([f[DAILY_RULES] for f in flag_frames if f is not None], axis=0).astype(bool)
    mat = pd.DataFrame(index=DAILY_RULES, columns=DAILY_RULES, dtype=float)
    for i in DAILY_RULES:
        n_i = flags[i].sum()
        for j in DAILY_RULES:
            mat.loc[i, j] = (flags[i] & flags[j]).sum() / n_i if n_i else np.nan
    return mat


def class_shares_by_group(rates: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Mean class burden within each group, normalized to shares per group (one basis: rates)."""
    cb = class_burdens(rates).join(groups.rename("group"))
    means = cb.groupby("group")[CLASSES].mean()
    return means.div(means.sum(axis=1), axis=0)
