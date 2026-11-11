"""Analysis tables for one seed and aggregation over seeds. Each table answers a specific
reviewer request (see docs/REVIEW_MAP.md)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from hydrokg.metrics import stats
from hydrokg.metrics.burden import (burden_table, class_shares_by_group, cooccurrence, rate_matrix,
                                    rule_share_of_index)
from hydrokg.metrics.signatures import signature_table
from hydrokg.metrics.skill import skill_table
from hydrokg.rules.context import BasinContext
from hydrokg.rules.engine import audit_basin
from hydrokg.rules.registry import DAILY_RULES, RULE_IDS

INDICES = ["V_b", "V_phys", "V_err", "V_equal_rules", "V_equal_classes"]


def _load(path: Path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def model_frame(audit: dict, preds: dict, cfg: dict) -> pd.DataFrame:
    """Per-basin skill, burden indices and stratification of one model."""
    table = audit["table"]
    frame = skill_table(preds).join(burden_table(table, cfg))
    for col in ("aridity_class", "landcover_class", "aridity_index", "snow_dominated", "intermittent"):
        frame[col] = table[col]
    return frame


def skill_trust(frame: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """R1.3: correlation of KGE with each index, partial on aridity, and within-bin spread."""
    rows = []
    for idx in INDICES:
        s = stats.spearman(frame["kge"], frame[idx])
        p = stats.partial_spearman(frame["kge"], frame[idx], frame[["aridity_index"]])
        rows.append({"index": idx, "n": s["n"], "spearman_rho": s["rho"], "spearman_p": s["p"],
                     "variance_shared": s["rho"] ** 2 if np.isfinite(s["rho"]) else np.nan,
                     "partial_rho_given_aridity": p["rho"], "partial_p": p["p"]})
    for cls, sub in frame.groupby("aridity_class"):
        s = stats.spearman(sub["kge"], sub["V_b"])
        rows.append({"index": f"V_b | aridity={cls}", "n": s["n"], "spearman_rho": s["rho"], "spearman_p": s["p"]})
    bins = stats.dispersion_within_bins(frame["kge"], frame["V_b"], cfg["experiment"]["kge_bins"])
    return {"skill_trust_correlation": pd.DataFrame(rows), "skill_trust_dispersion_by_kge_bin": bins}


def comparison(base: pd.DataFrame, other: pd.DataFrame, label: str) -> dict[str, pd.DataFrame]:
    """R2.4 quadrants, paired tests, KGE components and stratified bootstrap."""
    d = pd.DataFrame(index=base.index)
    d["d_kge"] = other["kge"] - base["kge"]
    for idx in INDICES:
        d[f"d_{idx}"] = base[idx] - other[idx]          # positive = fewer violations
    for comp in ("r", "alpha", "beta"):
        d[f"abs_err_{comp}_base"] = (base[comp] - 1).abs()
        d[f"abs_err_{comp}_other"] = (other[comp] - 1).abs()
    d["aridity_class"] = base["aridity_class"]
    d["landcover_class"] = base["landcover_class"]

    quad = {idx: stats.quadrant_table(d["d_kge"], d[f"d_{idx}"]) for idx in ("V_b", "V_phys", "V_err")}
    tests = [{"metric": "kge", **stats.paired_test(base["kge"], other["kge"])}]
    for idx in INDICES:
        tests.append({"metric": idx, **stats.paired_test(base[idx], other[idx])})
    for comp in ("r", "alpha", "beta"):
        t = stats.paired_test(d[f"abs_err_{comp}_base"], d[f"abs_err_{comp}_other"])
        tests.append({"metric": f"|{comp}-1|", **t})
    summary = pd.DataFrame(tests)
    summary["share_kge_improved"] = float((d["d_kge"] > 0).mean())
    boot = pd.concat({
        "d_kge": stats.grouped_bootstrap(d["d_kge"], d["aridity_class"]),
        "d_V_b": stats.grouped_bootstrap(d["d_V_b"], d["aridity_class"]),
        "d_V_phys": stats.grouped_bootstrap(d["d_V_phys"], d["aridity_class"]),
        "d_V_err": stats.grouped_bootstrap(d["d_V_err"], d["aridity_class"]),
    }, axis=0)
    out = {f"{label}_basin_deltas": d, f"{label}_paired_tests": summary, f"{label}_bootstrap_by_aridity": boot}
    for idx, q in quad.items():
        out[f"{label}_quadrants_{idx}"] = q
    return out


def floor_sensitivity(preds: dict, contexts: dict[str, BasinContext], cfg: dict,
                      fractions=(0.0, 0.01, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    """R2.2: mean R1-R3 rates and evaluable days by aridity class as the low-flow floor
    q_floor = max(q_abs, f * mean_train(Q_obs)) varies with f."""
    q_abs = cfg["rules"]["low_flow_floor"]["absolute"]
    rows = []
    for f in fractions:
        for b, df in preds.items():
            ctx = contexts.get(b)
            if ctx is None or not np.isfinite(ctx.q_mean_train):
                continue
            c = BasinContext(basin_id=b, q_floor=max(q_abs, f * ctx.q_mean_train))
            res = audit_basin(df, c, cfg, rules=["R1", "R2", "R3"])
            rows.append({"fraction_of_mean": f, "basin_id": b, "aridity_class": ctx.aridity_class,
                         **{f"rate_{r}": res.rate(r) for r in ("R1", "R2", "R3")},
                         "eval_R1": res.evaluable["R1"]})
    d = pd.DataFrame(rows)
    return (d.groupby(["fraction_of_mean", "aridity_class"])[["rate_R1", "rate_R2", "rate_R3", "eval_R1"]]
            .mean().reset_index())


def analyze_seed(out_dir: str | Path, cfg: dict) -> None:
    out_dir = Path(out_dir)
    adir = out_dir / "analysis"
    adir.mkdir(exist_ok=True)
    audits = {p.stem: _load(p) for p in sorted((out_dir / "audits").glob("*_test.p"))}
    preds = {p.stem: _load(p) for p in sorted((out_dir / "predictions").glob("*_test.p"))}
    models = sorted(set(audits) & set(preds))
    frames = {m: model_frame(audits[m], preds[m], cfg) for m in models}
    tables: dict[str, pd.DataFrame] = {}

    # per-model summaries (rates, indices, skill)
    summary = []
    for m, f in frames.items():
        row = {"model": m, "n_basins": len(f), "kge_median": f["kge"].median(), "nse_median": f["nse"].median()}
        for col in [f"rate_{r}" for r in RULE_IDS] + INDICES + [c for c in f.columns if c.startswith("class_")]:
            row[f"{col}_mean"] = f[col].mean()
            row[f"{col}_median"] = f[col].median()
        summary.append(row)
    tables["model_summary"] = pd.DataFrame(summary)

    base = frames.get("baseline_test")
    if base is not None:
        rates = rate_matrix(audits["baseline_test"]["table"])
        tables["index_rule_shares"] = pd.DataFrame({w: rule_share_of_index(rates, w)
                                                    for w in ("equal_rules", "equal_classes")})
        tables["class_shares_by_aridity"] = class_shares_by_group(rates, base["aridity_class"])
        tables["class_shares_by_landcover"] = class_shares_by_group(rates, base["landcover_class"])
        flags = [r.daily_flags for r in audits["baseline_test"]["results"] if r.daily_flags is not None]
        if flags:
            tables["daily_rule_cooccurrence"] = cooccurrence(flags)
        tables.update(skill_trust(base, cfg))
        sig = signature_table(preds["baseline_test"], cfg["stratification"]["zero_flow"])
        tables["signatures_baseline"] = sig
        for m in models:
            if m == "baseline_test":
                continue
            tables.update(comparison(base, frames[m], f"{m}_vs_baseline"))
            sig_m = signature_table(preds[m], cfg["stratification"]["zero_flow"])
            err_cols = [c for c in sig_m.columns if c.startswith("err_")]
            tables[f"{m}_signature_abs_error_change"] = pd.DataFrame({
                "baseline_median_abs_error": sig[err_cols].abs().median(),
                "model_median_abs_error": sig_m[err_cols].abs().median(),
            })
    control = frames.get("control_test")
    if control is not None:
        for m in models:
            if m.startswith(("curriculum", "profile")) and "+" not in m:
                tables.update(comparison(control, frames[m], f"{m}_vs_control"))
    for m in models:
        if "+" in m:
            raw = m.split("+")[0] + "_test"
            if raw in frames:
                tables.update(comparison(frames[raw], frames[m], f"{m}_vs_uncorrected"))
    contexts_path = out_dir / "contexts.p"
    if contexts_path.exists() and "baseline_test" in preds:
        _, ctx_eval, _ = _load(contexts_path)
        tables["low_flow_floor_sensitivity"] = floor_sensitivity(preds["baseline_test"], ctx_eval, cfg)
    validity = out_dir / "audits" / "baseline_test_observation_validity.csv"
    if validity.exists():
        v = pd.read_csv(validity, index_col=0)
        tables["rule_validity_on_observations"] = v.describe().T

    for name, table in tables.items():
        table.to_csv(adir / f"{name}.csv")


def aggregate_seeds(experiment_dir: str | Path) -> pd.DataFrame:
    """Mean and standard deviation over seeds of every per-model summary statistic."""
    experiment_dir = Path(experiment_dir)
    frames = []
    for f in sorted(experiment_dir.glob("seed_*/analysis/model_summary.csv")):
        d = pd.read_csv(f, index_col=0)
        d["seed"] = f.parts[-3]
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    allf = pd.concat(frames)
    numeric = allf.select_dtypes("number").columns
    agg = allf.groupby("model")[list(numeric)].agg(["mean", "std", "count"])
    agg.to_csv(experiment_dir / "seed_aggregate_model_summary.csv")
    return agg
