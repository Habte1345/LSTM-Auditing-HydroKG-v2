"""
Figure 3 (v2): HydroKG scenarios vs Baseline, 5 seeds, test period 1995-2010.
All changes are paired within each seed: Delta = scenario - Baseline of the same seed.
Bars show the mean over seeds; error bars show the standard deviation over seeds.

(a) Median Delta KGE per scenario.
(b) Mean change in V_err (R1-R4) and V_phys (R0, R5, R6) per scenario.
(c) Mean change in each rule rate per scenario.
(d) Delta KGE of every basin, sorted (seed-mean per basin).
(e) Median Delta KGE by aridity class per scenario.
(f) Share of basins in each Delta KGE / Delta V_b quadrant per scenario.
"""
import sys
from pathlib import Path

repo_root = Path("/bighome/hdagne1/LSTM-Auditing-HydroKG-v2")
EXP_DIR = repo_root / "results" / "hydrokg_v2" / "temporal"
SEEDS = [658666, 111, 444, 666, 777]
CONFIG = repo_root / "configs" / "experiments" / "temporal.yaml"
SAVE_PATH = repo_root / "figures" / "Figure_4.png"
sys.path.insert(0, str(repo_root))

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

from hydrokg.config import load_config
from hydrokg.rules.registry import RULE_IDS
from hydrokg.pipeline.analysis import model_frame

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
mpl.rcParams['figure.dpi'] = 150
plt.rcParams.update({"axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})

cfg = load_config(CONFIG)

SCENARIOS = {"HydroKG-S": "hydrokg_s_test", "HydroKG-F": "hydrokg_f_test",
             "HydroKG": "hydrokg_test", "HydroKG (clipped)": "hydrokg+clip_test"}
COLORS = {"HydroKG-S": "#ff8c00", "HydroKG-F": "#3b78d8", "HydroKG": "darkgreen", "HydroKG (clipped)": "#8fbc8f"}
MAIN = ["HydroKG-S", "HydroKG-F", "HydroKG"]          # scenarios shown in (c)-(f)


def _load(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


# ============================================================
# 1. DATA: per-seed, per-basin changes relative to the seed's own Baseline
# ============================================================
deltas = []
for s in SEEDS:
    d = EXP_DIR / f"seed_{s}"
    frame = lambda label: model_frame(_load(d / "audits" / f"{label}.p"), _load(d / "predictions" / f"{label}.p"), cfg)
    base = frame("baseline_test")
    for name, label in SCENARIOS.items():
        f = frame(label).reindex(base.index)
        out = pd.DataFrame({"seed": s, "scenario": name, "aridity_class": base["aridity_class"],
                            "dKGE": f["kge"] - base["kge"], "dV_b": f["V_b"] - base["V_b"],
                            "dV_err": f["V_err"] - base["V_err"], "dV_phys": f["V_phys"] - base["V_phys"]},
                           index=base.index)
        for r in RULE_IDS:
            out[f"d{r}"] = f[f"rate_{r}"] - base[f"rate_{r}"]
        deltas.append(out)
D = pd.concat(deltas).rename_axis("basin_id").reset_index()
print(f"Basins per seed: {D.groupby(['seed', 'scenario']).size().unique()}")

aridity_order = [c for c in ["humid", "sub_humid", "semi_arid", "arid"] if c in set(D["aridity_class"])]
aridity_display = {"humid": "Humid", "sub_humid": "Sub-humid", "semi_arid": "Semi-arid", "arid": "Arid"}

# per-seed statistics -> mean and std over seeds
per_seed = D.groupby(["scenario", "seed"]).agg(
    dKGE_median=("dKGE", "median"), improved=("dKGE", lambda x: 100 * (x > 0).mean()),
    dV_err=("dV_err", "mean"), dV_phys=("dV_phys", "mean"), dV_b=("dV_b", "mean"),
    **{f"d{r}": (f"d{r}", "mean") for r in RULE_IDS})
stat = per_seed.groupby("scenario").agg(["mean", "std"])

# seed-mean change per basin (for the sorted curve and the paired tests)
basin_mean = D.groupby(["scenario", "basin_id"])[["dKGE", "dV_b", "dV_err", "dV_phys"]].mean()

# quadrants per seed: KGE up / down x V_b down / up
def quadrants(g):
    up, fewer = g["dKGE"] > 0, g["dV_b"] < 0
    return pd.Series({"KGE up, V_b down": (up & fewer).mean(), "KGE up, V_b up": (up & ~fewer).mean(),
                      "KGE down, V_b down": (~up & fewer).mean(), "KGE down, V_b up": (~up & ~fewer).mean()}) * 100
quad = D.groupby(["scenario", "seed"]).apply(quadrants).groupby("scenario").mean()

# median Delta KGE by aridity class per seed
by_ar = D.groupby(["scenario", "seed", "aridity_class"])["dKGE"].median().groupby(["scenario", "aridity_class"]).agg(["mean", "std"])

# ============================================================
# 2. NUMBERS FOR THE TEXT
# ============================================================
print("\nPer scenario (mean over seeds; std in second column):\n", stat.round(4).to_string())
print("\nPaired Wilcoxon test on seed-mean basin changes:")
for name in SCENARIOS:
    g = basin_mean.loc[name].dropna()
    for col in ["dKGE", "dV_err", "dV_phys"]:
        p = wilcoxon(g[col]).pvalue if (g[col] != 0).any() else np.nan
        print(f"  {name:13s} {col:8s} median = {g[col].median():+.4f}   p = {p:.1e}")
print("\nQuadrant shares (% of basins, mean over seeds):\n", quad.round(1).to_string())
print("\nMedian Delta KGE by aridity class:\n", by_ar.round(4).to_string())

# ============================================================
# 3. FIGURE
# ============================================================
fig = plt.figure(figsize=(14, 8.4), dpi=200)
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
names = list(SCENARIOS)
xs = np.arange(len(names))
tick_names = names


def style(ax, letter):
    ax.set_title(f"({letter})", loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", lw=0.8, zorder=1)


# --- (a) median Delta KGE ---
ax = fig.add_subplot(gs[0, 0])
m, sd = stat[("dKGE_median", "mean")].reindex(names), stat[("dKGE_median", "std")].reindex(names)
ax.bar(xs, m, yerr=sd, color=[COLORS[n] for n in names], edgecolor="black", capsize=4, zorder=2)
ax.set_xticks(xs)
ax.set_xticklabels(tick_names, rotation=20, ha="right", rotation_mode="anchor")
ax.set_ylabel(r"Median $\Delta$KGE")
style(ax, "a")

# --- (b) change in V_err and V_phys ---
ax = fig.add_subplot(gs[0, 1])
w = 0.38
for k, (col, hatch) in enumerate([("dV_err", ""), ("dV_phys", "///")]):
    ax.bar(xs + (k - 0.5) * w, stat[(col, "mean")].reindex(names), w, yerr=stat[(col, "std")].reindex(names),
           color=[COLORS[n] for n in names], edgecolor="black", hatch=hatch, capsize=3, zorder=2)
ax.set_xticks(xs)
ax.set_xticklabels(tick_names, rotation=20, ha="right", rotation_mode="anchor")
ax.set_ylabel(r"Mean $\Delta V$")
ax.legend(handles=[Patch(facecolor="white", edgecolor="black", label=r"$V_{err}$"),
                   Patch(facecolor="white", edgecolor="black", hatch="///", label=r"$V_{phys}$")],
          frameon=False, loc="upper right")
style(ax, "b")

# --- (c) change per rule ---
ax = fig.add_subplot(gs[0, 2])
xr = np.arange(len(RULE_IDS))
w = 0.8 / len(MAIN)
for k, n in enumerate(MAIN):
    ax.bar(xr + (k - (len(MAIN) - 1) / 2) * w, [stat.loc[n, (f"d{r}", "mean")] for r in RULE_IDS], w,
           yerr=[stat.loc[n, (f"d{r}", "std")] for r in RULE_IDS], color=COLORS[n], edgecolor="black",
           capsize=2, error_kw={"lw": 0.8}, zorder=2)
ax.set_xticks(xr)
ax.set_xticklabels(RULE_IDS)
ax.set_xlabel("Rule")
ax.set_ylabel(r"Mean $\Delta$ violation rate")
style(ax, "c")

# --- (d) sorted Delta KGE per basin ---
ax = fig.add_subplot(gs[1, 0])
for n in MAIN:
    v = np.sort(basin_mean.loc[n, "dKGE"].dropna().values)
    ax.plot(np.arange(len(v)), v, color=COLORS[n], lw=1.6, zorder=2)
ax.set_ylim(-0.1, 0.1)
ax.set_xlabel(r"Basins ranked by $\Delta$KGE")
ax.set_ylabel(r"$\Delta$KGE")
style(ax, "d")

# --- (e) median Delta KGE by aridity class ---
ax = fig.add_subplot(gs[1, 1])
xa = np.arange(len(aridity_order))
for k, n in enumerate(MAIN):
    vals = [by_ar.loc[(n, c), "mean"] for c in aridity_order]
    errs = [by_ar.loc[(n, c), "std"] for c in aridity_order]
    ax.bar(xa + (k - (len(MAIN) - 1) / 2) * w, vals, w, yerr=errs, color=COLORS[n], edgecolor="black",
           capsize=2, error_kw={"lw": 0.8}, zorder=2)
ax.set_xticks(xa)
ax.set_xticklabels([aridity_display[c] for c in aridity_order])
ax.set_xlabel("Aridity class")
ax.set_ylabel(r"Median $\Delta$KGE")
style(ax, "e")

# --- (f) quadrant shares ---
ax = fig.add_subplot(gs[1, 2])
qcols = {"KGE up, V_b down": "darkgreen", "KGE up, V_b up": "#9acd32",
         "KGE down, V_b down": "#ffb347", "KGE down, V_b up": "#d62728"}
qlabels = {"KGE up, V_b down": r"$\Delta$KGE > 0, $\Delta V_b$ < 0", "KGE up, V_b up": r"$\Delta$KGE > 0, $\Delta V_b$ > 0",
           "KGE down, V_b down": r"$\Delta$KGE < 0, $\Delta V_b$ < 0", "KGE down, V_b up": r"$\Delta$KGE < 0, $\Delta V_b$ > 0"}
bottom = np.zeros(len(MAIN))
for q, col in qcols.items():
    vals = quad.reindex(MAIN)[q].values
    ax.bar(MAIN, vals, bottom=bottom, width=0.6, color=col, edgecolor="white", linewidth=0.8, label=qlabels[q])
    bottom += vals
ax.set_ylim(0, 100)
ax.set_ylabel("Basins (%)")
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
ax.set_title("(f)", loc="left", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# one legend for the scenarios, above the panels
fig.legend(handles=[Patch(facecolor=COLORS[n], edgecolor="black", label=n) for n in names],
           loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(names), frameon=False, fontsize=11)

# SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(SAVE_PATH, dpi=600, bbox_inches="tight")
plt.show()