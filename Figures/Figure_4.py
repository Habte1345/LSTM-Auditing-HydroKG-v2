"""
Figure 4 (v2): why and where the HydroKG feedback works, and how robust the setup is.
Content not shown in Figures 3 and 4. All changes are paired within each seed
(Delta = experiment - Baseline of the same seed); lines and points show the mean over the
five seeds, shading and whiskers the standard deviation over seeds.

(a) Median Delta KGE by quintile of the training-period violation rate that drives the
    sampling weights (s_b, mean rate of R0-R3 at epoch 27).
(b) Median Delta V_b by the same quintiles.
(c) Median change in the KGE components |r-1|, |alpha-1|, |beta-1|.
(d) Change in the median absolute error of five hydrologic signatures.
(e) Sensitivity of R1 (dashed) and R3 (solid) to the low-flow floor, by aridity class.
(f) Epoch-mean training loss of the Baseline for each seed.
"""
import re
import sys
import pickle
from pathlib import Path

repo_root = Path("/bighome/hdagne1/LSTM-Auditing-HydroKG-v2")
EXP_DIR = repo_root / "results" / "hydrokg_v2" / "temporal"
LOG_DIR = repo_root / "logs"
LOG_PREFIX = "baseline_58304285"                            # SLURM array of the Baseline training
SEED_TASK = {658666: 0, 111: 1, 444: 4, 666: 6, 777: 7}     # array task of each seed
SEEDS = [658666, 111, 444, 666, 777]
CONFIG = repo_root / "configs" / "experiments" / "temporal.yaml"
SAVE_PATH = repo_root / "figures" / "Figure_5.png"
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from hydrokg.config import load_config
from hydrokg.metrics.skill import skill_table
from hydrokg.metrics.burden import burden_table

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
mpl.rcParams['figure.dpi'] = 150
plt.rcParams.update({"axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})

cfg = load_config(CONFIG)
EXPERIMENTS = {"HydroKG-S": "hydrokg_s_test", "HydroKG-F": "hydrokg_f_test", "HydroKG": "hydrokg_test"}
COLORS = {"HydroKG-S": "#ff8c00", "HydroKG-F": "#3b78d8", "HydroKG": "darkgreen"}
MARKERS = {"HydroKG-S": "o", "HydroKG-F": "s", "HydroKG": "D"}
N_Q = 5                                                     # quintiles of s_b
aridity_order = ["humid", "sub_humid", "semi_arid", "arid"]
aridity_display = {"humid": "Humid", "sub_humid": "Sub-humid", "semi_arid": "Semi-arid", "arid": "Arid"}
jet = mpl.colormaps["jet"]
aridity_colors = {c: jet(f) for c, f in {"humid": 0.18, "sub_humid": 0.38, "semi_arid": 0.62, "arid": 0.85}.items()}


def _load(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


# ============================================================
# 1. DATA
# ============================================================
quint_rows, comp_rows, sig_rows = [], [], []
SIGNATURES = {"err_runoff_ratio": "Runoff ratio", "err_fdc_mid_slope": "FDC mid-slope",
              "err_baseflow_index": "Baseflow index", "err_q5": "Low flow (Q5)",
              "err_zero_flow_freq": "Zero-flow frequency"}

for s in SEEDS:
    d = EXP_DIR / f"seed_{s}"
    # training-period violation rate that sets the sampling weights (epoch-27 audit)
    train = _load(d / "audits" / "baseline_train_feedback.p")["table"]
    s_b = train[[f"rate_R{r}" for r in range(4)]].mean(axis=1, skipna=True)
    q = pd.qcut(s_b.rank(method="first"), N_Q, labels=range(1, N_Q + 1))

    base_pred = _load(d / "predictions" / "baseline_test.p")
    base_skill = skill_table(base_pred)
    base_vb = burden_table(_load(d / "audits" / "baseline_test.p")["table"], cfg)["V_b"]
    for name, label in EXPERIMENTS.items():
        exp_skill = skill_table(_load(d / "predictions" / f"{label}.p")).reindex(base_skill.index)
        exp_vb = burden_table(_load(d / "audits" / f"{label}.p")["table"], cfg)["V_b"].reindex(base_vb.index)
        frame = pd.DataFrame({"dKGE": exp_skill["kge"] - base_skill["kge"], "dV_b": exp_vb - base_vb,
                              "q": q}).dropna()
        for qi, g in frame.groupby("q", observed=True):
            quint_rows.append({"seed": s, "experiment": name, "q": int(qi),
                               "dKGE": g["dKGE"].median(), "dV_b": g["dV_b"].median()})
        comp_rows.append({"seed": s, "experiment": name,
                          **{c: ((exp_skill[c] - 1).abs() - (base_skill[c] - 1).abs()).median()
                             for c in ("r", "alpha", "beta")}})
        t = pd.read_csv(d / "analysis" / f"{label}_signature_abs_error_change.csv", index_col=0).reindex(list(SIGNATURES))
        change = 100 * (t["model_median_abs_error"] - t["baseline_median_abs_error"]) / t["baseline_median_abs_error"]
        change[t["baseline_median_abs_error"] == 0] = np.nan
        sig_rows += [{"seed": s, "experiment": name, "signature": k, "change": v} for k, v in change.items()]

quint = pd.DataFrame(quint_rows).groupby(["experiment", "q"])[["dKGE", "dV_b"]].agg(["mean", "std"])
comp = pd.DataFrame(comp_rows).groupby("experiment")[["r", "alpha", "beta"]].agg(["mean", "std"])
sig = pd.DataFrame(sig_rows).groupby(["experiment", "signature"])["change"].agg(["mean", "std"])

floor = pd.concat([pd.read_csv(EXP_DIR / f"seed_{s}" / "analysis" / "low_flow_floor_sensitivity.csv", index_col=0)
                   for s in SEEDS])
floor = floor.groupby(["aridity_class", "fraction_of_mean"])[["rate_R1", "rate_R3"]].mean().reset_index()

pat = re.compile(r"Epoch (\d+):.*?\|\s*(\d+)/\d+.*?Loss: ([0-9.eE+-]+)")
loss = {}
for s in SEEDS:
    text = ""
    for ext in ("out", "err"):
        f = LOG_DIR / f"{LOG_PREFIX}_{SEED_TASK[s]}.{ext}"
        if f.exists():
            text += f.read_text(errors="ignore").replace("\r", "\n")
    batches = {(int(e), int(i)): float(v) for e, i, v in pat.findall(text)}
    if batches:
        loss[s] = pd.Series(batches).groupby(level=0).mean()
loss = pd.DataFrame(loss)

print("Median change by quintile of training-period violation rate (mean, std over seeds):\n",
      quint.round(4).to_string())
print("\nKGE components (median change):\n", comp.round(4).to_string())
print("\nSignature error change (%):\n", sig.round(1).to_string())

# ============================================================
# 2. FIGURE
# ============================================================
plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12, "axes.linewidth": 1.2})
jet = mpl.colormaps["jet"]
COLORS = {"HydroKG-S": jet(0.85), "HydroKG-F": jet(0.15), "HydroKG": jet(0.55)}    # red-orange, blue, green-yellow
LW, MS = 2.8, 8                                                                     # line width, marker size

fig = plt.figure(figsize=(16, 10), dpi=200)
gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)
names = list(EXPERIMENTS)


def style(ax, letter, zero=True):
    ax.set_title(f"({letter})", loc="left", fontsize=16, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=1.2, length=5)
    if zero:
        ax.axhline(0, color="black", lw=1.2, zorder=1)


# --- (a, b) change by quintile of the training-period violation rate ---
for letter, col, ylabel, pos in [("a", "dKGE", r"Median $\Delta$KGE", gs[0, 0]),
                                 ("b", "dV_b", r"Median $\Delta V_b$", gs[0, 1])]:
    ax = fig.add_subplot(pos)
    for n in names:
        m = quint.loc[n, (col, "mean")]
        sd = quint.loc[n, (col, "std")].fillna(0)
        ax.fill_between(m.index, m - sd, m + sd, color=COLORS[n], alpha=0.18, lw=0)
        ax.plot(m.index, m, "-", marker=MARKERS[n], color=COLORS[n], lw=LW, ms=MS,
                markeredgecolor="black", markeredgewidth=0.6)
    ax.set_xticks(range(1, N_Q + 1))
    # ax.set_xticklabels(["Lowest", "2", "3", "4", "Highest"])
    ax.set_xlabel("Training-period violation rate")
    ax.set_ylabel(ylabel)
    style(ax, letter)

# --- (c) KGE components ---
ax = fig.add_subplot(gs[0, 2])
comps = [("r", r"$|r-1|$"), ("alpha", r"$|\alpha-1|$"), ("beta", r"$|\beta-1|$")]
xc = np.arange(len(comps))
for n in names:
    m = np.array([comp.loc[n, (c, "mean")] for c, _ in comps])
    sd = np.nan_to_num(np.array([comp.loc[n, (c, "std")] for c, _ in comps]))
    ax.fill_between(xc, m - sd, m + sd, color=COLORS[n], alpha=0.18, lw=0)
    ax.plot(xc, m, "-", marker=MARKERS[n], color=COLORS[n], lw=LW, ms=MS,
            markeredgecolor="black", markeredgewidth=0.6)
ax.set_xticks(xc)
ax.set_xticklabels([lab for _, lab in comps], fontsize=14)
ax.set_xlabel("KGE component")
ax.set_ylabel("Median change")
style(ax, "c")

# --- (d) hydrologic signatures ---
ax = fig.add_subplot(gs[1, 0])
sig_names = list(SIGNATURES)
xs = np.arange(len(sig_names))
for n in names:
    m = np.array([sig.loc[(n, g), "mean"] for g in sig_names])
    sd = np.nan_to_num(np.array([sig.loc[(n, g), "std"] for g in sig_names]))
    ax.fill_between(xs, m - sd, m + sd, color=COLORS[n], alpha=0.18, lw=0)
    ax.plot(xs, m, "-", marker=MARKERS[n], color=COLORS[n], lw=LW, ms=MS,
            markeredgecolor="black", markeredgewidth=0.6)
ax.set_xticks(xs)
ax.set_xticklabels(list(SIGNATURES.values()), rotation=25, ha="right", rotation_mode="anchor")
ax.set_ylabel("Change in median error (%)")
style(ax, "d")

# --- (e) low-flow floor sensitivity ---
ax = fig.add_subplot(gs[1, 1])
for c in aridity_order:
    g = floor[floor["aridity_class"] == c].sort_values("fraction_of_mean")
    if g.empty:
        continue
    ax.plot(g["fraction_of_mean"], g["rate_R3"], "-o", color=aridity_colors[c], ms=MS - 1, lw=LW,
            markeredgecolor="black", markeredgewidth=0.6)
    ax.plot(g["fraction_of_mean"], g["rate_R1"], "--s", color=aridity_colors[c], ms=MS - 1, lw=LW,
            markeredgecolor="black", markeredgewidth=0.6)
ax.axvline(cfg["rules"]["low_flow_floor"]["fraction_of_mean"], color="0.3", lw=1.2, ls=":")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel(r"Low-flow floor (fraction of mean $Q_{obs}$)")
ax.set_ylabel("Mean violation rate")
ax.legend(handles=[Line2D([0], [0], color=aridity_colors[c], lw=4, label=aridity_display[c]) for c in aridity_order],
          frameon=False, ncol=2, loc="lower left")
style(ax, "e", zero=False)

# --- (f) training loss per seed ---
ax = fig.add_subplot(gs[1, 2])
diverged = [s for s in loss.columns if (loss[s].pct_change() > 0.10).any()]
converged = [s for s in loss.columns if s not in diverged]
jet = mpl.colormaps["jet"]
seed_colors = {s: jet(f) for s, f in zip(converged, [0.05, 0.25, 0.40])}            # blues and cyan
seed_colors.update({s: jet(f) for s, f in zip(diverged, [0.95, 0.75])})            # red and orange
conv_ls = ["-", "--", ":", "-."]
for s, ls in zip(converged, conv_ls):
    ax.plot(loss.index, loss[s], ls=ls, lw=LW, color=seed_colors[s], label=f"Seed {s}")
for s in diverged:
    ax.plot(loss.index, loss[s], "-", lw=LW, color=seed_colors[s], label=f"Seed {s}")
for e in cfg["lstm"]["lr_schedule"]:
    ax.axvline(int(e), color="0.5", lw=1.0, ls=":")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss")
ax.legend(frameon=False, loc="upper right", fontsize=11)
style(ax, "f", zero=False)

# inset: converged seeds only, epochs 11-30
ins = ax.inset_axes([0.6, 0.17, 0.4, 0.3])
late = loss.loc[loss.index >= 11]
for s, ls in zip(converged, conv_ls):
    ins.plot(late.index, late[s], ls=ls, lw=LW - 0.8, color=seed_colors[s])
lo, hi = late[converged].min().min(), late[converged].max().max()
pad = 0.05 * (hi - lo)
ins.set_ylim(lo - pad, hi + pad)
ins.set_xlim(11, loss.index.max())
ins.tick_params(labelsize=9)
ins.set_title("Converged seeds", fontsize=9, pad=2)
ins.patch.set_alpha(0.9)
print("Diverged runs:", diverged, " converged runs:", converged)

# one legend for the experiments
fig.legend(handles=[Line2D([0], [0], color=COLORS[n], marker=MARKERS[n], lw=LW, ms=MS,
                           markeredgecolor="black", markeredgewidth=0.6, label=n) for n in names],
           loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=len(names), frameon=False, fontsize=14)

# SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(SAVE_PATH, dpi=600, bbox_inches="tight")
plt.show()