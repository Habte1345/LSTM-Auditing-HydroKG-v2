"""
Figure 2 (v2 results): same layout as the original figure, with the data replaced by the
five-seed ensemble of the v2 experiment (Baseline vs HydroKG, test period 1995-2010).
Ensemble = mean Q_sim of the five seeds, audited with the training-period basin contexts.
"""
import sys
from pathlib import Path

repo_root = Path("/bighome/hdagne1/LSTM-Auditing-HydroKG-v2")
sys.path.insert(0, str(repo_root))

import pickle
import tempfile

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.ticker

from hydrokg.config import load_config
from hydrokg.rules.engine import audit_basin
from hydrokg.metrics.skill import skill_table
from hydrokg.metrics.burden import burden_table

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
mpl.rcParams['figure.dpi'] = 150

# ============================================================
# 1. LOAD v2 RESULTS: 5-SEED ENSEMBLE, BASELINE AND HYDROKG
# ============================================================
EXP_DIR = repo_root / "results" / "hydrokg_v2" / "temporal"
SEEDS = [658666, 111, 444, 666, 777]
cfg = load_config(repo_root / "configs" / "experiments" / "temporal.yaml")

rule_cols = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]


def _load(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


# evaluation contexts (training-period thresholds; identical for every seed)
_, ctx_eval, _ = _load(EXP_DIR / f"seed_{SEEDS[0]}" / "contexts.p")


def ensemble_predictions(label):
    """Mean Q_sim across seeds; Q_obs and P taken from the first seed."""
    runs = [_load(EXP_DIR / f"seed_{s}" / "predictions" / f"{label}.p") for s in SEEDS]
    ens = {}
    for b in runs[0]:
        df = runs[0][b].copy()
        df["qsim"] = np.mean([r[b]["qsim"].values for r in runs], axis=0)
        ens[b] = df
    return ens


def load_stage(label):
    preds = ensemble_predictions(label)
    rows = []
    for b, df in preds.items():
        if b in ctx_eval:
            row = audit_basin(df, ctx_eval[b], cfg).summary()
            row["aridity_class"] = ctx_eval[b].aridity_class
            row["landcover_class"] = ctx_eval[b].landcover_class
            rows.append(row)
    table = pd.DataFrame(rows).set_index("basin_id")
    burden = burden_table(table, cfg)
    out = pd.DataFrame({"kge": skill_table(preds)["kge"].reindex(table.index),
                        "violation_burden": burden["V_b"],
                        "aridity_class": table["aridity_class"],
                        "landcover_class": table["landcover_class"]})
    out = out.join(burden[[f"rate_{r}" for r in rule_cols]].rename(columns=lambda c: c[5:]))
    out.index = out.index.astype(str).str.zfill(8)
    return out


baseline = load_stage("baseline_test")
enhanced = load_stage("hydrokg_test")
common = baseline.index.intersection(enhanced.index)
print(f"Paired basins: {len(common)}")
baseline, enhanced = baseline.loc[common], enhanced.loc[common]

delta_kge = enhanced["kge"] - baseline["kge"]
delta_burden = enhanced["violation_burden"] - baseline["violation_burden"]   # negative = fewer violations
aridity_class = baseline["aridity_class"]
aridity_order = [c for c in ["humid", "sub_humid", "semi_arid", "arid"] if c in aridity_class.unique()]


jet = mpl.colormaps["jet"]

model_colors = {"Base": 'magenta', "HydroKG": 'darkgreen'}
aridity_display = {"humid": "Humid", "sub_humid": "Sub-humid", "semi_arid": "Semi-arid", "arid": "Arid"}
aridity_jet_frac = {"humid": 0.18, "sub_humid": 0.38, "semi_arid": 0.62, "arid": 0.85}
aridity_colors = {c: jet(aridity_jet_frac[c]) for c in aridity_order}

# ============================================================
# 2. REAL BASIN GEOMETRY
# ============================================================
camels_url = ("https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/"
              "data/contents/camels_attributes_v2.0.feather")
tmp = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
tmp.write(requests.get(camels_url).content)
basins_geo = gpd.read_feather(tmp.name).reset_index(drop=False)
if basins_geo.crs is None:
    basins_geo = basins_geo.set_crs("EPSG:4326")
basins_geo["gauge_id"] = basins_geo["gauge_id"].astype(str).str.zfill(8)
basins_geo = basins_geo.set_index("gauge_id")

land_cover_mapping = {
    "Croplands": "CL/NVM", "cropland/natural vegetation mosaic": "CL/NVM",
    "Deciduous Broadleaf Forest": "DBF", "Evergreen Needleleaf Forest": "EF",
    "Evergreen Broadleaf Forest": "EF", "Mixed Forests": "MF",
    "Grasslands": "GL", "Savannas": "WS + SL", "Woody Savannas": "WS + SL",
    "Closed Shrublands": "WS + SL", "Open Shrublands": "WS + SL",
}
veg_order = ["CL/NVM", "DBF", "EF", "MF", "GL", "WS + SL"]
landcover_grouped_full = basins_geo["dom_land_cover"].astype(str).str.strip().map(land_cover_mapping)
landcover_grouped = landcover_grouped_full.reindex(common)
landcover_order = veg_order
print("Land-cover basins matched to the 6 mapped classes:", landcover_grouped.notna().sum(), "of", len(common))

states_url = ("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
              "master/data/geojson/us-states.json")
states_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
states_tmp.write(requests.get(states_url).content)
states = gpd.read_file(states_tmp.name).to_crs(5070)
states = states[~states["name"].isin(["Alaska", "Hawaii", "Puerto Rico"])]

merged_geo = basins_geo.loc[basins_geo.index.intersection(common)].to_crs(5070)
merged_geo["geometry"] = merged_geo.geometry.centroid
merged_geo["delta_kge"] = delta_kge.reindex(merged_geo.index)
merged_geo["delta_burden"] = delta_burden.reindex(merged_geo.index)
print(f"Basins with real geometry matched: {len(merged_geo)}")

# colour limits of the two maps (v2 changes are smaller than v1)
DVB_LIM = 0.02
DKGE_LIM = 0.10

# ============================================================
# 3. FIGURE
# ============================================================
fig = plt.figure(figsize=(14, 8), dpi=200)
gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

from scipy.stats import spearmanr, rankdata

# --- (a) ---
ax = fig.add_subplot(gs[0, 0])
kge_edges = np.arange(-0.1, 1.0001, 0.1)                     # KGE bins of width 0.1
MIN_BASINS = 5                                               # skip bins with fewer basins
curves = {}
for name, df in [("Base", baseline), ("HydroKG", enhanced)]:
    g = df.groupby(pd.cut(baseline["kge"], kge_edges), observed=True)["violation_burden"]
    stats = pd.DataFrame({"q50": g.median(), "n": g.size()})
    curves[name] = stats[stats["n"] >= MIN_BASINS]
for name, z in [("Base", 2), ("HydroKG", 3)]:
    st = curves[name]
    x = np.array([iv.mid for iv in st.index])
    ax.plot(x, st["q50"], "-o", color=model_colors[name], lw=2.2, ms=5, label=name, zorder=z)
ax.set_xlim(-0.05, 0.90)
ax.set_xlabel("KGE (Base)")
ax.set_ylabel("Median $V_b$")
ax.set_title("(a)", loc="left", fontsize=10.5)
ax.legend(frameon=1, fontsize=10, loc="lower left", ncols=1)
ax.spines[["top", "right"]].set_visible(False)

# Spearman correlation between KGE and V_b of the baseline, and partial correlation given aridity
def _ai(b):
    key = b if b in ctx_eval else b.lstrip("0")
    return ctx_eval[key].aridity_index if key in ctx_eval else np.nan
sp = pd.DataFrame({"kge": baseline["kge"], "vb": baseline["violation_burden"],
                   "ai": [_ai(b) for b in baseline.index]}).dropna()
rho, _ = spearmanr(sp["kge"], sp["vb"])
rk = {c: rankdata(sp[c]) for c in sp.columns}
resid = lambda y: y - np.polyval(np.polyfit(rk["ai"], y, 1), rk["ai"])
prho, _ = spearmanr(resid(rk["kge"]), resid(rk["vb"]))
print(f"Spearman rho = {rho:.2f}; partial rho given aridity = {prho:.2f}; n = {len(sp)}")
ax.text(0.03, 0.30, rf"$\rho$ = {rho:.2f}" + "\n" + rf"$\rho_{{AI}}$ = {prho:.2f}",
        transform=ax.transAxes, fontsize=10, va="bottom", ha="left")

# inset: paired change in V_b per KGE bin (HydroKG - Base, same basins)
ins = ax.inset_axes([0.5, 0.6, 0.42, 0.38])
d = delta_burden.groupby(pd.cut(baseline["kge"], kge_edges), observed=True)
dstats = pd.DataFrame({"q25": d.quantile(0.25), "q50": d.median(), "q75": d.quantile(0.75), "n": d.size()})
dstats = dstats[dstats["n"] >= MIN_BASINS]
xd = np.array([iv.mid for iv in dstats.index])
ins.fill_between(xd, dstats["q25"], dstats["q75"], color='blue', alpha=0.25, lw=0)
ins.plot(xd, dstats["q50"], "-o", color='red', lw=1.6, ms=3)
ins.axhline(0, color="black", lw=0.8)
ins.set_xlim(ax.get_xlim())
ins.tick_params(labelsize=7)
ins.set_ylabel(r"$\Delta V_b$", fontsize=8, labelpad=1)
ins.patch.set_alpha(0.9)

from matplotlib.colors import BoundaryNorm
NBINS = 8                                          # number of colour steps

# --- (b) ---
ax = fig.add_subplot(gs[0, 1])
levels_b = np.linspace(-DVB_LIM, DVB_LIM, NBINS + 1)
norm_b = BoundaryNorm(levels_b, plt.get_cmap("gist_rainbow").N, extend="both")
states.boundary.plot(ax=ax, edgecolor="0.4", linewidth=0.5, zorder=1)
sc = ax.scatter(merged_geo.geometry.x, merged_geo.geometry.y, c=merged_geo["delta_burden"],
                cmap="gist_rainbow", norm=norm_b, s=20, edgecolor="k", linewidth=0.1, zorder=2)
ax.set_aspect("auto")
ax.set_title("(b)", loc="left", fontsize=12)
ax.set_axis_off()
pos_b = ax.get_position()
cax_b = fig.add_axes([pos_b.x1 + 0.005, pos_b.y0 + pos_b.height * 0.05, 0.008, pos_b.height * 0.90])
cb_b = fig.colorbar(sc, cax=cax_b, orientation="vertical", extend="both", ticks=levels_b[::2],
                    label=r"$\Delta V_b$ (HydroKG $-$ Base)")
cb_b.set_ticklabels([f"{t:g}" for t in levels_b[::2]])

# --- (c) ---
ax = fig.add_subplot(gs[0, 2])
levels_c = np.linspace(-DKGE_LIM, DKGE_LIM, NBINS + 1)
norm_c = BoundaryNorm(levels_c, plt.get_cmap("gist_rainbow").N, extend="both")
states.boundary.plot(ax=ax, edgecolor="0.4", linewidth=0.5, zorder=1)
sc = ax.scatter(merged_geo.geometry.x, merged_geo.geometry.y, c=merged_geo["delta_kge"],
                cmap="gist_rainbow", norm=norm_c, s=20, edgecolor="k", linewidth=0.1, zorder=2)
ax.set_aspect("auto")
ax.set_title("(c)", loc="left", fontsize=12)
ax.set_axis_off()
pos_c = ax.get_position()
cax_c = fig.add_axes([pos_c.x1 + 0.005, pos_c.y0 + pos_c.height * 0.05, 0.008, pos_c.height * 0.90])
cb_c = fig.colorbar(sc, cax=cax_c, orientation="vertical", extend="both", ticks=levels_c[::2],
                    label=r"$\Delta$KGE (HydroKG $-$ Base)")
cb_c.set_ticklabels([f"{t:g}" for t in levels_c[::2]])

# --- (d) ---
ax = fig.add_subplot(gs[1, 0])
for c in aridity_order:
    mask = aridity_class == c
    ax.scatter(baseline.loc[mask, "kge"], enhanced.loc[mask, "kge"], s=20,
               color=aridity_colors[c], alpha=0.9, edgecolors="none", label=aridity_display[c])
lims = [0, 0.9]
ax.plot(lims, lims, color="black", linewidth=1, linestyle="--", zorder=1)
ax.set_xlabel("Base LSTM KGE")
ax.set_ylabel("HydroKG LSTM KGE")
ax.set_xlim(0, 0.98)
ax.set_ylim(0, 0.98)
ax.set_title("(d)", loc="left", fontsize=10.5)
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# --- (e) ---
ax = fig.add_subplot(gs[1, 1])
for c in aridity_order:
    mask = aridity_class == c
    ax.scatter(baseline.loc[mask, "violation_burden"], enhanced.loc[mask, "violation_burden"],
               s=20, color=aridity_colors[c], alpha=0.9, edgecolors="none", label=aridity_display[c])
lims_b = [0,0.5]
ax.plot(lims_b, lims_b, color="black", linewidth=1, linestyle="--", zorder=1)
ax.set_xlabel("Base LSTM $V_b$")
ax.set_ylabel("HydroKG LSTM $V_b$")
ax.set_title("(e)", loc="left", fontsize=12)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.5)
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# --- (f) ---
ax = fig.add_subplot(gs[1, 2])
positions_base = np.arange(len(aridity_order)) - 0.2
positions_enh = np.arange(len(aridity_order)) + 0.2
data_base = [baseline.loc[aridity_class == c, "violation_burden"].dropna() for c in aridity_order]
data_enh = [enhanced.loc[aridity_class == c, "violation_burden"].dropna() for c in aridity_order]
bp1 = ax.boxplot(data_base, positions=positions_base, widths=0.32, patch_artist=True, showfliers=False)
bp2 = ax.boxplot(data_enh, positions=positions_enh, widths=0.32, patch_artist=True, showfliers=False)
for box in bp1["boxes"]:
    box.set(facecolor=model_colors["Base"], alpha=0.6)
for box in bp2["boxes"]:
    box.set(facecolor=model_colors["HydroKG"], alpha=0.6)
ax.set_xticks(np.arange(len(aridity_order)))
ax.set_xticklabels([aridity_display[c] for c in aridity_order], rotation=20, ha="right")
ax.set_ylabel("$V_b$")
ax.set_title("(f)", loc="left", fontsize=12)
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Base", "HydroKG"], frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

# --- (g) ---
ax = fig.add_subplot(gs[2, 0])
positions_base = np.arange(len(landcover_order)) - 0.2
positions_enh = np.arange(len(landcover_order)) + 0.2
data_base_lc = [baseline.loc[landcover_grouped == c, "violation_burden"].dropna() for c in landcover_order]
data_enh_lc = [enhanced.loc[landcover_grouped == c, "violation_burden"].dropna() for c in landcover_order]
bp3 = ax.boxplot(data_base_lc, positions=positions_base, widths=0.32, patch_artist=True, showfliers=False)
bp4 = ax.boxplot(data_enh_lc, positions=positions_enh, widths=0.32, patch_artist=True, showfliers=False)
for box in bp3["boxes"]:
    box.set(facecolor=model_colors["Base"], alpha=0.6)
for box in bp4["boxes"]:
    box.set(facecolor=model_colors["HydroKG"], alpha=0.6)
ax.set_xticks(np.arange(len(landcover_order)))
ax.set_xticklabels(landcover_order, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("$V_b$")
ax.set_title("(g)", loc="left", fontsize=12)
ax.legend([bp3["boxes"][0], bp4["boxes"][0]], ["Base", "HydroKG"], frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

# --- (h) ---
ax = fig.add_subplot(gs[2, 1])
x = np.arange(len(rule_cols))
width = 0.35
base_means = [baseline[r].mean() for r in rule_cols]
enh_means = [enhanced[r].mean() for r in rule_cols]
ax.bar(x - width / 2, base_means, width, color=model_colors["Base"], edgecolor="black", label="Base")
ax.bar(x + width / 2, enh_means, width, color=model_colors["HydroKG"], edgecolor="black", label="HydroKG")
for i, (b, e) in enumerate(zip(base_means, enh_means)):
    pct = (e - b) / b * 100 if b > 0 else float("nan")
    ax.text(i + width / 2, e + max(b, e) * 0.05 + 0.005, f"{pct:+.0f}%", ha="center", fontsize=8, rotation=90,
            color="darkgreen" if pct < 0 else "darkred")
ax.set_xticks(x)
ax.set_xticklabels(rule_cols)
ax.set_ylabel("Mean violation rate")
ax.set_title("(h)", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

# --- (i) ---
ax = fig.add_subplot(gs[2, 2])
sorted_delta = delta_kge.dropna().sort_values().values
n_basins_total = len(sorted_delta)
crossover_idx = int(np.searchsorted(sorted_delta, 0))

ax.plot(np.arange(n_basins_total), sorted_delta, color="black", linewidth=1.3, zorder=3)
ax.axhline(0, color="0.3", linewidth=0.8, zorder=1)
ax.axvline(crossover_idx, color="0.3", linewidth=0.8, linestyle=":", zorder=1)
ax.fill_between(np.arange(n_basins_total), sorted_delta, 0, where=(sorted_delta >= 0),
                color=model_colors["HydroKG"], alpha=0.5, zorder=2)
ax.fill_between(np.arange(n_basins_total), sorted_delta, 0, where=(sorted_delta < 0),
                color=model_colors["Base"], alpha=0.5, zorder=2)

pct_improved = 100 * (delta_kge > 0).mean()
pct_worsened = 100 * (delta_kge < 0).mean()
legend_handles = [
    Patch(facecolor=model_colors["HydroKG"], alpha=0.6, label=f"(ΔKGE > 0): {pct_improved:.0f}%"),
    Patch(facecolor=model_colors["Base"], alpha=0.6, label=f"(ΔKGE < 0): {pct_worsened:.0f}% "),
]
ax.legend(handles=legend_handles, frameon=False, fontsize=10, loc="upper left")
ax.set_xlabel("Basins (sorted low to high)")
ax.set_ylabel(r"$\Delta$KGE")
ax.set_title("(i)", loc="left", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)

# save_path = repo_root / "figures" / "Figure_2.png"
# save_path.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(save_path, dpi=600, bbox_inches="tight")
plt.show()