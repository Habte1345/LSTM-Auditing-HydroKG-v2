"""Writes a small synthetic data set in the exact CAMELS-US file layout read by the
vendored LSTM code (NLDAS-extended forcing, USGS streamflow, attribute tables)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ATTRIBUTE_FILES = {
    "clim": ["p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity", "high_prec_freq",
             "high_prec_dur", "high_prec_timing", "low_prec_freq", "low_prec_dur", "low_prec_timing"],
    "topo": ["gauge_lat", "gauge_lon", "elev_mean", "slope_mean", "area_gages2", "area_geospa_fabric"],
    "vege": ["frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff", "dom_land_cover_frac",
             "dom_land_cover", "root_depth_50", "root_depth_99"],
    "soil": ["soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
             "max_water_content", "sand_frac", "silt_frac", "clay_frac", "water_frac", "organic_frac", "other_frac"],
    "geol": ["geol_1st_class", "glim_1st_class_frac", "geol_2nd_class", "glim_2nd_class_frac",
             "carbonate_rocks_frac", "geol_porostiy", "geol_permeability"],
    "name": ["huc_02", "gauge_name"],
}
LANDCOVER = ["Mixed Forests", "Grasslands", "Croplands", "Open Shrublands", "Deciduous Broadleaf Forest",
             "Evergreen Needleleaf Forest"]


def simulate_basin(rng, dates, wetness: float, snow: float):
    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    season = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 15) / 365)
    p = rng.gamma(0.35, 7.0 * wetness, n) * (rng.random(n) < 0.45) * (0.6 + 0.8 * season)
    tmax = 18 - 14 * np.cos(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 3, n)
    tmin = tmax - 9 - rng.normal(0, 1.5, n)
    srad = 220 - 120 * np.cos(2 * np.pi * (doy - 172) / 365) + rng.normal(0, 25, n)
    vp = 900 + 600 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 80, n)
    pet = np.clip(0.15 * (tmax + 5), 0.1, None)
    soil, snowpack, q = 50.0, 0.0, np.zeros(n)
    for t in range(n):
        rain = p[t] if tmax[t] > 2 + 6 * snow else 0.0
        snowpack += p[t] - rain
        melt = min(snowpack, max(0.0, 0.8 * (tmax[t] - 3)))
        snowpack -= melt
        soil += rain + melt
        et = min(soil, pet[t] * soil / 150.0)
        soil -= et
        q[t] = 0.03 * max(soil - 20.0, 0.0) + 0.002 * soil
        soil -= q[t]
    return pd.DataFrame({"p": p, "srad": srad, "tmax": tmax, "tmin": tmin, "vp": vp, "q": q, "pet": pet}, index=dates)


def write_synthetic_camels(root: str | Path, n_basins: int = 8, start="1990-01-01", end="1996-12-31",
                           seed: int = 0) -> list[str]:
    root = Path(root)
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end)
    basins = [f"{10000000 + 1111 * i:08d}" for i in range(n_basins)]
    attrs = {k: {} for k in ATTRIBUTE_FILES}
    for i, b in enumerate(basins):
        huc = f"{(i % 3) + 1:02d}"
        wetness = 0.5 + 1.5 * (i / max(1, n_basins - 1))
        snow = 0.6 if i % 4 == 0 else 0.05
        sim = simulate_basin(rng, dates, wetness, snow)
        area = int(rng.uniform(100, 3000)) * 10 ** 6
        fdir = root / "basin_mean_forcing" / "nldas_extended" / huc
        fdir.mkdir(parents=True, exist_ok=True)
        with open(fdir / f"{b}_lump_nldas_forcing_leap.txt", "w") as fp:
            fp.write(f"{40 + i * 0.1}\n{300 + 10 * i}\n{area}\n")
            fp.write("Year Mnth Day Hr Dayl(s) PRCP(mm/day) SRAD(W/m2) SWE(mm) Tmax(C) Tmin(C) Vp(Pa)\n")
            for d, row in sim.iterrows():
                fp.write(f"{d.year} {d.month:02d} {d.day:02d} 12 43200 {row.p:.4f} {row.srad:.3f} 0 "
                         f"{row.tmax:.3f} {row.tmin:.3f} {row.vp:.3f}\n")
        qdir = root / "usgs_streamflow" / huc
        qdir.mkdir(parents=True, exist_ok=True)
        cfs = sim["q"] * area * 10 ** 6 / (28316846.592 * 86400 * 10 ** 6) * 10 ** 6
        cfs = cfs.to_numpy().copy()
        missing = rng.choice(len(cfs), size=20, replace=False)
        cfs[missing] = -999.0
        with open(qdir / f"{b}_streamflow_qc.txt", "w") as fp:
            for d, v in zip(dates, cfs):
                fp.write(f"{b} {d.year} {d.month:02d} {d.day:02d} {v:.3f} A\n")
        p_mean, pet_mean = float(sim["p"].mean()), float(sim["pet"].mean())
        values = {
            "p_mean": p_mean, "pet_mean": pet_mean, "p_seasonality": rng.normal(), "frac_snow": snow,
            "aridity": pet_mean / p_mean, "high_prec_freq": rng.uniform(5, 30), "high_prec_dur": rng.uniform(1, 2),
            "high_prec_timing": "son", "low_prec_freq": rng.uniform(200, 300), "low_prec_dur": rng.uniform(3, 10),
            "low_prec_timing": "jja", "gauge_lat": 40 + i * 0.1, "gauge_lon": -90 - i * 0.1,
            "elev_mean": 300 + 10 * i, "slope_mean": rng.uniform(1, 50), "area_gages2": area / 1e6,
            "area_geospa_fabric": area / 1e6, "frac_forest": rng.uniform(0, 1), "lai_max": rng.uniform(1, 6),
            "lai_diff": rng.uniform(0, 4), "gvf_max": rng.uniform(0.3, 0.9), "gvf_diff": rng.uniform(0.1, 0.5),
            "dom_land_cover_frac": rng.uniform(0.5, 1), "dom_land_cover": LANDCOVER[i % len(LANDCOVER)],
            "root_depth_50": 0.2, "root_depth_99": 1.5, "soil_depth_pelletier": rng.uniform(1, 50),
            "soil_depth_statsgo": rng.uniform(0.5, 1.5), "soil_porosity": rng.uniform(0.3, 0.5),
            "soil_conductivity": rng.uniform(0.5, 5), "max_water_content": rng.uniform(0.2, 0.8),
            "sand_frac": rng.uniform(10, 60), "silt_frac": rng.uniform(10, 50), "clay_frac": rng.uniform(5, 40),
            "water_frac": 0.0, "organic_frac": 0.0, "other_frac": 0.0, "geol_1st_class": "Siliciclastic",
            "glim_1st_class_frac": 0.8, "geol_2nd_class": "None", "glim_2nd_class_frac": 0.1,
            "carbonate_rocks_frac": rng.uniform(0, 0.5), "geol_porostiy": rng.uniform(0.05, 0.2),
            "geol_permeability": rng.uniform(-16, -12), "huc_02": int(huc), "gauge_name": f"Synthetic {i}",
        }
        for group, cols in ATTRIBUTE_FILES.items():
            attrs[group][b] = {c: values[c] for c in cols}
    adir = root / "camels_attributes_v2.0"
    adir.mkdir(parents=True, exist_ok=True)
    for group, rows in attrs.items():
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "gauge_id"
        df.to_csv(adir / f"camels_{group}.txt", sep=";")
    return basins
