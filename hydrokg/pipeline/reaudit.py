"""Re-audit existing prediction pickles (e.g. the v1 baseline and enhanced predictions)
with the revised rules, missing-value masking and training-period contexts. No model is
run. Writes the same analysis tables as a full experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hydrokg.config import period, save_config
from hydrokg.data import camels
from hydrokg.data.predictions import attach_series, load_predictions, mask_missing, missing_report
from hydrokg.kg import store
from hydrokg.kg.graph import apply_kg_applicability, build_catchment_graph
from hydrokg.pipeline.analysis import analyze_seed
from hydrokg.pipeline.audit import audit_predictions
from hydrokg.pipeline.experiment import _pickle
from hydrokg.rules.context import build_context


def reaudit(cfg: dict, prediction_files: list[str], labels: list[str], run_dir: str, out: str) -> None:
    if len(prediction_files) != len(labels):
        raise ValueError("--predictions and --labels must have the same length")
    out = Path(out)
    (out / "audits").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    camels_root = Path(cfg["paths"]["camels_root"])
    db_path = Path(run_dir) / "attributes.db"
    train = period(cfg, "train")

    first = load_predictions(prediction_files[0])
    basins = sorted(first)
    attrs = camels.static_attributes(db_path, basins)
    frames = {b: camels.basin_period_frame(camels_root, b, train, cfg["qc"]["qobs_missing_below"]) for b in basins}
    contexts = {b: build_context(b, attrs.loc[b], frames[b], cfg, True) for b in basins}
    _pickle(out / "contexts.p", (frames, contexts, contexts))
    kg = build_catchment_graph(contexts.values(), cfg, camels.model_attributes(db_path, basins))
    apply_kg_applicability(contexts, kg)
    p_series = {b: camels.precipitation(camels_root, b) for b in basins}

    for path, label in zip(prediction_files, labels):
        raw = load_predictions(path)
        missing_report(raw, cfg["qc"]["qobs_missing_below"]).to_csv(out / "predictions" / f"{label}_missing.csv", index=False)
        preds = attach_series(mask_missing(raw, cfg["qc"]["qobs_missing_below"]), p_series, "p")
        _pickle(out / "predictions" / f"{label}_test.p", preds)
        res = audit_predictions(preds, contexts, cfg, kg, workers=cfg["experiment"]["audit_workers"],
                                keep_flags=(label == "baseline"), desc=label)
        res["table"].to_csv(out / "audits" / f"{label}_test.csv")
        if len(res["validity"]):
            res["validity"].to_csv(out / "audits" / f"{label}_test_observation_validity.csv")
        _pickle(out / "audits" / f"{label}_test.p", res)
    store.save(kg, out / "hydrokg_catchments.ttl")
    analyze_seed(out, cfg)
