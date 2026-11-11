"""Full pipeline on a synthetic CAMELS-format data set: baseline training with the vendored
LSTM code, contexts, KG, audits (SHACL engine), all fine-tuning arms, corrections,
provenance leakage check and analysis tables. Small and slow-ish (about a minute)."""

from pathlib import Path

import pandas as pd
import pytest

from hydrokg.config import load_config
from hydrokg.kg import store
from hydrokg.kg.provenance import leakage_report
from hydrokg.pipeline.experiment import Experiment
from tests.synthetic_camels import write_synthetic_camels

OVERRIDES = [
    "periods.train=['1991-10-01','1994-09-30']", "periods.test=['1994-10-01','1996-09-30']",
    "lstm.seq_length=30", "lstm.epochs=1", "lstm.batch_size=128", "lstm.hidden_size=8",
    "fine_tune.epochs=2", "rules.R6.window_years=2", "rules.R6.min_days_per_window=650",
    "experiment.corrections=[none,clip,analog]", "kg.engine=shacl", "analogs.k=2",
]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("camels")
    basins = write_synthetic_camels(root, n_basins=8)
    basin_file = root / "basins.txt"
    basin_file.write_text("\n".join(basins) + "\n")
    return root, basin_file


@pytest.mark.parametrize("mode", ["temporal", "basin_holdout"])
def test_full_pipeline(synthetic, tmp_path, mode):
    root, basin_file = synthetic
    overrides = OVERRIDES + [f"paths.camels_root={root}", f"paths.basin_list={basin_file}",
                             f"paths.output_dir={tmp_path}", f"split.mode={mode}", "split.holdout_fraction=0.25"]
    cfg = load_config(overrides=overrides)
    exp = Experiment(cfg, seed=7)
    exp.run()
    out = exp.out
    for arm in cfg["experiment"]["arms"]:
        assert (out / "arms" / arm / "model_finetuned.pt").exists()
        assert (out / "audits" / f"{arm}_test.csv").exists()
        assert (out / "audits" / f"{arm}+analog_test.csv").exists()
    kg = store.load(out / "kg" / "hydrokg.ttl")
    assert leakage_report(kg, mode == "basin_holdout").empty
    analysis = out / "analysis"
    for name in ("model_summary", "skill_trust_correlation", "index_rule_shares",
                 "curriculum_profile_test_vs_control_quadrants_V_b", "control_test_vs_baseline_paired_tests",
                 "analog_benchmark", "rule_validity_on_observations", "low_flow_floor_sensitivity"):
        assert (analysis / f"{name}.csv").exists(), name
    summary = pd.read_csv(analysis / "model_summary.csv")
    assert {"baseline_test", "control_test", "curriculum_profile_test", "baseline+clip_test"} <= set(summary["model"])
    clipped = pd.read_csv(out / "audits" / "baseline+clip_test.csv", index_col=0)
    assert (clipped["n_R0"] == 0).all()
    if mode == "basin_holdout":
        prof = pd.read_csv(out / "arms" / "profile" / "violation_profile.csv", index_col=0)
        assert set(prof.columns) == {"violation_rate_R0", "violation_rate_R5", "violation_rate_R6"}
    # a resumed run reuses every finished step
    Experiment(cfg, seed=7).run()
