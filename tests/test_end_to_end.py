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
    "lstm.seq_length=30", "lstm.epochs=3", "lstm.batch_size=128", "lstm.hidden_size=8",
    "fine_tune.epochs=2", "experiment.analog_benchmark=true", "rules.R6.window_years=2", "rules.R6.min_days_per_window=650",
    "experiment.corrections=[none,clip,analog]", "experiment.arms=[hydrokg_s,hydrokg_f,hydrokg,baseline_replay]", "kg.engine=shacl", "analogs.k=2",
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
                 "hydrokg_test_vs_baseline_quadrants_V_b", "hydrokg_s_test_vs_baseline_paired_tests",
                 "analog_benchmark", "rule_validity_on_observations", "low_flow_floor_sensitivity"):
        assert (analysis / f"{name}.csv").exists(), name
    summary = pd.read_csv(analysis / "model_summary.csv")
    assert {"baseline_test", "hydrokg_s_test", "hydrokg_f_test", "hydrokg_test", "baseline+clip_test"} <= set(summary["model"])
    clipped = pd.read_csv(out / "audits" / "baseline+clip_test.csv", index_col=0)
    assert (clipped["n_R0"] == 0).all()
    if mode == "basin_holdout":
        prof = pd.read_csv(out / "arms" / "hydrokg_f" / "violation_profile.csv", index_col=0)
        assert set(prof.columns) == {"violation_rate_R0", "violation_rate_R5", "violation_rate_R6"}
    # training feedback came from the knowledge graph
    import json
    assert (out / "kg" / "feedback_rates_from_kg.csv").exists()
    meta = json.loads((out / "arms" / "hydrokg" / "finetune.json").read_text())
    assert all(h["feedback_source"] == "knowledge_graph" for h in meta["history"])
    assert meta["start_epoch"] == 1 and meta["end_epoch"] == 3 and meta["optimizer_and_rng_restored"]
    # without feedback, re-training the final epochs reproduces the baseline exactly
    import torch
    replay = torch.load(out / "arms" / "baseline_replay" / "model_finetuned.pt")
    base = torch.load(Path(exp.run_cfg["db_path"]).parent / "model_epoch3.pt")
    assert all(torch.equal(replay[k], base[k]) for k in base)
    # a resumed run reuses every finished step
    Experiment(cfg, seed=7).run()


def test_reuse_existing_baseline_without_h5(synthetic, tmp_path):
    """A reused baseline run whose train_data.h5 was not kept (gitignored) is rebuilt."""
    import shutil
    root, basin_file = synthetic
    base = OVERRIDES + [f"paths.camels_root={root}", f"paths.basin_list={basin_file}",
                        f"paths.output_dir={tmp_path / 'a'}", "experiment.arms=[hydrokg_s]",
                        "experiment.corrections=[none]", "kg.engine=vectorized"]
    first = Experiment(load_config(overrides=base), seed=3)
    first.basins_and_split()
    first.baseline()
    run_dir = Path(first.run_cfg["db_path"]).parent
    reused = tmp_path / "reused_run"
    shutil.copytree(run_dir, reused)
    shutil.rmtree(reused / "data")
    cfg = load_config(overrides=base[:-4] + [f"paths.output_dir={tmp_path / 'b'}", "experiment.arms=[hydrokg_s]",
                                             "experiment.corrections=[none]", "kg.engine=vectorized",
                                             f"experiment.reuse_baseline_run_dir={reused}"])
    exp = Experiment(cfg, seed=3)
    exp.run()
    assert (reused / "data" / "train" / "train_data.h5").exists()
    assert (exp.out / "audits" / "hydrokg_s_test.csv").exists()
