import numpy as np
import pandas as pd
import pytest
import torch

from hydrokg.config import load_config
from hydrokg.enhance.feedback import LeakageError, curriculum_weights, require_training_period
from hydrokg.enhance.finetune import warm_start
from hydrokg.lstm.model import Model
from hydrokg.metrics import stats
from hydrokg.metrics.burden import burden_table, rule_weights, weighted_burden
from hydrokg.metrics.signatures import lyne_hollick_baseflow, signatures
from hydrokg.metrics.skill import kge_components

CFG = load_config()


def test_kge_components():
    o = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    s = np.array([2.0, 4.0, 6.0, 8.0, 1.0])
    c = kge_components(o, s)
    assert c["r"] == pytest.approx(1.0) and c["alpha"] == pytest.approx(2.0) and c["beta"] == pytest.approx(2.0)
    assert c["kge"] == pytest.approx(1 - np.sqrt(2)) and c["n"] == 4


def test_equal_class_weights_balance_classes():
    w = rule_weights("equal_classes", ["R0", "R1", "R2", "R3", "R4", "R5", "R6"])
    assert w["R4"] == 1.0 and w["R0"] == 1.0
    assert w["R1"] == pytest.approx(1 / 3) and w["R5"] == pytest.approx(0.5)
    rates = pd.DataFrame({"R0": [0.1], "R1": [0.3], "R2": [0.0], "R3": [0.3], "R4": [0.6], "R5": [0.0], "R6": [np.nan]})
    v = weighted_burden(rates, "equal_classes").iloc[0]
    # classes: PI=0.1, MF=0.2, TF=0.6, BF(R5 only)=0.0  -> mean 0.225
    assert v == pytest.approx(0.225)


def test_burden_table_families():
    table = pd.DataFrame({
        **{f"rate_{r}": [x] for r, x in zip(["R0", "R1", "R2", "R3", "R4", "R5", "R6"], [0.1, 0.2, 0.0, 0.4, 0.5, 0.0, 0.2])},
        **{f"eval_{r}": [10] for r in ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]},
        **{f"applicable_{r}": [True] for r in ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]},
        **{f"excl_{r}": [1] for r in ["R0", "R1", "R2", "R3"]},
    }, index=["b"])
    t = burden_table(table, CFG)
    assert t.loc["b", "V_phys"] == pytest.approx(0.1)
    assert t.loc["b", "V_err"] == pytest.approx(0.275)


def test_quadrants_and_partial_correlation():
    q = stats.quadrant_table(pd.Series([0.1, -0.1, 0.2, 0.0]), pd.Series([0.1, 0.1, -0.1, 0.0]))
    assert q["n_basins"].loc["improved", "improved"] == 1
    assert q["n_basins"].loc["worse", "improved"] == 1
    assert q["n_basins"].loc["unchanged", "unchanged"] == 1
    rng = np.random.default_rng(0)
    z = pd.Series(rng.normal(size=400))
    x = z + 0.1 * rng.normal(size=400)
    y = z + 0.1 * rng.normal(size=400)
    assert stats.spearman(x, y)["rho"] > 0.9
    assert abs(stats.partial_spearman(x, y, z.to_frame("z"))["rho"]) < 0.25


def test_baseflow_filter_bounds():
    rng = np.random.default_rng(1)
    q = np.abs(rng.gamma(0.5, 2.0, 1000))
    b = lyne_hollick_baseflow(q)
    assert np.all(b >= -1e-12) and np.all(b <= q + 1e-12)
    s = signatures(pd.Series(q), pd.Series(q * 2), 0.001)
    assert 0 < s["baseflow_index"] < 1 and s["runoff_ratio"] == pytest.approx(0.5)


def test_curriculum_weights_and_leak_guard():
    rates = pd.DataFrame({"R0": [0.0, 0.2, np.nan], "R1": [0.0, 0.4, 0.1], "R2": 0.0, "R3": 0.0}, index=["a", "b", "c"])
    w = curriculum_weights(rates, ["a", "b", "c"], temperature=1.0, floor=0.1)
    assert w.mean() == pytest.approx(1.0) and w["b"] > w["c"] > w["a"] > 0
    require_training_period(("1981-01-01", "1990-01-01"), ("1980-10-01", "1995-09-30"), "x")
    with pytest.raises(LeakageError):
        require_training_period(("1995-10-01", "2010-09-30"), ("1980-10-01", "1995-09-30"), "x")


def test_warm_start_is_functionally_identical():
    torch.manual_seed(0)
    base = Model(input_size_dyn=32, hidden_size=16, dropout=0.0, concat_static=True)
    ext = Model(input_size_dyn=39, hidden_size=16, dropout=0.0, concat_static=True)
    report = warm_start(ext, base.state_dict())
    assert report["extended"] == ["lstm.weight_ih"]
    x = torch.randn(4, 20, 32)
    x_ext = torch.cat([x, torch.randn(4, 20, 7)], dim=-1)
    base.eval(); ext.eval()
    assert torch.allclose(base(x)[0], ext(x_ext)[0], atol=1e-6)
