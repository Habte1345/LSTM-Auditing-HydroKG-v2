import numpy as np
import pandas as pd
import pytest

from hydrokg.config import load_config
from hydrokg.rules.context import BasinContext, build_context, fit_fu_omega, fu_budyko
from hydrokg.rules.engine import audit_basin, audit_observations, exclusive_attribution, online_batch_counts

CFG = load_config()


def frame(qobs, qsim, start="2000-10-01", p=None):
    idx = pd.date_range(start, periods=len(qsim))
    df = pd.DataFrame({"qobs": qobs, "qsim": qsim}, index=idx)
    if p is not None:
        df["p"] = p
    return df


def ctx(**kw):
    base = dict(basin_id="00000001", q_floor=0.1, event_height=np.nan, pet_mean=np.nan)
    base.update(kw)
    return BasinContext(**base)


def test_daily_rules_counts_and_evaluable():
    qobs = [1.0, 1.0, 1.0, 0.05, np.nan, 2.0]
    qsim = [-0.1, 0.1, 6.0, 5.0, 3.0, 2.5]
    res = audit_basin(frame(qobs, qsim), ctx(), CFG, rules=["R0", "R1", "R2", "R3"])
    assert res.evaluable == {"R0": 6, "R1": 4, "R2": 4, "R3": 4}   # Q_obs > 0.1 on days 0,1,2,5
    assert res.counts["R0"] == 1                                     # day 0
    assert res.counts["R1"] == 3                                     # ratios -0.1, 0.1, 6.0
    assert res.counts["R2"] == 1                                     # -0.1 < 0.01
    assert res.counts["R3"] == 2                                     # |-1.1|/1 and 5/1 > 1
    # root cause: day 0 (R0,R1,R2,R3) -> R0; day 1 (R1) -> R1; day 2 (R1,R3) -> R1
    assert res.exclusive_counts == {"R0": 1, "R1": 2, "R2": 0, "R3": 0}


def test_floor_not_identifiable_disables_reference_rules():
    res = audit_basin(frame([1.0, 1.0], [0.5, -1.0]), ctx(q_floor=np.nan), CFG, rules=["R0", "R1", "R2", "R3"])
    assert res.evaluable["R1"] == 0 and not res.applicable["R1"]
    assert res.counts["R0"] == 1


def test_exclusive_attribution_precedence():
    flags = pd.DataFrame({"R0": [True, False, False], "R1": [True, True, True],
                          "R2": [False, True, False], "R3": [True, True, True]})
    assert exclusive_attribution(flags, ["R0", "R2", "R1", "R3"]).tolist() == ["R0", "R2", "R1"]


def test_peak_timing():
    n = 200
    qobs = np.full(n, 0.5)
    qsim = np.full(n, 0.5)
    qobs[50], qsim[51] = 10.0, 9.0       # lag 1 -> ok
    qobs[120], qsim[126] = 12.0, 8.0     # lag 6 -> violation
    res = audit_basin(frame(qobs, qsim), ctx(event_height=5.0), CFG, rules=["R4"])
    assert res.evaluable["R4"] == 2 and res.counts["R4"] == 1
    assert sorted(res.events["lag_days"].tolist()) == [1, 6]


def test_water_limit_water_years():
    idx = pd.date_range("2000-10-01", "2002-09-30")
    p = np.full(len(idx), 2.0)
    qsim = np.where(idx < pd.Timestamp("2001-10-01"), 1.0, 2.5)    # WY2001 ok, WY2002 Q > P
    df = pd.DataFrame({"qsim": qsim, "qobs": 1.0, "p": p}, index=idx)
    res = audit_basin(df, ctx(), CFG, rules=["R5"])
    assert res.evaluable["R5"] == 2 and res.counts["R5"] == 1
    res_na = audit_basin(df, ctx(r5_applicable=False), CFG, rules=["R5"])
    assert res_na.evaluable["R5"] == 0 and not res_na.applicable["R5"]


def test_budyko_rule_energy_curve_and_water_breach():
    idx = pd.date_range("1995-10-01", "2000-09-30")      # exactly 5 water years -> 1 window
    p = np.full(len(idx), 2.0)
    pet = 1.0                                            # AI = 0.5
    omega = 2.6
    ei_fu = float(fu_budyko(0.5, omega))
    def run(q):
        df = pd.DataFrame({"qsim": np.full(len(idx), q), "qobs": np.nan, "p": p}, index=idx)
        return audit_basin(df, ctx(pet_mean=pet, omega=omega, r6_curve_applicable=True), CFG, rules=["R5", "R6"])
    ok = run(2.0 * (1 - ei_fu))                          # on the curve
    assert ok.evaluable["R6"] == 1 and ok.counts["R6"] == 0
    energy = run(0.6)                                    # EI = 0.7 > AI = 0.5
    assert energy.counts["R6"] == 1 and bool(energy.r6_windows["energy_violation"].iloc[0])
    breach = run(2.5)                                    # EI < 0: water limit, belongs to R5
    assert breach.counts["R6"] == 0 and bool(breach.r6_windows["water_limit_breach"].iloc[0])
    assert breach.counts["R5"] == 5


def test_fu_fit_roundtrip():
    for ai, omega in [(0.4, 2.0), (1.0, 3.5), (2.5, 1.8)]:
        ei = float(fu_budyko(ai, omega))
        assert fit_fu_omega(ai, ei, (1.0001, 20.0)) == pytest.approx(omega, rel=1e-6)
    assert np.isnan(fit_fu_omega(0.5, 0.9, (1.0001, 20.0)))   # EI above min(1, AI)


def test_context_from_training_data():
    idx = pd.date_range("1990-10-01", "1995-09-30")
    rng = np.random.default_rng(0)
    q = rng.gamma(1.0, 1.0, len(idx))
    df = pd.DataFrame({"qobs": q, "p": q * 2.5}, index=idx)
    attrs = pd.Series({"aridity": 0.8, "pet_mean": 2.0, "frac_snow": 0.4, "dom_land_cover": "Grasslands", "huc": "05"})
    c = build_context("x", attrs, df, CFG)
    assert c.aridity_class == "sub_humid" and c.landcover_class == "GL" and c.snow_dominated
    assert c.q_floor == pytest.approx(max(0.01, 0.05 * q.mean()))
    assert c.runoff_ratio_train == pytest.approx(0.4) and c.r5_applicable
    assert c.omega_source == "training_observations"
    blind = build_context("x", attrs, df, CFG, use_observations=False, omega_fallback=2.0)
    assert np.isnan(blind.q_floor) and blind.omega == 2.0 and blind.omega_source == "regional_fallback"


def test_observation_validity_uses_observed_flow():
    idx = pd.date_range("2000-10-01", "2001-09-30")
    df = pd.DataFrame({"qobs": np.full(len(idx), 3.0), "qsim": 1.0, "p": 2.0}, index=idx)
    res = audit_observations(df, ctx(), CFG)
    assert res.counts["R5"] == 1      # observed Q > P flags the rule's false-alarm rate


def test_online_batch_counts_per_basin():
    out = online_batch_counts(np.array(["a", "a", "b"]), np.array([1.0, 1.0, 1.0]),
                              np.array([-1.0, 1.0, 9.0]), {"a": 0.1, "b": 0.1}, CFG["rules"])
    assert out.loc["a", "n_R0"] == 1 and out.loc["b", "n_R1"] == 1 and out.loc["a", "eval_R1"] == 2
