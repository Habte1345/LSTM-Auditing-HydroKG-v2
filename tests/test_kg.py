import numpy as np
import pandas as pd
import pytest
from rdflib import RDF, URIRef

from hydrokg.config import load_config
from hydrokg.kg import store
from hydrokg.kg.graph import HK, apply_kg_applicability, build_catchment_graph, excluded_rules, load_ontology
from hydrokg.kg.provenance import Provenance, leakage_report
from hydrokg.kg.validation import ShapesGraph, shacl_audit_basin
from hydrokg.rules.context import BasinContext, fit_fu_omega
from hydrokg.rules.engine import audit_basin
from hydrokg.rules.registry import RULES

CFG = load_config()


def test_ontology_matches_registry():
    g = load_ontology(CFG)
    for rid, info in RULES.items():
        r = HK[rid]
        assert str(g.value(r, HK.inViolationClass)).endswith(info.violation_class)
        assert str(g.value(r, HK.inEvidenceFamily)).endswith(info.family)
        assert str(g.value(r, HK.temporalScale)) == info.scale
    for priority, rid in enumerate(CFG["attribution_precedence"]):
        assert int(g.value(HK[rid], HK.rootCausePriority)) == priority


def _random_basin(seed, rr_scale=2.0, pet_scale=1.2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1995-10-01", "2005-09-30")
    q = rng.gamma(0.7, 1.5, len(idx))
    q[rng.choice(len(idx), 40, replace=False)] = np.nan
    qsim = q * np.exp(rng.normal(0, 0.7, len(idx))) - 0.05
    p = np.nan_to_num(q, nan=1.0) * rr_scale + rng.gamma(0.4, 2, len(idx))
    df = pd.DataFrame({"qobs": q, "qsim": qsim, "p": p}, index=idx)
    qm, pm = np.nanmean(q), p.mean()
    rr = qm / pm
    pet = pm * pet_scale
    omega = fit_fu_omega(pet / pm, 1 - rr, (1.0001, 20.0))
    c = BasinContext(basin_id=f"{seed:08d}", q_floor=max(0.01, 0.05 * qm), event_height=np.nanquantile(q, 0.9),
                     pet_mean=pet, runoff_ratio_train=rr, omega=omega, aridity_class="humid", landcover_class="MF",
                     frac_snow=0.1)
    return df, c


@pytest.mark.parametrize("seed,rr_scale,pet_scale", [(1, 2.0, 1.2), (2, 0.8, 1.0), (3, 2.5, 0.2), (4, 1.5, 3.0)])
def test_shacl_engine_equals_vectorized_engine(seed, rr_scale, pet_scale):
    df, c = _random_basin(seed, rr_scale, pet_scale)
    kg = build_catchment_graph([c], CFG)
    apply_kg_applicability({c.basin_id: c}, kg)
    vec = audit_basin(df, c, CFG)
    sh, records, _ = shacl_audit_basin(df, c, CFG, kg, ShapesGraph(CFG))
    assert vec.counts == sh.counts
    assert vec.evaluable == sh.evaluable
    assert vec.exclusive_counts == sh.exclusive_counts
    assert vec.applicable == sh.applicable


def test_applicability_is_inferred_from_ontology():
    undercatch = BasinContext(basin_id="00000010", runoff_ratio_train=1.2, pet_mean=2.0, omega=float("nan"))
    normal = BasinContext(basin_id="00000011", runoff_ratio_train=0.4, pet_mean=2.0, omega=2.5, frac_snow=0.5)
    kg = build_catchment_graph([undercatch, normal], CFG)
    excl = excluded_rules(kg)
    assert excl["00000010"] == {"R5", "R6_curve"}
    assert "00000011" not in excl
    assert (URIRef(str(HK) + "catchment_00000011"), RDF.type, HK.SnowDominatedCatchment) in kg
    ctxs = apply_kg_applicability({"00000010": undercatch, "00000011": normal}, kg)
    assert not ctxs["00000010"].r5_applicable and not ctxs["00000010"].r6_curve_applicable
    assert ctxs["00000011"].r5_applicable and ctxs["00000011"].r6_curve_applicable


def test_outcomes_and_class_summary_queries():
    df, c = _random_basin(5)
    kg = build_catchment_graph([c], CFG)
    apply_kg_applicability({c.basin_id: c}, kg)
    res = audit_basin(df, c, CFG)
    audit = URIRef("https://w3id.org/hydrokg/run/test/audit_x")
    store.add_audit_outcomes(kg, audit, [res])
    rates = store.rates_from_kg(kg, audit)
    for r in res.counts:
        if res.evaluable[r]:
            assert rates.loc[c.basin_id, r] == pytest.approx(res.rate(r))
    summary = store.class_family_summary(kg, audit)
    expected = np.nanmean([res.rate(r) for r in ("R0", "R5", "R6") if res.evaluable[r]])
    assert summary.loc[c.basin_id, "ObservationFree"] == pytest.approx(expected)


def test_violation_superclass_counting():
    df, c = _random_basin(6)
    kg = build_catchment_graph([c], CFG)
    apply_kg_applicability({c.basin_id: c}, kg)
    _, records, _ = shacl_audit_basin(df, c, CFG, kg, ShapesGraph(CFG))
    audit = URIRef("https://w3id.org/hydrokg/run/test/audit_y")
    store.add_violation_facts(kg, audit, records, daily=True)
    n_free = store.violation_counts_by_superclass(kg, HK.ObservationFreeViolation).get(c.basin_id, 0)
    assert n_free == sum(1 for r in records if r["rule_id"] in ("R0", "R5", "R6"))


def _pipeline_provenance(feedback_period, holdout=False, obs_basin_set="all"):
    g = load_ontology(CFG)
    p = Provenance(g, "https://w3id.org/hydrokg/run/test")
    train, test = ("1980-10-01", "1995-09-30"), ("1995-10-01", "2010-09-30")
    obs_train = p.dataset(HK.DischargeObservations, "q_train", train, obs_basin_set)
    obs_test = p.dataset(HK.DischargeObservations, "q_test", test, "all")
    ckpt = p.entity(HK.ModelCheckpoint, "ckpt", p.activity(HK.TrainingActivity, "train", [obs_train]))
    used_obs = obs_train if feedback_period == "train" else obs_test
    pred_fb = p.entity(HK.PredictionSet, "pred_fb", p.activity(HK.PredictionActivity, "pfb", [ckpt]))
    report = p.entity(HK.AuditReport, "audit_fb", p.activity(HK.AuditActivity, "afb", [pred_fb, used_obs]))
    fb = p.entity(HK.FeedbackSignal, "fb", p.activity(HK.FeedbackActivity, "f", [report]))
    ft = p.entity(HK.ModelCheckpoint, "ft", p.activity(HK.FineTuningActivity, "ftune", [ckpt, fb, obs_train]))
    p.entity(HK.EvaluatedPredictionSet, "pred_test", p.activity(HK.PredictionActivity, "pt", [ft, fb]), test,
             "test" if holdout else "all")
    return g


def test_leakage_query_detects_v1_design_and_passes_v2():
    assert len(leakage_report(_pipeline_provenance("test"))) > 0       # v1: feedback from test-period audit
    assert len(leakage_report(_pipeline_provenance("train"))) == 0     # v2
    # spatial: training discharge covering held-out basins
    assert len(leakage_report(_pipeline_provenance("train", True, "all"), basin_holdout=True)) > 0
    assert len(leakage_report(_pipeline_provenance("train", True, "train"), basin_holdout=True)) == 0
