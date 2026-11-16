"""SHACL audit engine. For one basin:

  1. build a validation graph: the catchment (with its inferred applicability), one
     sosa:Observation-typed hk:DailyRecord per day, one hk:PeakEvent per observed peak,
     one hk:WaterYearWindow per water year and one hk:BudykoWindow per multi-year window;
  2. run the SPARQL derivations (hydrokg.kg.rules_rdf.derivation_updates);
  3. validate against the SHACL Core shapes (hydrokg.kg.rules_rdf.shapes_turtle);
  4. read every sh:ValidationResult as a violation of the rule its shape checks.

Event detection and window aggregation (feature extraction) are shared with the
vectorized engine; the admissibility decision for every rule is made by SHACL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyshacl import validate
from rdflib import RDF, XSD, Graph, Literal, Namespace, URIRef

from hydrokg.kg.graph import HK, bind_prefixes, catchment_iri
from hydrokg.kg.rules_rdf import derivation_updates, explanation_rule, shapes_turtle
from hydrokg.rules.context import BasinContext
from hydrokg.rules.engine import AuditResult, peak_events, r5_windows, r6_windows
from hydrokg.rules.registry import DAILY_RULES, RULE_IDS, RULES

SH = Namespace("http://www.w3.org/ns/shacl#")
VIOLATION_TYPE = {
    "R0": HK.NegativeFlowViolation, "R1": HK.ExtremeRatioViolation, "R2": HK.ZeroFlowCollapseViolation,
    "R3": HK.HighRelativeErrorViolation, "R4": HK.PeakTimingViolation, "R5": HK.WaterLimitViolation,
    "R6": HK.BudykoViolation,
}


def _dbl(v: float) -> Literal:
    return Literal(float(v), datatype=XSD.double)


class ShapesGraph:
    """Parsed shapes plus the map from (node or property) shape to rule id."""

    def __init__(self, cfg: dict):
        self.text = shapes_turtle(cfg)
        self.graph = Graph().parse(data=self.text, format="turtle")
        self.shape_rule: dict = {}
        for node_shape, rule in self.graph.subject_objects(HK.checksRule):
            rid = str(rule).split("#")[-1]
            self.shape_rule[node_shape] = rid
            for prop_shape in self.graph.objects(node_shape, SH.property):
                self.shape_rule[prop_shape] = rid


def basin_validation_graph(df: pd.DataFrame, ctx: BasinContext, cfg: dict,
                           catchment_graph: Graph) -> tuple[Graph, dict]:
    """Validation graph for one basin; returns the graph and the evaluable counts."""
    g = bind_prefixes(Graph())
    c = catchment_iri(ctx.basin_id)
    for triple in catchment_graph.triples((c, None, None)):
        g.add(triple)
    excluded = {str(r).split("#")[-1] for r in catchment_graph.objects(c, HK.excludedFromRule)}
    for r in catchment_graph.objects(c, HK.excludedFromCurve):
        g.add((c, HK.excludedFromCurve, r))
    evaluable = {r: 0 for r in RULE_IDS}

    # daily records
    qsim = df["qsim"].to_numpy(dtype=float)
    qobs = df["qobs"].to_numpy(dtype=float) if "qobs" in df else np.full(len(df), np.nan)
    floor = ctx.q_floor
    for t, day in enumerate(df.index):
        if not np.isfinite(qsim[t]):
            continue
        d = HK[f"d_{ctx.basin_id}_{day:%Y%m%d}"]
        g.add((d, RDF.type, HK.DailyRecord))
        g.add((d, HK.ofCatchment, c))
        g.add((d, HK.onDate, Literal(day.date(), datatype=XSD.date)))
        g.add((d, HK.qSim, _dbl(qsim[t])))
        evaluable["R0"] += 1
        if np.isfinite(qobs[t]):
            g.add((d, HK.qObs, _dbl(qobs[t])))
            if np.isfinite(floor) and qobs[t] > floor:
                for r in ("R1", "R2", "R3"):
                    evaluable[r] += 1

    # peak events (R4)
    if np.isfinite(ctx.event_height) and "qobs" in df:
        events = peak_events(df["qobs"], df["qsim"], ctx, cfg["rules"]["R4"])
        for i, ev in events.iterrows():
            e = HK[f"event_{ctx.basin_id}_{ev['t_obs']:%Y%m%d}"]
            g.add((e, RDF.type, HK.PeakEvent))
            g.add((e, HK.ofCatchment, c))
            g.add((e, HK.onDate, Literal(ev["t_obs"].date(), datatype=XSD.date)))
            g.add((e, HK.lagDays, Literal(int(ev["lag_days"]), datatype=XSD.integer)))
        evaluable["R4"] = len(events)

    # water-year windows (R5), created only where the ontology does not exclude R5
    if "p" in df and "R5" not in excluded:
        windows = r5_windows(df["qsim"], df["p"], cfg["rules"]["R5"])
        for _, w in windows.iterrows():
            node = HK[f"wy_{ctx.basin_id}_{int(w['water_year'])}"]
            g.add((node, RDF.type, HK.WaterYearWindow))
            g.add((node, HK.ofCatchment, c))
            g.add((node, HK.meanQSim, _dbl(w["q_sim_mean"])))
            g.add((node, HK.meanP, _dbl(w["p_mean"])))
        evaluable["R5"] = len(windows)

    # multi-year Budyko windows (R6)
    if "p" in df and np.isfinite(ctx.pet_mean):
        windows = r6_windows(df["qsim"], df["p"], ctx, cfg["rules"]["R6"])
        for _, w in windows.iterrows():
            node = HK[f"bw_{ctx.basin_id}_{int(w['start_wy'])}_{int(w['end_wy'])}"]
            g.add((node, RDF.type, HK.BudykoWindow))
            g.add((node, HK.ofCatchment, c))
            g.add((node, HK.meanQSim, _dbl(w["q_sim_mean"])))
            g.add((node, HK.meanP, _dbl(w["p_mean"])))
            if np.isfinite(w["ei_fu"]):
                g.add((node, HK.fuEvaporativeIndex, _dbl(w["ei_fu"])))
        evaluable["R6"] = len(windows)
    return g, evaluable


def shacl_audit_basin(df: pd.DataFrame, ctx: BasinContext, cfg: dict, catchment_graph: Graph,
                      shapes: ShapesGraph | None = None,
                      keep_graph: bool = False) -> tuple[AuditResult, list[dict], Graph | None]:
    """Audit one basin with the SHACL engine.

    Returns (AuditResult, violation records, validation graph or None). Each violation
    record has basin_id, rule_id, focus IRI and date."""
    shapes = shapes or ShapesGraph(cfg)
    g, evaluable = basin_validation_graph(df, ctx, cfg, catchment_graph)
    for update in derivation_updates():
        g.update(update)
    _, report, _ = validate(g, shacl_graph=shapes.graph, inference="none", advanced=False)

    hits: dict[str, set] = {r: set() for r in RULE_IDS}
    for result in report.subjects(RDF.type, SH.ValidationResult):
        rule_id = shapes.shape_rule.get(report.value(result, SH.sourceShape))
        if rule_id is not None:
            hits[rule_id].add(report.value(result, SH.focusNode))

    res = AuditResult(basin_id=ctx.basin_id)
    excluded = {str(r).split("#")[-1] for r in catchment_graph.objects(catchment_iri(ctx.basin_id), HK.excludedFromRule)}
    for r in RULE_IDS:
        res.counts[r] = len(hits[r])
        res.evaluable[r] = evaluable[r]
    res.applicable = {
        "R0": True, "R1": bool(np.isfinite(ctx.q_floor)), "R2": bool(np.isfinite(ctx.q_floor)),
        "R3": bool(np.isfinite(ctx.q_floor)), "R4": bool(np.isfinite(ctx.event_height)),
        "R5": "p" in df and "R5" not in excluded, "R6": "p" in df and bool(np.isfinite(ctx.pet_mean)),
    }

    records = []
    for r in RULE_IDS:
        for focus in hits[r]:
            date = g.value(focus, HK.onDate)
            records.append({"basin_id": ctx.basin_id, "rule_id": r, "focus": focus,
                            "date": pd.Timestamp(str(date)) if date is not None else pd.NaT})

    # root-cause attribution by inference over the day nodes that failed daily shapes
    for rid in DAILY_RULES:
        g.add((HK[rid], HK.rootCausePriority,
               Literal(cfg["attribution_precedence"].index(rid), datatype=XSD.integer)))
        for focus in hits[rid]:
            g.add((focus, HK.violates, HK[rid]))
    explained = set(g.query(explanation_rule()))
    for triple in explained:
        g.add(triple)
    res.exclusive_counts = {r: len(hits[r]) - sum(1 for f in hits[r] if (f, HK.explainedRule, HK[r]) in g)
                            for r in DAILY_RULES}
    return res, records, (g if keep_graph else None)
