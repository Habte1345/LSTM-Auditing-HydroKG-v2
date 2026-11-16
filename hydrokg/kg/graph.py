"""Construction of the HydroKG graph: ontology, catchments, inferred catchment classes and
rule applicability."""

from __future__ import annotations

from importlib import resources
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef
from rdflib.namespace import PROV, SKOS

from hydrokg.kg.rules_rdf import catchment_inference_rules
from hydrokg.rules.context import BasinContext

HK = Namespace("https://w3id.org/hydrokg#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")


def catchment_iri(basin_id: str) -> URIRef:
    return HK[f"catchment_{basin_id}"]


def concept_iri(kind: str, label: str) -> URIRef:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(label))
    return HK[f"{kind}_{safe}"]


def _lit(value) -> Optional[Literal]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return Literal(bool(value), datatype=XSD.boolean)
    if isinstance(value, (int, np.integer)):
        return Literal(int(value), datatype=XSD.integer)
    if isinstance(value, (float, np.floating)):
        return Literal(float(value), datatype=XSD.double) if np.isfinite(value) else None
    return Literal(str(value))


def bind_prefixes(g: Graph) -> Graph:
    g.bind("hk", HK)
    g.bind("sosa", SOSA)
    g.bind("prov", PROV)
    g.bind("skos", SKOS)
    return g


def load_ontology(cfg: dict) -> Graph:
    """Ontology with root-cause priorities taken from cfg['attribution_precedence']."""
    g = bind_prefixes(Graph())
    text = resources.files("hydrokg.kg").joinpath("resources/hydrokg.ttl").read_text()
    g.parse(data=text, format="turtle")
    for rule in g.subjects(RDF.type, HK.Rule):
        g.remove((rule, HK.rootCausePriority, None))
    for priority, rule_id in enumerate(cfg["attribution_precedence"]):
        g.add((HK[rule_id], HK.rootCausePriority, Literal(priority, datatype=XSD.integer)))
    return g


def add_catchment(g: Graph, ctx: BasinContext) -> URIRef:
    c = catchment_iri(ctx.basin_id)
    g.add((c, RDF.type, HK.Catchment))
    g.add((c, HK.basinId, Literal(ctx.basin_id)))
    props = {
        HK.aridityIndex: ctx.aridity_index, HK.snowFraction: ctx.frac_snow, HK.petMean: ctx.pet_mean,
        HK.lowFlowFloor: ctx.q_floor, HK.eventHeight: ctx.event_height,
        HK.runoffRatioTrain: ctx.runoff_ratio_train, HK.zeroFlowFractionTrain: ctx.zero_flow_fraction_train,
        HK.fuOmega: ctx.omega,
    }
    for prop, value in props.items():
        lit = _lit(value)
        if lit is not None:
            g.add((c, prop, lit))
    for kind, label, prop, cls in (("aridity", ctx.aridity_class, HK.hasAridityClass, HK.AridityClass),
                                   ("landcover", ctx.landcover_class, HK.hasLandCoverClass, HK.LandCoverClass),
                                   ("huc", ctx.huc02, HK.inRegion, HK.HydrologicRegion)):
        if label and label != "unknown":
            node = concept_iri(kind, label)
            g.add((node, RDF.type, cls))
            g.add((node, SKOS.prefLabel, Literal(str(label))))
            g.add((c, prop, node))
    return c


def add_attribute_levels(g: Graph, attributes: pd.DataFrame, n_bins: int = 5) -> None:
    """Quantile levels of continuous attributes as concept nodes (hk:hasAttributeLevel)."""
    for column in attributes.columns:
        values = pd.to_numeric(attributes[column], errors="coerce")
        if values.notna().sum() < n_bins or values.nunique() < n_bins:
            continue
        bins = pd.qcut(values, n_bins, labels=False, duplicates="drop")
        for basin, level in bins.dropna().items():
            node = concept_iri(f"level_{column}", f"q{int(level) + 1}of{n_bins}")
            g.add((node, RDF.type, HK.AttributeLevel))
            g.add((node, SKOS.prefLabel, Literal(f"{column} quantile {int(level) + 1}/{n_bins}")))
            g.add((catchment_iri(str(basin)), HK.hasAttributeLevel, node))


def infer_catchment_facts(g: Graph, cfg: dict) -> int:
    """Apply the catchment-classification and applicability rules in order; returns the
    number of inferred triples."""
    n = 0
    for query in catchment_inference_rules(cfg):
        new = list(g.query(query))
        for triple in new:
            if triple not in g:
                g.add(triple)
                n += 1
    return n


def excluded_rules(g: Graph) -> dict[str, set[str]]:
    """{basin_id: {rule_id, ...}} for rules (or the R6 curve, as 'R6_curve') the ontology
    marks as not applicable."""
    out: dict[str, set[str]] = {}
    q = """PREFIX hk: <https://w3id.org/hydrokg#>
SELECT ?b ?rid ?kind WHERE {
  { ?c hk:excludedFromRule ?r . BIND("rule" AS ?kind) } UNION
  { ?c hk:excludedFromCurve ?r . BIND("curve" AS ?kind) }
  ?c hk:basinId ?b . ?r hk:ruleId ?rid .
}"""
    for b, rid, kind in g.query(q):
        out.setdefault(str(b), set()).add(str(rid) if str(kind) == "rule" else f"{rid}_curve")
    return out


def apply_kg_applicability(contexts: dict[str, BasinContext], g: Graph) -> dict[str, BasinContext]:
    """Set the context applicability flags from the KG inference (the KG is authoritative)."""
    excl = excluded_rules(g)
    for basin, ctx in contexts.items():
        e = excl.get(basin, set())
        ctx.r5_applicable = "R5" not in e
        ctx.r6_curve_applicable = ("R6_curve" not in e) and bool(np.isfinite(ctx.pet_mean))
    return contexts


def build_catchment_graph(contexts: Iterable[BasinContext], cfg: dict,
                          attributes_for_levels: Optional[pd.DataFrame] = None) -> Graph:
    g = load_ontology(cfg)
    for ctx in contexts:
        add_catchment(g, ctx)
    if attributes_for_levels is not None:
        add_attribute_levels(g, attributes_for_levels)
    infer_catchment_facts(g, cfg)
    return g
