"""Persistent HydroKG: ontology + catchments + audit outcomes + violation facts + provenance.

Feedback signals for training (curriculum weights, violation profile) are read from this
graph with SPARQL, restricted to audits whose provenance shows they used training-period
observations only (hydrokg.kg.provenance.assert_feedback_is_leak_free).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rdflib import RDF, XSD, Graph, Literal, URIRef
from rdflib.namespace import PROV

from hydrokg.kg.graph import HK, catchment_iri
from hydrokg.kg.validation import VIOLATION_TYPE
from hydrokg.rules.engine import AuditResult
from hydrokg.rules.registry import RULE_IDS


def outcome_iri(audit: URIRef, basin_id: str, rule_id: str) -> URIRef:
    return URIRef(f"{audit}/outcome/{basin_id}/{rule_id}")


def add_audit_outcomes(g: Graph, audit: URIRef, results: Iterable[AuditResult]) -> None:
    """One hk:RuleOutcome per (basin, rule) for an audit activity."""
    for res in results:
        c = catchment_iri(res.basin_id)
        for r in RULE_IDS:
            if not res.applicable.get(r, False) or res.evaluable.get(r, 0) == 0:
                continue
            o = outcome_iri(audit, res.basin_id, r)
            g.add((o, RDF.type, HK.RuleOutcome))
            g.add((o, HK.forCatchment, c))
            g.add((o, HK.forRule, HK[r]))
            g.add((o, HK.nViolations, Literal(int(res.counts[r]), datatype=XSD.integer)))
            g.add((o, HK.nEvaluable, Literal(int(res.evaluable[r]), datatype=XSD.integer)))
            g.add((o, HK.rate, Literal(res.counts[r] / res.evaluable[r], datatype=XSD.double)))
            g.add((o, PROV.wasGeneratedBy, audit))


def add_violation_facts(g: Graph, audit: URIRef, records: list[dict], daily: bool) -> int:
    """Violation individuals (typed by rule-specific class). Daily facts only if `daily`."""
    n = 0
    for rec in records:
        if not daily and rec["rule_id"] in ("R0", "R1", "R2", "R3"):
            continue
        v = URIRef(f"{audit}/violation/{rec['basin_id']}/{rec['rule_id']}/{n}")
        g.add((v, RDF.type, VIOLATION_TYPE[rec["rule_id"]]))
        g.add((v, HK.violatesRule, HK[rec["rule_id"]]))
        g.add((v, HK.ofCatchment, catchment_iri(rec["basin_id"])))
        if pd.notna(rec.get("date")):
            g.add((v, HK.onDate, Literal(rec["date"].date(), datatype=XSD.date)))
        g.add((v, PROV.wasGeneratedBy, audit))
        n += 1
    return n


def rates_from_kg(g: Graph, audit: URIRef) -> pd.DataFrame:
    """Basin x rule rate matrix of one audit (NaN where not applicable)."""
    q = """PREFIX hk: <https://w3id.org/hydrokg#> PREFIX prov: <http://www.w3.org/ns/prov#>
SELECT ?b ?rid ?rate WHERE {
  ?o a hk:RuleOutcome ; prov:wasGeneratedBy ?audit ; hk:forCatchment ?c ; hk:forRule ?r ; hk:rate ?rate .
  ?c hk:basinId ?b . ?r hk:ruleId ?rid .
}"""
    rows = [(str(b), str(r), float(v)) for b, r, v in g.query(q, initBindings={"audit": audit})]
    if not rows:
        return pd.DataFrame(columns=RULE_IDS)
    return (pd.DataFrame(rows, columns=["basin_id", "rule", "rate"])
            .pivot(index="basin_id", columns="rule", values="rate").reindex(columns=RULE_IDS))


def feedback_rates(g: Graph, audit: URIRef, train_period: tuple) -> pd.DataFrame:
    """The ONLY source of training feedback (curriculum weights, violation profile).

    1. Provenance check in the graph: every discharge dataset the audit activity used
       must lie inside the training period (hydrokg.kg.provenance.assert_feedback_is_leak_free).
    2. Rates of that audit, read from its hk:RuleOutcome nodes with SPARQL."""
    from hydrokg.kg.provenance import assert_feedback_is_leak_free
    assert_feedback_is_leak_free(g, audit, train_period)
    return rates_from_kg(g, audit)


def class_family_summary(g: Graph, audit: URIRef) -> pd.DataFrame:
    """Mean rate per basin and violation class / evidence family, by joining outcomes to the
    rule metadata asserted in the ontology."""
    q = """PREFIX hk: <https://w3id.org/hydrokg#> PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?b ?group (AVG(?rate) AS ?mean_rate) WHERE {
  ?o a hk:RuleOutcome ; prov:wasGeneratedBy ?audit ; hk:forCatchment ?c ; hk:forRule ?r ; hk:rate ?rate .
  ?c hk:basinId ?b .
  { ?r hk:inViolationClass ?k } UNION { ?r hk:inEvidenceFamily ?k }
  BIND(STRAFTER(STR(?k), "#") AS ?group)
} GROUP BY ?b ?group"""
    rows = [(str(b), str(k), float(v)) for b, k, v in g.query(q, initBindings={"audit": audit})]
    return pd.DataFrame(rows, columns=["basin_id", "group", "mean_rate"]).pivot(
        index="basin_id", columns="group", values="mean_rate")


def violation_counts_by_superclass(g: Graph, superclass: URIRef) -> pd.Series:
    """Number of stored violation facts per basin that belong to `superclass`
    (subclass closure by SPARQL property path; no materialization needed)."""
    q = """PREFIX hk: <https://w3id.org/hydrokg#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?b (COUNT(?v) AS ?n) WHERE {
  ?v a/rdfs:subClassOf* ?super ; hk:ofCatchment ?c . ?c hk:basinId ?b .
} GROUP BY ?b"""
    return pd.Series({str(b): int(n) for b, n in g.query(q, initBindings={"super": superclass})}, dtype=int)


def save(g: Graph, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(path), format="turtle")


def load(path: str | Path) -> Graph:
    from hydrokg.kg.graph import bind_prefixes
    return bind_prefixes(Graph()).parse(str(path), format="turtle")
