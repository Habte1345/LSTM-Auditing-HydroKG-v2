"""Executable form of the rules in the knowledge graph.

1. SPARQL derivations (UPDATE) add the quantities each rule constrains
   (ratio, relative error, evaporative/dryness index, Budyko deviation) only on the
   evaluation units where the rule is evaluable.
2. SHACL Core shapes state the admissible range of each quantity. A SHACL validation
   result on a focus node IS a rule violation.
3. SPARQL inference rules (CONSTRUCT) classify catchments, derive rule applicability from
   the ontology (hk:notApplicableTo), and derive root-cause explanations (hk:explainedRule).

Thresholds are rendered from the configuration, so config/default.yaml stays the single
source of truth and the shapes file written next to every audit documents exactly what
was checked.
"""

from __future__ import annotations

HK = "https://w3id.org/hydrokg#"

PREFIXES = """PREFIX hk: <https://w3id.org/hydrokg#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX prov: <http://www.w3.org/ns/prov#>
"""


def derivation_updates() -> list[str]:
    """SPARQL UPDATEs applied to a per-basin validation graph before SHACL validation."""
    daily = PREFIXES + """
INSERT { ?d hk:simObsRatio ?ratio ; hk:relativeError ?rel ; hk:qSimReferenced ?s }
WHERE {
  ?d a hk:DailyRecord ; hk:ofCatchment ?c ; hk:qSim ?s ; hk:qObs ?o .
  ?c hk:lowFlowFloor ?f .
  FILTER(?o > ?f)
  BIND(?s / ?o AS ?ratio)
  BIND(ABS(?s - ?o) / ?o AS ?rel)
}"""
    event = PREFIXES + """
INSERT { ?e hk:absLagDays ?a }
WHERE { ?e a hk:PeakEvent ; hk:lagDays ?k . BIND(IF(?k < 0, -?k, ?k) AS ?a) }"""
    budyko = PREFIXES + """
INSERT { ?w hk:evaporativeIndex ?ei ; hk:drynessIndex ?ai }
WHERE {
  ?w a hk:BudykoWindow ; hk:ofCatchment ?c ; hk:meanQSim ?q ; hk:meanP ?p .
  ?c hk:petMean ?pet .
  FILTER(?p > 0)
  BIND(1.0e0 - ?q / ?p AS ?ei)
  BIND(?pet / ?p AS ?ai)
}"""
    curve = PREFIXES + """
INSERT { ?w hk:absBudykoDeviation ?dev }
WHERE {
  ?w a hk:BudykoWindow ; hk:ofCatchment ?c ; hk:evaporativeIndex ?ei ; hk:fuEvaporativeIndex ?fu .
  FILTER NOT EXISTS { ?c hk:excludedFromCurve hk:R6 }
  FILTER(?ei >= 0.0)
  BIND(ABS(?ei - ?fu) AS ?dev)
}"""
    return [daily, event, budyko, curve]


def shapes_turtle(cfg: dict) -> str:
    r = cfg["rules"]
    energy = ""
    if r["R6"]["energy_limit"]:
        energy = """
hk:R6_energy_shape a sh:NodeShape ; hk:checksRule hk:R6 ;
    rdfs:comment "Energy limit: ET_sim <= PET, i.e. EI_sim <= AI." ;
    sh:targetClass hk:BudykoWindow ;
    sh:property [ sh:path hk:evaporativeIndex ; sh:lessThanOrEquals hk:drynessIndex ] ."""
    return f"""@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix hk:   <https://w3id.org/hydrokg#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .


hk:R0_shape a sh:NodeShape ; hk:checksRule hk:R0 ;
    rdfs:comment "Negative flow: Q_sim >= 0." ;
    sh:targetClass hk:DailyRecord ;
    sh:property [ sh:path hk:qSim ; sh:minInclusive "0.0"^^xsd:double ] .

hk:R1_shape a sh:NodeShape ; hk:checksRule hk:R1 ;
    rdfs:comment "Extreme ratio: {r['R1']['low_ratio']} <= Q_sim/Q_obs <= {r['R1']['high_ratio']} where Q_obs > q_floor." ;
    sh:targetClass hk:DailyRecord ;
    sh:property [ sh:path hk:simObsRatio ;
                  sh:minInclusive "{float(r['R1']['low_ratio'])}"^^xsd:double ;
                  sh:maxInclusive "{float(r['R1']['high_ratio'])}"^^xsd:double ] .

hk:R2_shape a sh:NodeShape ; hk:checksRule hk:R2 ;
    rdfs:comment "Zero-flow collapse: Q_sim >= {r['R2']['sim_zero']} mm/day where Q_obs > q_floor." ;
    sh:targetClass hk:DailyRecord ;
    sh:property [ sh:path hk:qSimReferenced ; sh:minInclusive "{float(r['R2']['sim_zero'])}"^^xsd:double ] .

hk:R3_shape a sh:NodeShape ; hk:checksRule hk:R3 ;
    rdfs:comment "High relative error: |Q_sim - Q_obs|/Q_obs <= {r['R3']['max_relative_error']} where Q_obs > q_floor." ;
    sh:targetClass hk:DailyRecord ;
    sh:property [ sh:path hk:relativeError ; sh:maxInclusive "{float(r['R3']['max_relative_error'])}"^^xsd:double ] .

hk:R4_shape a sh:NodeShape ; hk:checksRule hk:R4 ;
    rdfs:comment "Peak timing: |lag| <= {r['R4']['max_lag_days']} days." ;
    sh:targetClass hk:PeakEvent ;
    sh:property [ sh:path hk:absLagDays ; sh:maxInclusive {int(r['R4']['max_lag_days'])} ] .

hk:R5_shape a sh:NodeShape ; hk:checksRule hk:R5 ;
    rdfs:comment "Water limit: mean Q_sim <= mean P over a water year." ;
    sh:targetClass hk:WaterYearWindow ;
    sh:property [ sh:path hk:meanQSim ; sh:lessThanOrEquals hk:meanP ] .
{energy}
hk:R6_curve_shape a sh:NodeShape ; hk:checksRule hk:R6 ;
    rdfs:comment "Budyko consistency: |EI_sim - Fu(AI; omega_b)| <= {r['R6']['max_budyko_deviation']}." ;
    sh:targetClass hk:BudykoWindow ;
    sh:property [ sh:path hk:absBudykoDeviation ; sh:maxInclusive "{float(r['R6']['max_budyko_deviation'])}"^^xsd:double ] .
"""


def catchment_inference_rules(cfg: dict) -> list[str]:
    """CONSTRUCT rules: catchment classes from data properties, then rule applicability
    from the ontology axioms hk:notApplicableTo / hk:curveNotApplicableTo."""
    s = cfg["stratification"]
    r5 = cfg["rules"]["R5"]
    return [
        PREFIXES + f"""CONSTRUCT {{ ?c a hk:UndercatchCatchment }}
WHERE {{ ?c a hk:Catchment ; hk:runoffRatioTrain ?rr . FILTER(?rr >= {float(r5['undercatch_runoff_ratio'])}) }}""",
        PREFIXES + f"""CONSTRUCT {{ ?c a hk:SnowDominatedCatchment }}
WHERE {{ ?c a hk:Catchment ; hk:snowFraction ?f . FILTER(?f >= {float(s['snow_dominated_fraction'])}) }}""",
        PREFIXES + f"""CONSTRUCT {{ ?c a hk:IntermittentCatchment }}
WHERE {{ ?c a hk:Catchment ; hk:zeroFlowFractionTrain ?z . FILTER(?z >= {float(s['intermittent_zero_fraction'])}) }}""",
        PREFIXES + """CONSTRUCT { ?c a hk:NonIdentifiableBudykoCatchment }
WHERE { ?c a hk:Catchment . FILTER NOT EXISTS { ?c hk:fuOmega ?w } }""",
        PREFIXES + """CONSTRUCT { ?c hk:excludedFromRule ?r }
WHERE { ?r a hk:Rule ; hk:notApplicableTo ?k . ?c a ?k . }""",
        PREFIXES + """CONSTRUCT { ?c hk:excludedFromCurve ?r }
WHERE { ?r a hk:Rule ; hk:curveNotApplicableTo ?k . ?c a ?k . }""",
    ]


def explanation_rule() -> str:
    """Root cause: on a day that violates several daily rules, every rule except the one
    with the lowest hk:rootCausePriority is explained by that more fundamental failure."""
    return PREFIXES + """CONSTRUCT { ?d hk:explainedRule ?rv }
WHERE {
  ?d hk:violates ?rv . ?d hk:violates ?ru .
  ?rv hk:rootCausePriority ?pv . ?ru hk:rootCausePriority ?pu .
  FILTER(?pu < ?pv)
}"""
