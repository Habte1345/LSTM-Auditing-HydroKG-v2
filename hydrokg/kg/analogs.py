"""Analog (donor) catchment retrieval and the benchmark that compares the graph-based
method against table-based alternatives.

class_filter : catchments sharing aridity and/or land-cover class (the original HydroKG v1 rule)
table_knn    : Euclidean k-nearest neighbours on z-scored static attributes
graph_ppr    : personalized PageRank on the catchment-concept graph of the KG
               (aridity class, land-cover class, hydrologic region, attribute quantile levels)

Benchmark: for each target catchment, the composite of its analogs' normalized hydrographs
(Q_d / mean Q_d, averaged over k donors) is compared with the target's normalized hydrograph
over the TRAINING period (correlation r and KGE). Better analogs reproduce the target's
dynamics more closely.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from rdflib import Graph

from hydrokg.kg.graph import HK
from hydrokg.metrics.skill import kge_components

CONCEPT_QUERY = """PREFIX hk: <https://w3id.org/hydrokg#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?b ?concept WHERE { ?c hk:basinId ?b ; ?p ?concept . ?p rdfs:subPropertyOf hk:hasConcept . }"""

CLASS_QUERY = """PREFIX hk: <https://w3id.org/hydrokg#>
SELECT ?b ?arid ?lc WHERE { ?c a hk:Catchment ; hk:basinId ?b .
  OPTIONAL { ?c hk:hasAridityClass ?arid } OPTIONAL { ?c hk:hasLandCoverClass ?lc } }"""


class AnalogFinder:
    def __init__(self, kg: Graph, attributes: Optional[pd.DataFrame] = None, alpha: float = 0.85):
        self.alpha = alpha
        self.graph = nx.Graph()
        for b, concept in kg.query(CONCEPT_QUERY):
            self.graph.add_edge(f"basin:{b}", str(concept))
        self.classes = {str(b): (str(a) if a else None, str(l) if l else None) for b, a, l in kg.query(CLASS_QUERY)}
        self.attributes = None
        if attributes is not None:
            a = attributes.apply(pd.to_numeric, errors="coerce")
            a = a.loc[:, a.std() > 0]
            self.attributes = ((a - a.mean()) / a.std()).fillna(0.0)

    def class_filter(self, target: str, pool: list[str], k: int) -> list[str]:
        ta, tl = self.classes.get(target, (None, None))
        scored = []
        for b in pool:
            if b == target:
                continue
            a, l = self.classes.get(b, (None, None))
            match = int(a is not None and a == ta) + int(l is not None and l == tl)
            if match:
                scored.append((match, b))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [b for _, b in scored[:k]]

    def table_knn(self, target: str, pool: list[str], k: int) -> list[str]:
        if self.attributes is None:
            raise ValueError("table_knn needs attributes")
        x = self.attributes
        cand = [b for b in pool if b != target and b in x.index]
        d = np.sqrt(((x.loc[cand] - x.loc[target]) ** 2).sum(axis=1))
        return d.sort_values(kind="stable").index[:k].tolist()

    def graph_ppr(self, target: str, pool: list[str], k: int) -> list[str]:
        node = f"basin:{target}"
        if node not in self.graph:
            return []
        scores = nx.pagerank(self.graph, alpha=self.alpha, personalization={node: 1.0})
        cand = [(scores.get(f"basin:{b}", 0.0), b) for b in pool if b != target]
        cand.sort(key=lambda x: (-x[0], x[1]))
        return [b for s, b in cand[:k] if s > 0]

    def find(self, method: str, target: str, pool: list[str], k: int) -> list[str]:
        return getattr(self, method)(target, pool, k)


def benchmark(finder: AnalogFinder, flows: dict[str, pd.Series], targets: list[str],
              donors: list[str], k: int, methods=("class_filter", "table_knn", "graph_ppr")) -> pd.DataFrame:
    """Leave-one-out donor-composite skill per target and method (training-period flows)."""
    norm = {b: (s / s.mean()) for b, s in flows.items() if s.notna().sum() > 365 and s.mean() > 0}
    rows = []
    for method in methods:
        for t in targets:
            if t not in norm:
                continue
            analogs = [a for a in finder.find(method, t, [d for d in donors if d in norm], k)]
            if not analogs:
                rows.append({"method": method, "basin_id": t, "r": np.nan, "kge": np.nan, "n_analogs": 0})
                continue
            composite = pd.concat([norm[a] for a in analogs], axis=1).mean(axis=1, skipna=True)
            comp = kge_components(norm[t].reindex(composite.index), composite)
            rows.append({"method": method, "basin_id": t, "r": comp["r"], "kge": comp["kge"],
                         "n_analogs": len(analogs)})
    return pd.DataFrame(rows)
