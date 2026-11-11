"""Audit many basins with either engine, optionally in parallel processes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import pandas as pd
from rdflib import Graph
from tqdm import tqdm

from hydrokg.kg.graph import HK, bind_prefixes, catchment_iri
from hydrokg.rules.context import BasinContext
from hydrokg.rules.engine import AuditResult, audit_basin, audit_observations


def _catchment_triples(g: Graph, basin_id: str) -> list:
    return list(g.triples((catchment_iri(basin_id), None, None)))


def _worker(args):
    engine, df, ctx, cfg, triples, keep_flags = args
    if engine == "shacl":
        from hydrokg.kg.validation import shacl_audit_basin
        g = bind_prefixes(Graph())
        for t in triples:
            g.add(t)
        res, records, _ = shacl_audit_basin(df, ctx, cfg, g)
    else:
        res, records = audit_basin(df, ctx, cfg), []
    validity = audit_observations(df, ctx, cfg) if df["qobs"].notna().any() else None
    if not keep_flags:
        res.daily_flags = res.daily_evaluable = None
    return res, records, validity


def audit_predictions(preds: dict[str, pd.DataFrame], contexts: dict[str, BasinContext], cfg: dict,
                      catchment_graph: Graph, engine: Optional[str] = None, workers: int = 1,
                      keep_flags: bool = False, desc: str = "audit") -> dict:
    """Returns {'table': per-basin summary, 'results': [AuditResult], 'records': [...],
    'validity': per-basin rates of observation-free rules applied to observations}."""
    engine = engine or cfg["kg"]["engine"]
    jobs = [(engine, preds[b], contexts[b], cfg, _catchment_triples(catchment_graph, b), keep_flags)
            for b in preds if b in contexts]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(tqdm(pool.map(_worker, jobs, chunksize=4), total=len(jobs), desc=desc))
    else:
        outputs = [_worker(j) for j in tqdm(jobs, desc=desc)]
    results = [o[0] for o in outputs]
    records = [r for o in outputs for r in o[1]]
    table = pd.DataFrame([r.summary() for r in results]).set_index("basin_id")
    validity = pd.DataFrame([{"basin_id": o[2].basin_id, **{f"obs_rate_{k}": o[2].rate(k) for k in ("R0", "R5", "R6")}}
                             for o in outputs if o[2] is not None])
    if len(validity):
        validity = validity.set_index("basin_id")
    for ctx_col in ("aridity_class", "landcover_class", "aridity_index", "snow_dominated", "intermittent"):
        table[ctx_col] = [getattr(contexts[b], ctx_col) for b in table.index]
    return {"table": table, "results": results, "records": records, "validity": validity}
