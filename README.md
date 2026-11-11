# HydroKG-Audit

**Auditing and improving LSTM streamflow predictions with a hydrologic knowledge graph**

HydroKG audits daily streamflow predictions against seven hydrologic rules, stores the
outcome in an RDF knowledge graph, and uses training-period violations to guide
fine-tuning of an LSTM without changing its loss function. Version 2 is a complete
re-implementation addressing the reviews of manuscript 2026GL125916
(see [docs/REVIEW_MAP.md](docs/REVIEW_MAP.md)).

- Rules R0–R6 with explicit evaluation units, training-period thresholds and a
  physical-consistency / prediction-error split ([docs/METHODS.md](docs/METHODS.md) §4–5)
- Knowledge graph built on SOSA, PROV-O and SKOS: rules executed as **SHACL** constraints,
  rule applicability and root causes derived by **inference**, full **PROV-O** provenance
  with an automatic leakage check (§6)
- Baseline LSTM: the Kratzert et al. code of
  [HydroAuditToolFrameowrk](https://github.com/Habte1345/HydroAuditToolFrameowrk), vendored
  without numerical changes ([hydrokg/lstm/VENDORED.md](hydrokg/lstm/VENDORED.md))
- Leak-free fine-tuning arms with a continuation control, observation-free corrections,
  temporal and basin-hold-out designs, multi-seed aggregation (§7–9)

## Installation

```bash
git clone https://github.com/<owner>/hydrokg-audit.git
cd hydrokg-audit
pip install -e ".[dev]"          # Python >= 3.10
pytest                            # 28 tests, ~3 min (includes two end-to-end runs on synthetic data)
```

## Data

CAMELS-US (Newman et al., 2015; Addor et al., 2017) with the NLDAS-extended forcing used by
Kratzert et al. (2019). The root directory must contain `basin_mean_forcing/nldas_extended/`,
`usgs_streamflow/` and `camels_attributes_v2.0/`.

## Usage

```bash
# full experiment for one seed (resumable; every step is cached)
hydrokg run --experiment configs/experiments/temporal.yaml --seed 658666 \
            --set paths.camels_root=/data/CAMELS_US experiment.audit_workers=8

# spatial generalization
hydrokg run --experiment configs/experiments/basin_holdout.yaml --set paths.camels_root=/data/CAMELS_US

# mean and standard deviation over seeds
hydrokg aggregate --experiment-dir results/hydrokg_v2/temporal

# re-audit existing prediction files (e.g. v1 baseline and enhanced) with the revised rules
hydrokg reaudit --predictions baseline.p finetuned.p final.p --labels baseline v1_finetuned v1_final \
                --run-dir /path/to/baseline_run --out results/reaudit_v1 --set paths.camels_root=/data/CAMELS_US

# provenance leakage check on any saved graph
hydrokg leakage --kg results/hydrokg_v2/temporal/seed_658666/kg/hydrokg.ttl
```

SLURM templates: `scripts/slurm/run_seed.sbatch` (array over seeds) and
`scripts/slurm/reaudit_v1.sbatch`.

All settings are in `configs/default.yaml`; experiment files override them and any key can
be overridden on the command line (`--set rules.R6.max_budyko_deviation=0.1`).

## Outputs (per seed)

```
results/<experiment>/<split>/seed_<s>/
  config.yaml                     exact configuration
  split.p, split_summary.csv      basins used for training and testing
  baseline/                       LSTM run directory (cfg.json, attributes.db, model_epoch*.pt)
  contexts_eval.csv               training-period basin context (floors, omega, applicability)
  predictions/*.p                 {basin: DataFrame(qobs, qsim, p)}; *_missing.csv QC report
  audits/*.csv                    per-basin counts, evaluable units, rates, exclusive counts
  arms/<arm>/                     fine-tuned model, curriculum weights, online detections, profile
  kg/hydrokg.ttl                  knowledge graph (catchments, inferred classes, rule outcomes,
                                  violation facts, provenance)
  kg/shapes.ttl                   SHACL shapes exactly as validated
  kg/leakage_report.csv           result of the provenance leakage query (must be empty)
  analysis/*.csv                  tables listed in docs/REVIEW_MAP.md
```

## Repository layout

```
hydrokg/
  lstm/        vendored baseline LSTM (Apache-2.0)
  data/        CAMELS access, prediction files, basin splits
  rules/       rule registry, training-period basin context, vectorized engine
  kg/          ontology, SHACL shapes, SPARQL rules, SHACL engine, store, provenance, analogs
  metrics/     skill, burden indices, signatures, statistics
  enhance/     feedback signals, fine-tuning, corrections
  pipeline/    experiment orchestration, audits, analysis, CLI
configs/       default configuration, experiments, basin list
docs/          METHODS.md, REVIEW_MAP.md
tests/         unit tests, engine-equivalence tests, leakage tests, end-to-end tests
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

HydroKG code: MIT. Vendored LSTM code in `hydrokg/lstm/`: Apache-2.0 (Kratzert et al.).
See [LICENSE](LICENSE) and [NOTICE](NOTICE).
