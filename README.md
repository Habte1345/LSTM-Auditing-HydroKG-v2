# HydroKG-Audit

**Auditing and improving LSTM streamflow predictions with a hydrologic knowledge graph**

HydroKG-Audit evaluates daily streamflow predictions of a long short-term memory (LSTM)
network against hydrologic rules, records the outcomes and their provenance in an RDF
knowledge graph, and uses the recorded violations to guide fine-tuning of the network
without modifying its loss function.

The framework consists of:

- a rule-based audit of predictions at daily, event and water-balance scales;
- a knowledge graph in which rules are executed as SHACL constraints, rule applicability
  and root causes are inferred, and all data, models and audits carry PROV-O provenance;
- graph-guided fine-tuning through basin sampling weights and violation-profile inputs,
  with optional post-hoc correction.

The scientific specification is given in [docs/METHODS.md](docs/METHODS.md).

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/Habte1345/LSTM-Auditing-HydroKG-v2.git
cd LSTM-Auditing-HydroKG-v2
pip install -e ".[dev]"
pytest
```

A conda environment is provided in `environment.yml`.

## Data

CAMELS-US with NLDAS-extended forcing. The data root must contain:

```
CAMELS_US/
  basin_mean_forcing/nldas_extended/
  usgs_streamflow/
  camels_attributes_v2.0/
```

The basins used are listed in `configs/basin_list_670.txt`.

## Usage

```bash
# complete experiment for one seed
hydrokg run --experiment configs/experiments/temporal.yaml --seed 658666 \
            --set paths.camels_root=/path/to/CAMELS_US

# basin hold-out experiment
hydrokg run --experiment configs/experiments/basin_holdout.yaml --seed 658666 \
            --set paths.camels_root=/path/to/CAMELS_US

# summary over seeds
hydrokg aggregate --experiment-dir results/hydrokg_v2/temporal

# audit existing prediction files
hydrokg reaudit --predictions a.p b.p --labels a b --run-dir /path/to/baseline_run \
                --out results/audit --set paths.camels_root=/path/to/CAMELS_US

# provenance leakage query on a saved knowledge graph
hydrokg leakage --kg results/hydrokg_v2/temporal/seed_658666/kg/hydrokg.ttl
```

Each step writes its output before the next begins and is skipped when the output exists,
so interrupted runs resume. SLURM templates are provided in `scripts/slurm/`.

## Configuration

All parameters are defined in `configs/default.yaml`. Files in `configs/experiments/`
override selected keys, and any key can be overridden on the command line, e.g.
`--set rules.R6.max_budyko_deviation=0.10`.

## Outputs

```
results/<experiment>/<design>/seed_<seed>/
  config.yaml            configuration of the run
  split.p                training and evaluation basins
  contexts_eval.csv      basin thresholds and rule applicability
  predictions/           simulated and observed discharge
  audits/                rule counts, evaluable units and rates per basin
  arms/<arm>/            fine-tuned models and feedback signals
  kg/hydrokg.ttl         knowledge graph
  kg/shapes.ttl          SHACL shapes
  kg/leakage_report.csv  provenance leakage query result
  analysis/              skill, violation burden and comparison tables
```

## Repository structure

```
hydrokg/
  lstm/       baseline LSTM (Kratzert et al., 2018, 2019)
  data/       data access and basin splits
  rules/      rule definitions and evaluation
  kg/         ontology, SHACL shapes, inference, provenance, analog retrieval
  metrics/    skill, violation burden, signatures, statistics
  enhance/    feedback signals, fine-tuning, correction
  pipeline/   experiment orchestration and command-line interface
configs/      configuration files and basin list
docs/         methods
scripts/      SLURM templates
tests/        test suite
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT License. The baseline LSTM code in `hydrokg/lstm/` is distributed under the Apache
License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).