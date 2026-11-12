# HydroKG-Audit

**Auditing and improving LSTM streamflow predictions with a hydrologic knowledge graph**

HydroKG-Audit evaluates daily streamflow predictions of a long short-term memory (LSTM)
network against a set of hydrologic rules, records every outcome in an RDF knowledge graph,
and uses the recorded violations to guide fine-tuning of the network without modifying its
loss function. The framework separates two properties that aggregate skill scores combine:
statistical agreement with observations and consistency with physical constraints.

## Framework

The framework has three components.

1. **Rule-based audit.** Seven rules (R0–R6) are evaluated on explicit evaluation units
   (days, observed peak events and water-balance windows). Basin-specific thresholds are
   derived from training-period data only.
2. **Knowledge graph.** Catchments, evaluation units, rules, violations and the provenance
   of every dataset and model are represented in RDF. Rules are executed as SHACL
   constraints, rule applicability and root causes are derived by rule-based inference, and
   provenance is recorded with PROV-O.
3. **Graph-guided fine-tuning.** Violation rates read from the knowledge graph set the
   sampling weights of training basins (curriculum) and enter the network as static basin
   descriptors (violation profile). Optional post-hoc corrections are triggered only by
   rules that do not use observed discharge.

### Auditing rules

| Rule | Evaluation unit | Violation | Class | Evidence |
|---|---|---|---|---|
| R0 Negative flow | day | $\hat Q_t < 0$ | Physical impossibility | observation-free |
| R1 Extreme ratio | day with $Q_t > q_{f}$ | $\hat Q_t / Q_t < 0.2$ or $> 5$ | Magnitude | observation-referenced |
| R2 Zero-flow collapse | day with $Q_t > q_{f}$ | $\hat Q_t < 0.01$ mm d⁻¹ | Magnitude | observation-referenced |
| R3 High relative error | day with $Q_t > q_{f}$ | $\lvert \hat Q_t - Q_t \rvert / Q_t > 1$ | Magnitude | observation-referenced |
| R4 Peak timing | observed peak event | $\lvert \ell \rvert > 2$ days | Timing | observation-referenced |
| R5 Water limit | water year | $\overline{\hat Q} > \overline{P}$ | Budget | observation-free |
| R6 Budyko consistency | five-year window | $\hat\varepsilon > \phi$, or $\lvert \hat\varepsilon - F(\phi;\omega) \rvert > 0.15$ | Budget | observation-free |

$\hat Q$ and $Q$ are simulated and observed discharge, $P$ is precipitation, $q_f$ is a
basin low-flow floor, $\ell$ is the lag between observed and simulated peaks,
$\hat\varepsilon = 1 - \overline{\hat Q}/\overline{P}$ is the simulated evaporative index,
$\phi = E_p/\overline{P}$ is the dryness index, and $F(\phi;\omega)$ is the Fu (1981) curve with a
basin parameter $\omega$ fitted to training-period observations. A day that violates several
daily rules is attributed to its most fundamental cause (R0, R2, R1, R3). Rule rates are
summarized by a weighted violation burden $V$ and, separately, by a physical-consistency index
$V^{phys}$ (R0, R5, R6) and a prediction-error index $V^{err}$ (R1–R4).

### Knowledge graph

| Element | Representation |
|---|---|
| Ontology | `hydrokg/kg/resources/hydrokg.ttl`; reuses SOSA/SSN, PROV-O and SKOS |
| Evaluation units | `hk:DailyRecord`, `hk:PeakEvent`, `hk:WaterYearWindow`, `hk:BudykoWindow` (subclasses of `sosa:Observation`) |
| Rule execution | SPARQL derivations of constrained quantities, followed by SHACL Core validation; each validation result is a violation |
| Applicability | Catchment classes (e.g. precipitation undercatch, non-identifiable Budyko parameter) inferred from training-period properties; rule exclusions entailed from ontology axioms |
| Root cause | Same-day violations linked to the more fundamental rule (`hk:explainedRule`) |
| Provenance | Every dataset, checkpoint, prediction set, audit, feedback signal and correction is a PROV-O entity or activity with its period and basin set |
| Leakage check | SPARQL query over the provenance graph; a run stops if an evaluated prediction derives from discharge observations of the evaluation period or of held-out basins |
| Analog retrieval | Personalized PageRank on the catchment–concept graph |

A vectorized engine implements the same rule definitions for use inside the training loop;
the test suite verifies that both engines return identical results.

### Baseline model

The baseline is the LSTM of Kratzert et al. (2018, 2019): NLDAS-extended forcing and 27
static catchment attributes, 180-day input sequences, 128 hidden units, basin-normalized NSE
loss, Adam with a stepwise learning-rate schedule. The original implementation is included
in `hydrokg/lstm/` without numerical changes; modifications are listed in
`hydrokg/lstm/VENDORED.md`.

### Experimental designs

- **Temporal:** all basins are used for training (1 Oct 1980 – 30 Sep 1995) and evaluation
  (1 Oct 1995 – 30 Sep 2010).
- **Basin hold-out:** 20 % of basins, stratified by aridity class, are withheld from training
  and from every feedback computation.

Fine-tuning arms: `control` (continued training without feedback), `curriculum`, `profile`
and `curriculum_profile`. Each arm is evaluated without correction, with clipping of negative
flows, and with analog-based correction.

Full definitions and equations are given in [docs/METHODS.md](docs/METHODS.md).

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

CAMELS-US (Newman et al., 2015; Addor et al., 2017) with NLDAS-extended forcing
(Kratzert et al., 2019). The data root must contain:

```
CAMELS_US/
  basin_mean_forcing/nldas_extended/
  usgs_streamflow/
  camels_attributes_v2.0/
```

The 670 basins used are listed in `configs/basin_list_670.txt`.

## Usage

Run a complete experiment for one seed:

```bash
hydrokg run --experiment configs/experiments/temporal.yaml --seed 658666 \
            --set paths.camels_root=/path/to/CAMELS_US experiment.audit_workers=8
```

Other commands:

```bash
# basin hold-out design
hydrokg run --experiment configs/experiments/basin_holdout.yaml --seed 658666 \
            --set paths.camels_root=/path/to/CAMELS_US

# summary statistics over seeds
hydrokg aggregate --experiment-dir results/hydrokg_v2/temporal

# audit existing prediction files
hydrokg reaudit --predictions model_a.p model_b.p --labels a b \
                --run-dir /path/to/baseline_run --out results/audit \
                --set paths.camels_root=/path/to/CAMELS_US

# provenance leakage query on a saved knowledge graph
hydrokg leakage --kg results/hydrokg_v2/temporal/seed_658666/kg/hydrokg.ttl
```

Runs are resumable: each step writes its output before the next begins and is skipped when
the output exists. SLURM templates are provided in `scripts/slurm/`.

## Configuration

All parameters (periods, rule thresholds, model settings, fine-tuning, correction,
knowledge-graph options) are defined in `configs/default.yaml`. Experiment files in
`configs/experiments/` override selected keys, and any key can be overridden on the command
line, for example `--set rules.R6.max_budyko_deviation=0.10`. The configuration and the SHACL
shapes used are saved with every run.

## Outputs

```
results/<experiment>/<design>/seed_<seed>/
  config.yaml            configuration of the run
  split.p                training and evaluation basins
  contexts_eval.csv      basin thresholds, Fu parameter and rule applicability
  predictions/           simulated and observed discharge per basin and model
  audits/                rule counts, evaluable units and rates per basin and model
  arms/<arm>/            fine-tuned model, curriculum weights, online audits, violation profile
  kg/hydrokg.ttl         knowledge graph
  kg/shapes.ttl          SHACL shapes
  kg/leakage_report.csv  result of the leakage query
  analysis/              skill, violation burden, comparisons and diagnostics
```

## Repository structure

```
hydrokg/
  lstm/       baseline LSTM (Kratzert et al.)
  data/       CAMELS access, prediction files, basin splits
  rules/      rule definitions, basin context, vectorized engine
  kg/         ontology, SHACL shapes, inference rules, SHACL engine, provenance, analogs
  metrics/    skill scores, violation burden, hydrologic signatures, statistics
  enhance/    feedback signals, fine-tuning, corrections
  pipeline/   experiment orchestration, audits, analysis, command-line interface
configs/      default configuration, experiment configurations, basin list
docs/         methods
scripts/      SLURM job templates
tests/        unit, engine-equivalence, provenance and end-to-end tests
```

## References

- Addor, N., Newman, A. J., Mizukami, N., & Clark, M. P. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. *Hydrology and Earth System Sciences*, 21, 5293–5313.
- Fu, B. P. (1981). On the calculation of the evaporation from land surface. *Scientia Atmospherica Sinica*, 5, 23–31.
- Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *Hydrology and Earth System Sciences*, 22, 6005–6022.
- Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. (2019). Toward improved predictions in ungauged basins: Exploiting the power of machine learning. *Water Resources Research*, 55, 11344–11354.
- Newman, A. J., et al. (2015). Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA. *Hydrology and Earth System Sciences*, 19, 209–223.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

MIT License for the HydroKG-Audit code. The baseline LSTM code in `hydrokg/lstm/` is
distributed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).