# HydroKG-Audit

Auditing and improving LSTM streamflow predictions with a hydrologic knowledge graph.
Full details are in the accompanying paper.

## Overview

- **Baseline model:** an LSTM trained on CAMELS-US catchments (Kratzert et al., 2018, 2019).
- **Auditing rules:** predictions are checked against seven rules.

  | Rule | Checks |
  |---|---|
  | R0 | Negative streamflow |
  | R1 | Extreme over- or under-prediction |
  | R2 | Predicted flow collapsing to zero |
  | R3 | Large relative error |
  | R4 | Peak-timing error |
  | R5 | Annual water balance |
  | R6 | Consistency with the Budyko framework |

- **Knowledge graph:** audit results are stored in an RDF graph that executes the rules as
  SHACL constraints, infers which rules apply to each catchment, and records the provenance
  of all data, models and audits.
- **Graph-guided fine-tuning:** violation rates from the graph adjust catchment sampling and
  serve as additional catchment descriptors during fine-tuning.

## Installation

```bash
git clone https://github.com/Habte1345/LSTM-Auditing-HydroKG-v2.git
cd LSTM-Auditing-HydroKG-v2
pip install -e .
```

## Data

CAMELS-US with NLDAS-extended forcing.

## Usage

```bash
hydrokg run --experiment configs/experiments/temporal.yaml --seed 658666 \
            --set paths.camels_root=/path/to/CAMELS_US
```

Settings are in `configs/default.yaml`. Results are written to `results/`.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT. The baseline LSTM code in `hydrokg/lstm/` is under the Apache License 2.0
(see [NOTICE](NOTICE)).