# HydroKG-Audit

Auditing and improving LSTM streamflow predictions with a hydrologic knowledge graph.

A summary of the workflow is given in [docs/METHODS.md](docs/METHODS.md); full details are
in the accompanying paper.

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