# Methods overview

This document summarizes the workflow implemented in the repository. Definitions, equations
and parameter choices are given in the accompanying paper.

## Data

CAMELS-US catchments with NLDAS-extended meteorological forcing, static catchment attributes
and daily observed streamflow. Data are split into a training period (1980–1995) and an
evaluation period (1995–2010). A second design additionally withholds a set of catchments
from training.

## Baseline model

A long short-term memory (LSTM) network trained across all training catchments, following
Kratzert et al. (2018, 2019).

## Auditing rules

Predictions are checked against seven rules:

| Rule | Checks |
|---|---|
| R0 | Negative streamflow |
| R1 | Extreme over- or under-prediction |
| R2 | Predicted flow collapsing to zero |
| R3 | Large relative error |
| R4 | Peak-timing error |
| R5 | Annual water balance |
| R6 | Consistency with the Budyko framework |

Rules R0, R5 and R6 test physical consistency; rules R1–R4 test agreement with observations.
Catchment-specific thresholds are derived from training-period data.

## Knowledge graph

Audit results are stored in an RDF knowledge graph that:

- executes the rules as SHACL constraints;
- infers which rules apply to each catchment;
- records the provenance of all data, models and audits;
- checks that no evaluation-period observations are used to produce predictions.

## Graph-guided fine-tuning

Violation rates read from the knowledge graph are used to adjust how often each catchment is
sampled during training and as additional catchment descriptors. A control configuration
continues training without this feedback. Optional post-hoc corrections address negative
flows.

## Evaluation

Configurations are compared on predictive skill (Kling–Gupta efficiency) and on rule
violation rates across catchments and climate classes.