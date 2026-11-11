# Reviewer comments → implementation → output

Manuscript 2026GL125916. "Table" = file in `results/<experiment>/<split>/seed_<s>/analysis/`.

## Reviewer 1

| Comment | What changed | Where | Evidence produced |
|---|---|---|---|
| V_b dominated by R4 (~51 % of baseline V_b) | R4 is event-based (observed peaks ≥ training-period 90th percentile, ±10-day search) instead of annual maxima; V_b uses explicit weights (default: each class weighs 1); per-rule and per-class rates always reported | `rules/engine.py::peak_events`, `metrics/burden.py` | `index_rule_shares`, `model_summary` (`rate_*`, `class_*`, `V_*`) |
| Dominant class: rates in text, raw counts in Fig. S1b/4b | Categorical "dominant class" dropped; class composition computed on one basis (normalized rates) | `metrics/burden.py::class_shares_by_group` | `class_shares_by_aridity`, `class_shares_by_landcover` |
| R5/R6 are the two sides of 0 ≤ mean Q_sim ≤ mean P; R6 not a Budyko test | R6 = energy limit (ET_sim ≤ PET) and distance to the basin's Fu curve (ω fitted on training data); water-limit breaches counted by R5 only | `rules/engine.py::r6_windows`, `rules/context.py` | `rate_R6`, `rule_validity_on_observations` |
| KGE does predict V_b coarsely | Reported directly | `metrics/stats.py` | `skill_trust_correlation` (Spearman, shared variance, partial ρ given aridity, per-class ρ), `skill_trust_dispersion_by_kge_bin` |
| Humid→arid gradient built into R1/R3 (division by Q_obs, Q_obs > 0 guard) | Basin low-flow floor from training data; sensitivity to the floor; observation-free index V_phys reported separately | `rules/context.py`, `pipeline/analysis.py::floor_sensitivity` | `low_flow_floor_sensitivity`, `V_phys` vs `V_err` |
| Three extra epochs with no baseline continuation | Control arm: same epochs, learning rate and optimizer step, no feedback | `enhance/finetune.py`, `configs/default.yaml: experiment.arms` | `*_vs_control_*` tables |
| Ablation of resampling / feature / correction; with and without correction | Arms control, curriculum, profile, curriculum_profile × corrections none, clip, analog | `pipeline/experiment.py` | `*_vs_baseline_*`, `*_vs_control_*`, `*_vs_uncorrected_*` |
| Feature and correction may use evaluation-period observations | Feedback from training-period audits only; correction triggered only by observation-free rules; PROV-O leakage query run (and enforced) every time | `enhance/feedback.py::require_training_period`, `config.py::validate_config`, `kg/provenance.py` | `kg/leakage_report.csv` (empty) |
| Benchmark R0 against clipping | `clip` correction | `enhance/correction.py` | `*+clip_*` vs `*+analog_*` |
| State observations available at prediction time; which basins are held out | Two explicit designs (temporal, basin_holdout) | `data/splits.py`, `docs/METHODS.md §1` | `split.p`, `split_summary.csv` |
| Four basin counts (531, 670, 669, ~318) | One basin list (670); counts reported per table (`n`) | `configs/basin_list_670.txt` | every table |
| Seed replication | Three seeds; aggregation | `pipeline/analysis.py::aggregate_seeds` | `seed_aggregate_model_summary.csv` |
| Basin-level KGE–V_b correlation with aridity control instead of bubble chart | Partial rank correlation | `metrics/stats.py::partial_spearman` | `skill_trust_correlation` |
| Minor: R3 threshold, R4 subscript | R3: relative error > 1; R4: lag between simulated and observed peak | `configs/default.yaml` | `kg/shapes.ttl` (exact constraints of every run) |
| Minor: R2 has no threshold | Q_sim < 0.01 mm d⁻¹ where Q_obs > q_floor | `configs/default.yaml` | — |
| Minor: R2 grouped under physical impossibility | R2 is a magnitude failure (observation-referenced) | `rules/registry.py`, ontology | — |
| Minor: rules not independent (negative flow trips R0, R1, R3) | Co-occurrence matrix and root-cause (exclusive) rates | `metrics/burden.py::cooccurrence`, `kg/rules_rdf.py::explanation_rule` | `daily_rule_cooccurrence`, `exclusive_rate_*` |
| Minor: R5 flags snow basins with precipitation undercatch | R5 not applicable where training-period observed runoff ratio ≥ 1 (inferred in the KG) | ontology `hk:UndercatchCatchment` | `rule_validity_on_observations` |
| Minor: which KGE component improves | r, α, β reported | `pipeline/analysis.py::comparison` | `*_paired_tests` (`|r-1|`, `|alpha-1|`, `|beta-1|`) |

## Reviewer 2

| Comment | What changed | Where | Evidence produced |
|---|---|---|---|
| 1. Extra training, schedule, resampling, features and correction confounded | Control arm and full ablation (above) | `enhance/finetune.py` | `*_vs_control_*` |
| 1. What does the graph add beyond a table? | Rules executed as SHACL constraints; applicability and root cause inferred; violation hierarchy by entailment; PROV-O leakage check; graph-based analogs benchmarked against attribute kNN | `hydrokg/kg/` | `kg/hydrokg.ttl`, `kg/shapes.ttl`, `kg/leakage_report.csv`, `analog_benchmark` |
| 2. V_b partly prediction error | V_phys (R0, R5, R6) and V_err (R1–R4) reported separately; independent signatures | `metrics/burden.py`, `metrics/signatures.py` | `V_phys`, `V_err`, `*_signature_abs_error_change` |
| 2. Near-zero flow sensitivity | Floor sensitivity | `pipeline/analysis.py::floor_sensitivity` | `low_flow_floor_sensitivity` |
| 3. Data split unclear; held-out observations in feature/correction | Explicit designs; leakage query | `data/splits.py`, `kg/provenance.py` | `split_summary.csv`, `kg/leakage_report.csv` |
| 4. Quadrant proportions; humid claim | Quadrant tables for V_b, V_phys, V_err; bootstrap by aridity class | `metrics/stats.py` | `*_quadrants_*`, `*_bootstrap_by_aridity` |

## Additional issue found during the re-implementation

The original evaluation kept CAMELS missing-discharge codes (−999 cfs → large negative
mm d⁻¹) in 8 basins (2,681 days), and KGE was computed on them. Masking changes the
KGE of affected basins by up to 3.2. All metrics now use masked observations
(`data/predictions.py::mask_missing`); each run writes `predictions/*_missing.csv`.
