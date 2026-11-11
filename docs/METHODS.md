# Methods

## 1. Data and periods

CAMELS-US, 670 basins (`configs/basin_list_670.txt`). Forcing: NLDAS-extended (precipitation,
shortwave radiation, maximum and minimum temperature, vapour pressure); 27 static attributes;
USGS daily discharge converted to mm d⁻¹. Training period 1 Oct 1980 – 30 Sep 1995; test
period 1 Oct 1995 – 30 Sep 2010. CAMELS missing-discharge codes (−999 cfs) are set to
missing before any metric or rule is evaluated.

Two designs:

| Design | Basins in training | Basins in testing | Generalization tested |
|---|---|---|---|
| `temporal` | all 670 | all 670 | in time |
| `basin_holdout` | 80 %, stratified by aridity class | remaining 20 % | in time and space |

Observed discharge of the test period is read only to score predictions. In
`basin_holdout`, no discharge of held-out basins is read by any step that produces a
prediction.

## 2. Baseline LSTM

The LSTM, data pipeline and training procedure are those of the HydroAuditToolFrameowrk
code (Kratzert et al., 2018, 2019; `hydrokg/lstm/VENDORED.md`): 5 dynamic inputs plus 27
static attributes concatenated at every step, 180-day input sequences, 128 hidden units,
initial forget-gate bias 5, dropout 0.2, basin-normalized NSE loss

$$
\mathcal{L}_{NSE} = \frac{1}{B}\sum_{i=1}^{B} \frac{\left(\hat Q_i - Q_i\right)^2}{\left(s_{b(i)} + \epsilon\right)^2}, \qquad \epsilon = 0.1,
$$

where $s_{b(i)}$ is the standard deviation of training-period discharge of the basin of
sample $i$; Adam with learning rate $5\times10^{-3}$, reduced to $5\times10^{-4}$ at
epoch 11 and $10^{-4}$ at epoch 21; gradient-norm clipping at 1; batch size 512.
The original run used 5 epochs; the revised experiments use 30.

## 3. Basin context (training period only)

For basin $b$ with training-period observations $Q_o$, precipitation $P$ and long-term
potential evapotranspiration $E_p$ (CAMELS `pet_mean`):

$$
q_{f,b} = \max\left(q_{abs},\; f\,\overline{Q_o}\right),\qquad
h_b = \mathrm{quantile}_{0.9}(Q_o),\qquad
\phi_b = E_p/\overline{P},\qquad
\varepsilon_b = 1 - \overline{Q_o}/\overline{P}.
$$

The Fu (1981) parameter $\omega_b$ solves $F(\phi_b;\omega_b) = \varepsilon_b$ with

$$
F(\phi;\omega) = 1 + \phi - \left(1 + \phi^{\omega}\right)^{1/\omega}, \qquad \omega \in (1, 20].
$$

Defaults: $q_{abs} = 0.01$ mm d⁻¹, $f = 0.05$. Basins whose $\varepsilon_b$ lies outside
the range of $F$ have no identifiable $\omega_b$. For held-out basins in feedback
computations, $\omega_b$ is the median of training basins of the same aridity class.

## 4. Auditing rules

Let $\hat Q_t$ and $Q_t$ be simulated and observed discharge on day $t$. Every rule is
evaluated on explicit evaluation units; its rate is the number of violating units divided
by the number of evaluable units.

| Rule | Unit (evaluable when) | Violation | Class | Evidence |
|---|---|---|---|---|
| R0 Negative flow | day ($\hat Q_t$ defined) | $\hat Q_t < 0$ | Physical impossibility | observation-free |
| R1 Extreme ratio | day ($Q_t > q_{f,b}$) | $\hat Q_t/Q_t < 0.2$ or $> 5$ | Magnitude | observation-referenced |
| R2 Zero-flow collapse | day ($Q_t > q_{f,b}$) | $\hat Q_t < 0.01$ mm d⁻¹ | Magnitude | observation-referenced |
| R3 High relative error | day ($Q_t > q_{f,b}$) | $\lvert\hat Q_t - Q_t\rvert/Q_t > 1$ | Magnitude | observation-referenced |
| R4 Peak timing | observed peak event | $\lvert\ell\rvert > 2$ d | Timing | observation-referenced |
| R5 Water limit | water year (≥ 330 d) | $\overline{\hat Q} > \overline{P}$ | Budget | observation-free |
| R6 Budyko consistency | 5-water-year window (≥ 1650 d) | $\hat\varepsilon > \phi$, or $\hat\varepsilon \ge 0$ and $\lvert\hat\varepsilon - F(\phi;\omega_b)\rvert > 0.15$ | Budget | observation-free |

R4: observed peaks are local maxima of $Q$ with $Q \ge h_b$, at least 30 days apart; the
lag is $\ell = \arg\max_{|k|\le 10} \hat Q_{t+k}$. R6: $\hat\varepsilon = 1 - \overline{\hat Q}/\overline{P}$
and $\phi = E_p/\overline{P}$ over the window. A window with $\hat\varepsilon < 0$ breaches
the water limit and is counted by R5 only.

**Applicability** (inferred in the knowledge graph, §6): R5 is not applied to basins whose
training-period observed runoff ratio is ≥ 1 (the forcing cannot close the water balance);
the curve part of R6 is not applied where $\omega_b$ is not identifiable.

**Root-cause attribution.** A day that violates several daily rules is attributed to the
most fundamental one, in the order R0, R2, R1, R3. Exclusive rates count each violating
day once.

**Rule validity.** The observation-free rules are also applied to observed discharge; the
resulting rate is the rule's false-alarm rate and is reported with every audit.

## 5. Violation-burden indices

With $\rho_{b,r}$ the rate of rule $r$ in basin $b$ (undefined if the rule is not
applicable or has no evaluable unit):

$$
V_b = \frac{\sum_{r} w_r\,\rho_{b,r}}{\sum_{r} w_r},
$$

with $w_r = 1/n_{c(r),b}$ (each violation class weighs one; $n_{c,b}$ is the number of
available rules of class $c$ in basin $b$) by default, and $w_r = 1$ as an alternative.
Indices by evidence family are reported separately:

$$
V^{phys}_b = \operatorname{mean}_{r\in\{R0,R5,R6\}} \rho_{b,r}, \qquad
V^{err}_b = \operatorname{mean}_{r\in\{R1,R2,R3,R4\}} \rho_{b,r}.
$$

## 6. Knowledge graph

The graph is RDF and reuses SOSA/SSN, PROV-O and SKOS (`hydrokg/kg/resources/hydrokg.ttl`).

1. **Rules as constraints.** For each basin, days, peak events and water-balance windows
   are instances of `hk:DailyRecord`, `hk:PeakEvent`, `hk:WaterYearWindow` and
   `hk:BudykoWindow` (subclasses of `sosa:Observation`). SPARQL updates derive the
   constrained quantities (ratio, relative error, evaporative and dryness indices, Budyko
   deviation) on evaluable units only; SHACL Core shapes rendered from the configuration
   state the admissible range of each quantity. Every SHACL validation result is a
   violation of the rule its shape checks.
2. **Applicability by inference.** SPARQL rules classify catchments
   (`hk:UndercatchCatchment`, `hk:SnowDominatedCatchment`, `hk:IntermittentCatchment`,
   `hk:NonIdentifiableBudykoCatchment`) from training-period properties; the axioms
   `hk:notApplicableTo` and `hk:curveNotApplicableTo` then entail which rules are excluded
   for which catchment. Both audit engines read applicability from the graph.
3. **Root cause by inference.** Day nodes that fail several daily shapes are linked to
   the rules whose failure is explained by a more fundamental rule (`hk:explainedRule`,
   ordered by `hk:rootCausePriority`).
4. **Violation hierarchy.** Rule-specific violation classes are subclasses of a class
   (e.g. `hk:MagnitudeFailureViolation`) and of an evidence family
   (`hk:ObservationFreeViolation`); class- and family-level counts are obtained by
   subclass entailment.
5. **Provenance.** Every dataset, model checkpoint, prediction set, audit, feedback
   signal and correction is a PROV-O entity or activity carrying its period and basin set.
   An entity generated by an activity is derived from everything the activity used.
   The leakage check is one SPARQL query: an evaluated prediction set leaks if it is
   transitively derived from discharge observations that overlap its period (or, in
   `basin_holdout`, from discharge of held-out basins). Every run executes the check and
   stops if it fails.
6. **Analogs.** Personalized PageRank on the catchment–concept graph (aridity class,
   land-cover class, HUC-2 region, quintile levels of the 27 attributes) retrieves donor
   catchments; it is benchmarked against attribute k-nearest neighbours and the class
   filter of v1 by the skill of donor-composite hydrographs over the training period.

7. **Feedback from the graph.** Curriculum weights and the violation profile are read from
   the graph: the training-period audit and every online audit of the fine-tuning loop are
   recorded as `hk:AuditActivity` nodes with their `hk:RuleOutcome`s, and a SPARQL query
   returns their rates only after a provenance query confirms that every discharge dataset
   the audit used lies inside the training period (`hydrokg.kg.store.feedback_rates`).

Semantics: schema axioms are RDFS/OWL; inference is rule-based (SPARQL CONSTRUCT) and
validation is SHACL, both under the closed-world assumption. Properties that take classes as
values (`hk:violationType`, `hk:notApplicableTo`) use OWL 2 punning, so the ontology targets
RDFS plus rule reasoning rather than OWL DL reasoning.

A vectorized engine implements the same definitions for use inside the training loop;
`tests/test_kg.py` asserts identical counts, evaluable units, applicability and exclusive
attribution.

## 7. Graph-guided fine-tuning

All arms warm-start from the baseline checkpoint and run the identical optimization step
(NSE loss, Adam, clipping) for 3 epochs at a constant learning rate of $5\times10^{-4}$:

| Arm | Sampling | Inputs |
|---|---|---|
| control | uniform shuffling | unchanged |
| curriculum | weighted by basin violation rate | unchanged |
| profile | uniform shuffling | + violation profile |
| curriculum_profile | weighted | + violation profile |

Curriculum weights: $s_b = \operatorname{mean}_{r\in\{R0..R3\}}\rho_{b,r}$,
$w_b = \max(s_b, 0.1\,\max s)^{1/T}$, $T = 1$; epoch 1 uses the training-period audit of
the baseline, later epochs the online detections of the previous epoch. Violation profile:
$(\rho_{b,r})_r$ from the training-period audit (all seven rules in `temporal`;
observation-free rules R0, R5, R6 in `basin_holdout`), appended as static attributes whose
input weights start at zero, so fine-tuning starts from the baseline function. The
detections never enter the loss.

## 8. Post-hoc correction

Only observation-free rules trigger a correction. `clip`: $\hat Q_t \leftarrow \max(\hat Q_t, 0)$
on R0 days. `analog`: on R0 days, $\hat Q_t \leftarrow \max\!\left(0,\; \tilde Q_b \cdot
\operatorname{median}_{d}\, \hat Q_{d,t}/\tilde Q_d\right)$ over the $k = 5$ analogs $d$
with $\hat Q_{d,t} \ge 0$, where $\tilde Q$ is the median simulated flow. Optional R5
projection rescales a violating water year onto $\overline{\hat Q} = \overline{P}$.

## 9. Statistics

Skill: KGE and its components $r$, $\alpha$, $\beta$ (Gupta et al., 2009) on days with both
series defined. Skill–trust relation: Spearman correlation of KGE with each index, rank
partial correlation controlling for the aridity index, and the spread of $V_b$ within KGE
bins. Comparisons: share of basins in each quadrant of $(\Delta\mathrm{KGE}, \Delta V)$,
Wilcoxon signed-rank tests, bootstrap 95 % intervals of mean changes by aridity class
(5000 resamples), and mean and standard deviation over three seeds. Independent
diagnostics (never used as training signals): runoff ratio, FDC mid-segment slope,
baseflow index (Lyne–Hollick, α = 0.925, three passes), zero-flow frequency, Q5 and Q95.
