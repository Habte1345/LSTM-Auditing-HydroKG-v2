# Methods

## 1. Data

CAMELS-US (Newman et al., 2015; Addor et al., 2017), 670 basins. Dynamic inputs are
NLDAS-extended daily precipitation, shortwave radiation, maximum and minimum temperature and
vapour pressure (Kratzert et al., 2019); static inputs are 27 catchment attributes. Observed
discharge is converted to mm d⁻¹, and missing-value codes are treated as missing.

The training period is 1 October 1980 – 30 September 1995 and the evaluation period
1 October 1995 – 30 September 2010. Two designs are used:

| Design | Training basins | Evaluation basins |
|---|---|---|
| Temporal | all | all |
| Basin hold-out | 80 %, stratified by aridity class | remaining 20 % |

Observed discharge of the evaluation period is used only to score predictions. In the basin
hold-out design, no discharge of held-out basins enters any computation that produces a
prediction.

## 2. Baseline model

The LSTM follows Kratzert et al. (2018, 2019): dynamic and static inputs are concatenated at
every time step, input sequences span 180 days, the network has 128 hidden units, an initial
forget-gate bias of 5 and dropout of 0.2 on the last hidden state. Training minimizes the
basin-normalized NSE loss

$$
\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B}\frac{\left(\hat Q_i - Q_i\right)^2}{\left(s_{b(i)} + \epsilon\right)^2},
\qquad \epsilon = 0.1,
$$

where $s_{b(i)}$ is the standard deviation of training-period discharge in the basin of
sample $i$. Optimization uses Adam (learning rate $5\times10^{-3}$, reduced to
$5\times10^{-4}$ at epoch 11 and $10^{-4}$ at epoch 21), gradient-norm clipping at 1, batch
size 512 and 30 epochs.

## 3. Basin context

For each basin $b$, the quantities used by the rules are computed from training-period
observed discharge $Q$, precipitation $P$ and long-term potential evapotranspiration $E_p$:

$$
q_{f,b} = \max\!\left(q_{abs},\, f\,\overline{Q}\right), \qquad
h_b = Q_{0.9}, \qquad
\phi_b = E_p / \overline{P}, \qquad
\varepsilon_b = 1 - \overline{Q} / \overline{P},
$$

with $q_{abs} = 0.01$ mm d⁻¹ and $f = 0.05$. The Fu (1981) parameter $\omega_b$ solves
$F(\phi_b;\omega_b) = \varepsilon_b$, where

$$
F(\phi;\omega) = 1 + \phi - \left(1 + \phi^{\omega}\right)^{1/\omega}, \qquad \omega \in (1, 20].
$$

$\omega_b$ is undefined when $\varepsilon_b$ lies outside the range of $F$. For basins
without usable observations, $\omega_b$ is the median of training basins in the same aridity
class.

## 4. Auditing rules

Each rule is evaluated on explicit evaluation units; its rate $\rho_{b,r}$ is the number of
violating units divided by the number of evaluable units.

| Rule | Evaluation unit | Violation | Class | Evidence |
|---|---|---|---|---|
| R0 Negative flow | day | $\hat Q_t < 0$ | Physical impossibility | observation-free |
| R1 Extreme ratio | day with $Q_t > q_{f,b}$ | $\hat Q_t/Q_t < 0.2$ or $> 5$ | Magnitude | observation-referenced |
| R2 Zero-flow collapse | day with $Q_t > q_{f,b}$ | $\hat Q_t < 0.01$ mm d⁻¹ | Magnitude | observation-referenced |
| R3 High relative error | day with $Q_t > q_{f,b}$ | $\lvert\hat Q_t - Q_t\rvert / Q_t > 1$ | Magnitude | observation-referenced |
| R4 Peak timing | observed peak event | $\lvert\ell\rvert > 2$ days | Timing | observation-referenced |
| R5 Water limit | water year (≥ 330 days) | $\overline{\hat Q} > \overline{P}$ | Budget | observation-free |
| R6 Budyko consistency | five-water-year window (≥ 1650 days) | $\hat\varepsilon > \phi$, or $\hat\varepsilon \ge 0$ and $\lvert\hat\varepsilon - F(\phi;\omega_b)\rvert > 0.15$ | Budget | observation-free |

Observed peaks are local maxima of $Q$ with $Q \ge h_b$, separated by at least 30 days; the
lag is $\ell = \arg\max_{|k|\le 10}\hat Q_{t+k}$. For R6,
$\hat\varepsilon = 1 - \overline{\hat Q}/\overline{P}$ and $\phi = E_p/\overline{P}$ over the
window; windows with $\hat\varepsilon < 0$ breach the water limit and are counted by R5
only.

R5 is not applied where the training-period observed runoff ratio is at least 1, and the
curve condition of R6 is not applied where $\omega_b$ is undefined. A day that violates
several daily rules is attributed to the most fundamental one in the order R0, R2, R1, R3.
The observation-free rules are also applied to observed discharge to estimate their
false-alarm rate.

## 5. Violation burden

$$
V_b = \frac{\sum_r w_r\,\rho_{b,r}}{\sum_r w_r}, \qquad w_r = \frac{1}{n_{c(r),b}},
$$

where $n_{c,b}$ is the number of rules of class $c$ available in basin $b$, so that each
violation class carries equal weight; rules that are not applicable or have no evaluable unit
are excluded. Indices by evidence family are reported separately:

$$
V^{phys}_b = \operatorname{mean}_{r \in \{R0, R5, R6\}} \rho_{b,r}, \qquad
V^{err}_b = \operatorname{mean}_{r \in \{R1, R2, R3, R4\}} \rho_{b,r}.
$$

## 6. Knowledge graph

The graph is expressed in RDF and reuses SOSA/SSN for observations, PROV-O for provenance
and SKOS for stratification concepts.

**Rule execution.** Days, peak events and water-balance windows are represented as
`hk:DailyRecord`, `hk:PeakEvent`, `hk:WaterYearWindow` and `hk:BudykoWindow`, subclasses of
`sosa:Observation`. SPARQL updates derive the constrained quantities on evaluable units, and
SHACL Core shapes state their admissible ranges. Each SHACL validation result is a violation
of the rule its shape represents.

**Inference.** SPARQL rules classify catchments from training-period properties (e.g.
`hk:UndercatchCatchment`, `hk:NonIdentifiableBudykoCatchment`); ontology axioms
(`hk:notApplicableTo`, `hk:curveNotApplicableTo`) then determine which rules apply to each
catchment. Root causes of co-occurring daily violations are derived with the same mechanism
(`hk:explainedRule`). Rule-specific violation classes are subclasses of a violation class and
of an evidence family, so aggregate counts follow from subclass entailment.

**Provenance.** Datasets, model checkpoints, prediction sets, audits, feedback signals and
corrections are PROV-O entities and activities annotated with their period and basin set.
An evaluated prediction set is flagged as leaking if it is transitively derived from
discharge observations overlapping its own period or, in the hold-out design, from discharge
of held-out basins.

**Analog retrieval.** Donor catchments are ranked by personalized PageRank on the graph
linking catchments to their aridity class, land-cover class, hydrologic region and attribute
quantile levels.

The schema uses RDFS/OWL; inference is rule-based and validation is performed under the
closed-world assumption.

## 7. Graph-guided fine-tuning

Fine-tuning starts from the baseline weights and runs for 3 epochs at a constant learning
rate of $5\times10^{-4}$ with the baseline optimization step. Four configurations are
compared:

| Configuration | Basin sampling | Additional inputs |
|---|---|---|
| Control | uniform | none |
| Curriculum | weighted | none |
| Profile | uniform | violation profile |
| Curriculum + profile | weighted | violation profile |

Sampling weights are $w_b = \max(s_b,\, 0.1\,\max_{b'} s_{b'})$ with
$s_b = \operatorname{mean}_{r\in\{R0,\dots,R3\}}\rho_{b,r}$. The first epoch uses the
training-period audit of the baseline; later epochs use the violations detected during the
preceding epoch. The violation profile $(\rho_{b,r})_r$ of the training-period audit (all
rules in the temporal design; R0, R5 and R6 in the hold-out design) is appended to the static
inputs, with input weights initialized to zero. Both signals are read from the knowledge
graph after a provenance query confirms that they derive from training-period data only.
Neither signal enters the loss.

## 8. Post-hoc correction

Corrections are triggered only by observation-free rules. Clipping sets
$\hat Q_t \leftarrow \max(\hat Q_t, 0)$ on R0 days. Analog correction replaces $\hat Q_t$ on
R0 days by

$$
\hat Q_t \leftarrow \max\!\left(0,\; \tilde Q_b \cdot \operatorname{median}_{d \in A_b}\frac{\hat Q_{d,t}}{\tilde Q_d}\right),
$$

where $A_b$ is the set of the $k = 5$ analogs of basin $b$ with $\hat Q_{d,t} \ge 0$ and
$\tilde Q$ is the median simulated discharge.

## 9. Evaluation

Skill is measured by the Kling–Gupta efficiency and its components (Gupta et al., 2009) on
days with both series defined. The relation between skill and violation burden is quantified
by Spearman and partial rank correlations controlling for the aridity index. Configurations
are compared by the share of basins in each quadrant of $(\Delta\mathrm{KGE}, \Delta V)$,
Wilcoxon signed-rank tests, bootstrap 95 % intervals of mean changes by aridity class and
the mean and standard deviation over three seeds. Runoff ratio, flow-duration-curve
mid-segment slope, baseflow index (Lyne–Hollick filter, α = 0.925), zero-flow frequency and
the 5th and 95th flow percentiles serve as diagnostics independent of the rules.

## References

- Addor, N., Newman, A. J., Mizukami, N., & Clark, M. P. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. *Hydrology and Earth System Sciences*, 21, 5293–5313.
- Fu, B. P. (1981). On the calculation of the evaporation from land surface. *Scientia Atmospherica Sinica*, 5, 23–31.
- Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria. *Journal of Hydrology*, 377, 80–91.
- Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *Hydrology and Earth System Sciences*, 22, 6005–6022.
- Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. (2019). Toward improved predictions in ungauged basins: Exploiting the power of machine learning. *Water Resources Research*, 55, 11344–11354.
- Newman, A. J., et al. (2015). Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA. *Hydrology and Earth System Sciences*, 19, 209–223.