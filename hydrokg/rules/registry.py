"""Rule metadata: violation class, evidence family, temporal scale, and whether the rule
reads observed discharge in the period being audited. The same facts are asserted in the
ontology (hydrokg/kg/resources/hydrokg.ttl); tests/test_kg.py checks they agree."""

from __future__ import annotations

from dataclasses import dataclass

PHYSICAL_IMPOSSIBILITY = "PhysicalImpossibility"
MAGNITUDE_FAILURE = "MagnitudeFailure"
TIMING_FAILURE = "TimingFailure"
BUDGET_FAILURE = "BudgetFailure"
CLASSES = [PHYSICAL_IMPOSSIBILITY, MAGNITUDE_FAILURE, TIMING_FAILURE, BUDGET_FAILURE]

OBSERVATION_FREE = "ObservationFree"          # physical-consistency evidence
OBSERVATION_REFERENCED = "ObservationReferenced"  # prediction-error evidence
FAMILIES = [OBSERVATION_FREE, OBSERVATION_REFERENCED]


@dataclass(frozen=True)
class RuleInfo:
    rule_id: str
    name: str
    violation_class: str
    family: str
    scale: str                    # daily | event | window
    online: bool                  # evaluable on a single training batch

    @property
    def uses_test_observations(self) -> bool:
        return self.family == OBSERVATION_REFERENCED


RULES = {
    "R0": RuleInfo("R0", "Negative flow", PHYSICAL_IMPOSSIBILITY, OBSERVATION_FREE, "daily", True),
    "R1": RuleInfo("R1", "Extreme ratio", MAGNITUDE_FAILURE, OBSERVATION_REFERENCED, "daily", True),
    "R2": RuleInfo("R2", "Zero-flow collapse", MAGNITUDE_FAILURE, OBSERVATION_REFERENCED, "daily", True),
    "R3": RuleInfo("R3", "High relative error", MAGNITUDE_FAILURE, OBSERVATION_REFERENCED, "daily", True),
    "R4": RuleInfo("R4", "Peak-timing error", TIMING_FAILURE, OBSERVATION_REFERENCED, "event", False),
    "R5": RuleInfo("R5", "Water-limit (mass balance)", BUDGET_FAILURE, OBSERVATION_FREE, "window", False),
    "R6": RuleInfo("R6", "Budyko consistency", BUDGET_FAILURE, OBSERVATION_FREE, "window", False),
}
RULE_IDS = list(RULES)
DAILY_RULES = [r for r, i in RULES.items() if i.scale == "daily"]
ONLINE_RULES = [r for r, i in RULES.items() if i.online]


def rules_in_class(violation_class: str) -> list[str]:
    return [r for r, i in RULES.items() if i.violation_class == violation_class]


def rules_in_family(family: str) -> list[str]:
    return [r for r, i in RULES.items() if i.family == family]
