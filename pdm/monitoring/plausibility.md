# Optional multi-sensor signal plausibility checks

`pdm.monitoring.plausibility` compares synchronized observations against
explicit physical relationships. The residual can reveal disagreement even
when individual channels are finite and within their measurement ranges.

```python
from pdm.monitoring.plausibility import (
    PlausibilityConfig, Relation, check_plausibility,
)

reports = check_plausibility(telemetry, {
    "dc_power": PlausibilityConfig(
        relation=Relation.DC_POWER,
        measured="power_w",
        inputs=("voltage_v", "current_a"),
        absolute_tolerance=2.0,
        relative_tolerance=0.05,
        min_consecutive_samples=5,
    ),
    "passive_resistor": PlausibilityConfig(
        relation=Relation.OHMS_LAW,
        measured="resistor_voltage_v",
        inputs=("resistance_ohm", "resistor_current_a"),
        absolute_tolerance=0.1,
        relative_tolerance=0.02,
        min_consecutive_samples=5,
    ),
    "redundant_temperature": PlausibilityConfig(
        relation=Relation.REDUNDANT,
        measured="temperature_a_c",
        inputs=("temperature_b_c",),
        absolute_tolerance=0.5,
        relative_tolerance=0,
        min_consecutive_samples=5,
    ),
})
for report in reports:
    print(report.rule, report.status.value, report.violation_fraction,
          report.max_absolute_residual, report.ongoing)
```

| Relation | Expected value | Assumptions |
|---|---|---|
| DC power | Voltage times current | Synchronized DC measurements, same circuit point and consistent sign convention |
| Ohm's law | Resistance times current | Passive ohmic element, voltage across and current through that element, known resistance at operating temperature |
| Redundant | Independent reference measurement | Same physical quantity, units, calibration and location or justified equivalence |

The DC power rule is not an AC real-power or three-phase power formula.
Do not multiply RMS voltage and RMS current and interpret the result as
real AC power without the necessary power-factor and topology information.
The passive-resistor rule is not a general motor or nonlinear load model.
Use independently measured or otherwise justified inputs; deriving both
sides from the same measurements cannot provide independent validation.

The absolute residual is compared against
`absolute_tolerance + relative_tolerance * abs(expected)`. A sample
violates only when its residual exceeds that bound. Absolute tolerance is
in measured units, relative tolerance is a dimensionless fraction (0.05
means 5%), and both are nonnegative. The absolute term handles expected
values near zero. The example thresholds are illustrative, not hardware
specifications; choose them from uncertainty, calibration and operating
conditions. Floating-point boundaries are subject to representation error.

| Status | Interpretation |
|---|---|
| `missing_columns` | Required channel(s) absent |
| `insufficient_samples` | No contiguous evaluated run long enough for assessment |
| `no_violation` | No sustained qualifying residual violation |
| `suspected_inconsistency` | At least one qualifying consecutive violation run |

Missing, invalid and non-finite readings or non-finite calculations are
unusable; negative resistance is outside the passive-resistor rule domain.
Unusable rows break runs and are counted separately. Valid negative and
zero DC measurements are preserved. The violation fraction uses evaluated
rows only and is undefined when none are evaluated. Isolated violations
are counted even if they do not meet the persistence threshold. A recovered
event remains visible; ongoing means the trailing run meets the threshold.

Use one asset, one continuous ordered acquisition session, compatible units
and synchronized measurements before imputation, clipping or normalization.
The utility does not align timestamps, infer absent rows, estimate unknown
resistance, or automatically select an operating regime. Sample counts are
not elapsed time. A residual violation indicates inconsistency with the
configured relationship, not which sensor or component failed; a satisfied
relationship cannot prove sensor or equipment health. This is an opt-in
software diagnostic, not a safety interlock or automatic inference gate.
