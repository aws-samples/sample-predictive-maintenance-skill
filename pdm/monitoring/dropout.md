# Optional intermittent sensor dropout diagnostics

An unstable electrical connection, transmitter interruption, acquisition error
or communication loss may produce recurring unusable observations separated
by valid measurements. The opt-in `pdm.monitoring.dropout` module distinguishes
these patterns from sustained loss of usable readings.

```python
from pdm.monitoring.dropout import DropoutConfig, detect_dropouts

reports = detect_dropouts(telemetry, {
    "motor_current_a": DropoutConfig(
        min_recovered_episodes=3,
        sustained_run_samples=10,
    ),
    "bus_voltage_v": DropoutConfig(
        min_recovered_episodes=2,
        sustained_run_samples=5,
    ),
})
for channel in reports:
    print(channel.feature, channel.status.value, channel.dropout_fraction,
          channel.n_recovered_short_episodes, channel.ongoing_sustained)
```

The thresholds are illustrative sample counts, not hardware specifications
or elapsed durations. A dropout episode is a consecutive run of missing
(null) or invalid (non-real, non-numeric or non-finite) observations.
Finite real readings, including zero and negative values, and finite numeric
strings are valid. Missing and invalid readings are counted separately but
do not split an episode when they occur consecutively.

A short episode is counted as recovered and intermittent only when it is
bounded on both sides by observed valid readings and its length is below
the sustained threshold. Leading and trailing short gaps remain visible
but are not counted as fully observed intermittent episodes. Sustained
episodes meet or exceed the configured run length, including at window
boundaries. An entirely unusable window counts as one episode, not two.

| Status | Interpretation |
|---|---|
| `missing_column` | Configured channel absent; no readings observed |
| `empty_window` | Present channel with no supplied rows |
| `no_valid_samples` | All supplied observations unusable |
| `insufficient_samples` | Window too short for either configured pattern |
| `no_qualifying_dropout` | No configured episode threshold met |
| `suspected_intermittent` | Enough fully observed short recovered episodes |
| `suspected_sustained` | At least one sustained unusable run |
| `suspected_both` | Both criteria met |

The diagnostic flags are independent of the status; a no-valid-samples
window can also meet the sustained threshold. The report includes episode
counts, longest, leading and trailing runs, and flags for ongoing dropout
and ongoing sustained dropout. Dropout fraction uses the supplied rows as
the denominator; it is undefined for missing columns or empty windows.

Analyze one asset and one continuous ordered acquisition session at a time,
before imputation or zero-fill. This function detects unusable readings in
rows that are actually supplied. It cannot infer samples or timestamp rows
that were never recorded; if a known sampling schedule is available,
represent missing expected observations explicitly before analysis and avoid
joining unrelated sessions. Do not interpret sample counts as time without
a verified sampling cadence.

The output describes observed telemetry symptoms, not proof of connector
failure, a broken wire or another physical cause. An unflagged result
does not establish equipment health. The module is a software diagnostic,
not an automatic inference gate or a hardware safety alarm.
