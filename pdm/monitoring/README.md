# Optional sensor flatline diagnostics

A sensor can continue delivering finite readings after a measurement channel
stops responding to a changing physical quantity. The `pdm.monitoring.flatline`
module detects suspiciously constant or near-constant **consecutive readings**,
independently of statistical distribution drift.

Run the check on one asset's ordered telemetry window before imputation,
aggregation that destroys sample order, or missing-feature zero-fill. Supply
explicit per-channel configuration based on instrumentation resolution and
expected operating conditions.

```python
from pdm.monitoring.flatline import FlatlineConfig, detect_flatlines

results = detect_flatlines(telemetry, {
    "motor_current_a": FlatlineConfig(tolerance=0.02, min_consecutive_samples=60),
    "bus_voltage_v": FlatlineConfig(tolerance=0.1, min_consecutive_samples=120),
})
for sensor in results:
    print(sensor.feature, sensor.status.value,
          sensor.longest_flatline_run, sensor.ongoing)
```

These numbers are illustrative, not universal electrical limits. The tolerance
is a maximum peak-to-peak range in the channel's own units, inclusive of the
boundary (subject to ordinary floating-point representation). The run length is
a count of consecutive supplied rows, not seconds. Do not combine unrelated
assets or sampling sessions. If timestamp gaps matter, partition the data into
continuous sessions first; this function does not infer missing timestamps.

| Result | Interpretation |
|---|---|
| `missing_column` | Configured channel absent |
| `insufficient_samples` | No consecutive valid run long enough for assessment |
| `no_flatline_detected` | Sufficient contiguous readings, no qualifying flatline |
| `suspected_flatline` | At least one qualifying near-constant run |

Null, non-numeric and non-finite values interrupt a run. A slow ramp cannot
qualify simply because adjacent values differ by less than the tolerance:
the entire run must stay within the configured range. The report includes
the longest valid run, longest flatline run, and trailing flatline run.
`ongoing` means the trailing run meets the configured minimum; a historical
flatline that recovered remains visible in the report.

A constant measurement, including zero current, can be entirely normal when
equipment is stopped or regulated. Enable this diagnostic only for channels
and operating regimes where variation is expected. It does not diagnose a
physical open circuit, ADC clipping, a frozen transmitter, or a specific
electrical fault. Correlate with operating state, independent measurements and
hardware diagnostics before taking action. It is an opt-in utility; it does not
change inference or existing drift alerts.
