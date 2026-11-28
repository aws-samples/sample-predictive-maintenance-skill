# Optional ADC saturation and signal clipping diagnostics

`pdm.monitoring.saturation` reports sustained occupancy of explicitly
configured lower and upper measurement rails. An analog signal beyond the
acquisition chain's usable range may yield repeated limit codes; rail
occupancy alone cannot prove clipping or distinguish a legitimate full-scale
measurement from a hardware fault.

Configure limits from the actual acquisition channel's encoding, scaling and
calibration, not from extrema observed in the training data. For example:

```python
from pdm.monitoring.saturation import SaturationConfig, detect_saturation

results = detect_saturation(telemetry, {
    # Example only: unsigned 12-bit codes with the full 0..4095 range exposed.
    "adc_code": SaturationConfig(
        lower_limit=0, upper_limit=4095,
        min_consecutive_samples=5, tolerance=0,
        rail_fraction_threshold=0.20,  # optional, illustrative
    ),
    # Example only: a calibrated bipolar voltage measurement.
    "bus_voltage_v": SaturationConfig(
        lower_limit=-10.0, upper_limit=10.0,
        min_consecutive_samples=5, tolerance=0.02,
    ),
})
for channel in results:
    print(channel.feature, channel.status.value,
          channel.lower.suspected, channel.upper.suspected,
          channel.rail_fraction, channel.occupancy_suspected,
          channel.ongoing)
```

The examples are illustrative, not universal hardware limits. Tolerance is
in the same units as the measured values. Rail bands are inclusive:
lower limit through lower limit plus tolerance, and upper limit minus
tolerance through upper limit. The bands must not overlap. Floating-point
engineering units are subject to ordinary representation precision.

| Result | Interpretation |
|---|---|
| `missing_column` | Configured channel absent; no observations assessed |
| `insufficient_samples` | No sufficiently long contiguous in-range window and no qualifying occupancy event |
| `no_saturation_detected` | Neither configured saturation criterion is met |
| `suspected_saturation` | Qualifying lower/upper rail run or optional aggregate occupancy criterion |

Both rails have independent counts, longest runs and trailing runs. An isolated
rail hit or rapid switching between rails does not establish sustained
same-rail occupancy, but remains visible in counts and `rail_fraction`. The
optional `rail_fraction_threshold` (disabled by default, greater than zero and
at most one) flags aggregate occupancy at or above the configured fraction
when the window contains at least `min_consecutive_samples` in-range readings,
even if they are not contiguous. This can flag repeated clipped peaks without
a sustained run. `occupancy_suspected` distinguishes this criterion from the
run-based criteria; it does not itself imply an ongoing run. The fraction
uses only finite in-range samples as its denominator and is undefined
(`None`) when there are none. Values outside the configured measurement limits
are counted separately (below or above range), not clipped or imputed.
Non-numeric, missing and non-finite samples are counted as invalid. Numeric
strings are accepted; timestamps, complex values and booleans are not analog
amplitudes. A present channel's counts partition its supplied rows; an absent
channel has no observed readings.

Invalid, out-of-range and opposite-rail samples interrupt sustained runs.
A historical rail event remains visible after recovery; `ongoing` means
the trailing same-rail run meets the configured minimum. The algorithm scans
each channel in linear time.

Analyze one asset and continuous ordered sampling session at a time, before
imputation, clipping, normalization or aggregation that hides original rail
codes. The consecutive-sample threshold is not a duration; split sessions at
known timestamp gaps. Zero may be a legitimate measurement, including at
the lower rail of a unipolar channel. Rail occupancy is a reason to investigate,
not a diagnosis of ADC saturation, amplifier clipping, short circuit or
transmitter failure. Check operating state and the signal chain independently.
This is an opt-in software diagnostic, not an automatic inference gate or
hardware safety alarm.
