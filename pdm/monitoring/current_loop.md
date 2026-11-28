# Optional 4–20 mA current-loop diagnostics

Industrial transmitters can encode both a process measurement and alarm
information in the same analog current signal. The opt-in
`pdm.monitoring.current_loop` utility separates measurement, transition and
alarm bands, and tracks sustained same-direction alarm events.

Use measured current in milliamperes, not a normalized process percentage or
a clipped 4–20 mA value. Select thresholds from the transmitter and input
module documentation and their configured alarm levels. An illustrative
NE43-style profile is provided; it is not a claim of device compliance.

```python
from pdm.monitoring.current_loop import CurrentLoopConfig, detect_current_loop

reports = detect_current_loop(telemetry, {
    "pressure_loop_ma": CurrentLoopConfig.ne43_example(min_consecutive_samples=3),
    "temperature_loop_ma": CurrentLoopConfig(
        low_alarm_ma=3.5, low_measurement_ma=3.8,
        nominal_min_ma=4.0, nominal_max_ma=20.0,
        high_measurement_ma=20.5, high_alarm_ma=22.0,
        min_consecutive_samples=5,
    ),
})
for channel in reports:
    print(channel.feature, channel.status.value, channel.n_measurement,
          channel.low_alarm_suspected, channel.high_alarm_suspected,
          channel.ongoing)
```

The example profile uses the following explicit boundary convention:

| Observed current (mA) | Classification |
|---|---|
| Negative, non-finite or non-numeric | Invalid |
| 0 through 3.6 | Low alarm |
| Greater than 3.6, less than 3.8 | Low transition |
| 3.8 through values below 4.0 | Underrange measurement |
| 4.0 through 20.0 | Nominal measurement |
| Greater than 20.0 through 20.5 | Overrange measurement |
| Greater than 20.5, less than 21.0 | High transition |
| 21.0 and above | High alarm |

The example includes the measurement endpoints 3.8 and 20.5 in extended
measurement bands. Manufacturer and input-module boundary conventions can
differ; check them before applying this profile. See
[Beckhoff's NAMUR basic information](https://infosys.beckhoff.com/content/1033/ioanalogmanual/3904850187.html)
for a manufacturer description of NE43 measurement and failure ranges.
All six thresholds must be strictly increasing and nonnegative.
The minimum run length is an explicit engineering setting, not part of NE43.

A missing channel has no observed readings. Invalid readings are distinct
from zero current, which can be a low-alarm indication in this unipolar
profile. The report retains per-band counts and longest/trailing runs.
`n_valid` counts finite nonnegative current observations; `n_measurement`
counts only nominal and extended-range measurements. Transition or alarm
readings are not treated as process measurements.

A sustained alarm requires consecutive observations in the same alarm band.
Other bands and invalid observations interrupt a run. A historical alarm
is retained after recovery; `ongoing` identifies a qualifying trailing run.
A window can have no sustained alarm while containing transition readings,
short alarms or no usable process measurements. That status does not
establish equipment health.

Analyze one asset and one continuous ordered sampling session before
imputation, clipping, or conversion to process units. Split at known
timestamp gaps; the sample threshold is not a duration. Numeric strings
are accepted; timestamps, complex values and booleans are invalid.
An alarm-band observation may be consistent with transmitter diagnostics,
loop power loss or wiring faults, but software cannot establish the
underlying physical cause from current alone. Check device status,
configuration and independent measurements. This diagnostic is not a
safety interlock and is not automatically wired into inference.
