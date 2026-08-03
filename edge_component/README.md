# PdM Edge Inference Component

AWS IoT Greengrass component for running predictive maintenance inference on edge devices. Publishes predictions to AWS IoT Core via MQTT.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Edge Device                                             │
│                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────┐ │
│  │ SensorSource │────►│ PDM Model    │────►│ IoT Core│ │
│  │ (pluggable)  │     │ (.predict()) │     │ (MQTT)  │ │
│  └──────────────┘     └──────────────┘     └─────────┘ │
└─────────────────────────────────────────────────────────┘
```

The component has three parts:
1. **Sensor Source** — provides sensor readings (pluggable interface)
2. **PDM Model** — runs inference using the standard `pdm` library
3. **Publisher** — sends predictions to IoT Core (or stdout in local mode)

## Quick Start (Local Testing)

```bash
# 1. Prepare model and data (done by deploy_edge.sh in production)
cp -r <trained-model-dir> edge_component/model/
cp -r pdm/ edge_component/pdm/
python scripts/extract_device_data.py --input data/raw_test.csv --output edge_component/data/device_sensors.csv

# 2. Run locally
cd edge_component
python main.py --local --model-dir ./model --source csv --csv-path ./data/device_sensors.csv --interval 5
```

## Sensor Sources

The component uses a pluggable sensor source interface. A source produces sensor readings (one dict per reading) that the model consumes.

### Built-in Sources

| Source | `--source` | Description |
|--------|-----------|-------------|
| CSV Replay | `csv` | Replays rows from a CSV file at a fixed interval. Loops forever. |

### Selecting a Source

```bash
# CSV replay (default)
python main.py --local --model-dir ./model --source csv --csv-path ./data/sensors.csv --interval 30
```

On Greengrass, the source type and parameters come from the component configuration in `recipe.yaml`.

---

## Implementing a Custom Sensor Source

To connect the component to a real sensor (OPC-UA, MQTT, Modbus, serial, REST API, etc.), implement the `SensorSource` interface.

### Step 1: Create a new file in `sensor_sources/`

```python
# sensor_sources/opcua_source.py
from sensor_sources.base import SensorSource


class OpcuaSource(SensorSource):
    """Reads sensor values from an OPC-UA server."""

    def __init__(self, endpoint: str, node_ids: list[str], interval: float = 1.0):
        self.endpoint = endpoint
        self.node_ids = node_ids
        self.interval = interval
        self._client = None

    @property
    def name(self) -> str:
        return "opcua"

    def start(self) -> None:
        from opcua import Client
        self._client = Client(self.endpoint)
        self._client.connect()

    def read(self, timeout: float) -> dict | None:
        import time
        time.sleep(self.interval)
        values = {}
        for node_id in self.node_ids:
            node = self._client.get_node(node_id)
            values[node_id] = node.get_value()
        return values

    def stop(self) -> None:
        if self._client:
            self._client.disconnect()
```

### Step 2: Register it in `sensor_sources/__init__.py`

```python
from sensor_sources.base import SensorSource
from sensor_sources.csv_replay import CsvReplaySource
from sensor_sources.opcua_source import OpcuaSource  # add this

__all__ = ["SensorSource", "CsvReplaySource", "OpcuaSource"]
```

### Step 3: Add it to `main.py`'s `_build_source()`

```python
def _build_source(source_type: str, args, base: Path) -> SensorSource:
    if source_type == "csv":
        csv_path = args.csv_path or _find_sensor_csv(base / "data")
        return CsvReplaySource(csv_path=csv_path, interval=args.interval)
    elif source_type == "opcua":
        return OpcuaSource(
            endpoint=args.opcua_endpoint,
            node_ids=args.opcua_nodes,
            interval=args.interval,
        )
    ...
```

### The `SensorSource` Interface

```python
class SensorSource(ABC):

    @property
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    def start(self) -> None:
        """Initialize connections. Called once before the inference loop."""
        ...

    def read(self, timeout: float) -> dict | None:
        """Return the next reading as {feature_name: value}, or None on timeout."""
        ...

    def stop(self) -> None:
        """Clean up. Called once when the component shuts down. Must be idempotent."""
        ...
```

### Contract

| Method | Called | Expected behavior |
|--------|--------|-------------------|
| `start()` | Once, before loop | Open connections, start background threads. Raise on failure. |
| `read(timeout)` | Repeatedly | Return `dict[str, float]` or `None`. Must not block longer than `timeout`. |
| `stop()` | Once, on shutdown | Close connections. Safe to call multiple times. |

### Reading Format

`read()` must return a dict mapping **feature names** to **numeric values**:

```python
{
    "air_temp": 300.1,
    "process_temp": 310.5,
    "rot_speed": 1503,
    "torque": 40.2,
    "tool_wear": 108,
    "type_encoded": 1
}
```

- Keys should match the model's expected features (from `model/metadata.json["feature_names"]`)
- Extra keys are ignored
- Missing keys are filled with 0
- Values must be numeric (int or float)

### Example Implementations

| Use case | Source | Notes |
|----------|--------|-------|
| Demo/testing | `CsvReplaySource` | Replays a CSV file in a loop |
| Industrial equipment | OPC-UA client | See `opcua_source.py` example above |
| MQTT broker | Local MQTT subscriber | Subscribe to sensor topic, buffer last reading |
| Modbus RTU/TCP | Modbus client | Poll register addresses at interval |
| REST API | HTTP polling | GET endpoint, parse JSON response |
| Serial port | pyserial reader | Read from /dev/ttyUSB0, parse protocol |
| Greengrass local pub/sub | IPC subscriber | Receive from another Greengrass component |

---

## Output Format

Each prediction published to IoT Core is a JSON object:

```json
{
  "device_id": "pdm-edge-001",
  "timestamp": 1722345678.123,
  "formulation": "classification",
  "predictions": {
    "machine_failure_pred": 0,
    "machine_failure_proba": 0.047
  },
  "status": "success"
}
```

The `predictions` field depends on the model formulation:

| Formulation | Fields |
|-------------|--------|
| `anomaly_detection` | `anomaly_score`, `is_anomaly` |
| `classification` | `machine_failure_pred`, `machine_failure_proba` |
| `multilabel` | `{label}_pred`, `{label}_proba` (per label) |
| `rul` | `RUL_pred` |
| `survival` | Survival function values |

## Greengrass Deployment

See `infrastructure/edge/README.md` for CDK deployment and `scripts/deploy_edge.sh` for automated deployment.
