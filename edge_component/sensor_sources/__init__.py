"""Sensor source abstraction for the PdM edge inference component.

A SensorSource provides sensor readings to the inference loop. Implement the
SensorSource interface to connect to any data source (OPC-UA, MQTT, Modbus,
file-based replay, etc.).
"""

from sensor_sources.base import SensorSource
from sensor_sources.csv_replay import CsvReplaySource

__all__ = ["SensorSource", "CsvReplaySource"]
