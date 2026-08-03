"""SensorSource abstract base class.

All sensor sources must implement this interface. The inference loop calls:
  1. source.start()       — once, before the loop begins
  2. source.read(timeout) — repeatedly, to get the next sensor reading
  3. source.stop()        — once, when the loop exits (SIGTERM or error)
"""

from abc import ABC, abstractmethod


class SensorSource(ABC):
    """Abstract interface for sensor data providers.

    A sensor source produces one reading at a time as a dict mapping
    feature names to numeric values. The inference loop consumes these
    readings and passes them to the model.

    Implementations can be:
      - File-based (CSV replay for demos/testing)
      - Protocol-based (OPC-UA, MQTT, Modbus)
      - Hardware-based (GPIO, serial, USB sensors)
      - API-based (REST endpoints, cloud data feeds)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging (e.g., 'csv-replay', 'opcua')."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Initialize the source (open connections, start background threads).

        Called once before the inference loop begins. Raise an exception
        if the source cannot be initialized (e.g., connection refused).
        """
        ...

    @abstractmethod
    def read(self, timeout: float) -> dict | None:
        """Return the next sensor reading, or None if no data is available.

        Args:
            timeout: Maximum seconds to wait for a reading. Return None
                     if no reading arrives within this time.

        Returns:
            A dict mapping feature names (str) to numeric values (float/int),
            or None if no data is available within the timeout.

        The returned dict should contain at minimum the features expected
        by the model (listed in model/metadata.json["feature_names"]).
        Extra keys are ignored; missing keys are filled with 0.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Clean up resources (close connections, stop threads).

        Called once when the inference loop exits. Must be idempotent
        (safe to call multiple times).
        """
        ...
