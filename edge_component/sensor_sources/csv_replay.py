"""CSV Replay sensor source — replays recorded sensor data from a CSV file.

Reads sensor data row by row at a configurable interval, looping back
to the start when the file is exhausted. Useful for demos and testing.
"""

import logging
import queue
import threading
import time
from pathlib import Path

import pandas as pd

from sensor_sources.base import SensorSource

logger = logging.getLogger(__name__)


class CsvReplaySource(SensorSource):
    """Replays sensor readings from a CSV file at a fixed interval.

    Each row in the CSV becomes one sensor reading (dict of feature→value).
    When all rows are consumed, the replay loops back to the beginning.

    Args:
        csv_path: Path to the sensor data CSV file.
        interval: Seconds between readings.
    """

    def __init__(self, csv_path: Path, interval: float = 60.0):
        self.csv_path = Path(csv_path)
        self.interval = interval
        self._queue: queue.Queue = queue.Queue(maxsize=10)
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def name(self) -> str:
        return "csv-replay"

    def start(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Sensor CSV not found: {self.csv_path}")
        self._running = True
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

    def read(self, timeout: float) -> dict | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _replay_loop(self) -> None:
        df = pd.read_csv(self.csv_path)
        logger.info(f"CsvReplaySource loaded {len(df)} rows from {self.csv_path.name}")
        idx = 0
        while self._running:
            row = df.iloc[idx % len(df)]
            self._queue.put(row.to_dict())
            idx += 1
            if idx % len(df) == 0:
                logger.info("CsvReplaySource: looping back to start of dataset")
            time.sleep(self.interval)
