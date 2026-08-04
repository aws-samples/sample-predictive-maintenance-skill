"""PdM Edge Inference Component — Greengrass component for on-device predictive maintenance.

Loads a trained PDM model, reads sensor data from a pluggable source, runs inference,
and publishes predictions to AWS IoT Core via Greengrass IPC.

Usage (local testing with CSV replay):
    python main.py --local --model-dir ./model --source csv --csv-path ./data/device_sensors.csv --interval 5

On Greengrass, paths and config come from component configuration (IPC get_configuration).
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import pandas as pd

from sensor_sources import SensorSource, CsvReplaySource

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EdgeInferenceComponent:
    """Main inference loop — loads model, consumes sensor data, publishes predictions."""

    def __init__(
        self,
        model_dir: Path,
        sensor_source: SensorSource,
        interval: float,
        local_mode: bool = False,
    ):
        self.model_dir = model_dir
        self.sensor_source = sensor_source
        self.interval = interval
        self.local_mode = local_mode
        self.running = True
        self.ipc_client = None
        self.prediction_topic = None
        self.thing_name = "local-device"

    def load_model(self):
        sys.path.insert(0, str(Path(__file__).parent))
        from pdm.base import PDMModel

        meta_path = self.model_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json in {self.model_dir}")
        metadata = json.loads(meta_path.read_text())
        self.formulation = metadata["formulation"]
        self.feature_names = metadata.get("feature_names", [])

        model_class = PDMModel.get_model_class(self.formulation)
        self.model = model_class.load(self.model_dir)
        logger.info(f"Loaded {self.formulation} model from {self.model_dir}")

    def setup_ipc(self):
        if self.local_mode:
            logger.info("Running in local mode — predictions go to stdout")
            return

        from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
        self.ipc_client = GreengrassCoreIPCClientV2()
        self.thing_name = os.getenv("AWS_IOT_THING_NAME", "unknown-thing")

        try:
            config = self.ipc_client.get_configuration()
            self.prediction_topic = config.value.get(
                "PredictionTopic", f"things/{self.thing_name}/pdm/predictions"
            )
        except Exception:
            self.prediction_topic = f"things/{self.thing_name}/pdm/predictions"

        logger.info(f"IPC initialized — publishing to: {self.prediction_topic}")

    def publish_prediction(self, prediction: dict):
        payload = json.dumps(prediction, default=str)
        if self.local_mode:
            print(payload)
            return

        self.ipc_client.publish_to_iot_core(
            topic_name=self.prediction_topic,
            qos="1",
            payload=payload,
        )

    def predict(self, sensor_reading: dict) -> dict:
        features_df = pd.DataFrame([sensor_reading])
        if self.feature_names:
            features_df = features_df.reindex(columns=self.feature_names, fill_value=0)

        result = self.model.predict(features_df)
        predictions = result.predictions.iloc[0].to_dict()

        return {
            "device_id": self.thing_name,
            "timestamp": time.time(),
            "formulation": self.formulation,
            "predictions": predictions,
            "status": "success",
        }

    def run(self):
        self.load_model()
        self.setup_ipc()
        self.sensor_source.start()
        logger.info(f"Inference loop started (interval={self.interval}s, source={self.sensor_source.name})")

        try:
            while self.running:
                reading = self.sensor_source.read(timeout=self.interval * 2)
                if reading is None:
                    continue

                try:
                    prediction = self.predict(reading)
                    self.publish_prediction(prediction)
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
                    self.publish_prediction({
                        "device_id": self.thing_name,
                        "timestamp": time.time(),
                        "status": "error",
                        "error": str(e),
                    })
        finally:
            self.sensor_source.stop()
            logger.info("Inference loop stopped")

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="PdM Edge Inference Component")
    parser.add_argument("--local", action="store_true", help="Run in local mode (stdout, no IPC)")
    parser.add_argument("--model-dir", type=Path, help="Path to model directory")
    parser.add_argument("--source", type=str, default="csv", help="Sensor source type (default: csv)")
    parser.add_argument("--csv-path", type=Path, help="Path to sensor CSV (for csv source)")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between readings")
    args = parser.parse_args()

    base = Path(os.getenv("COMPONENT_DIR", Path(__file__).parent))
    model_dir = args.model_dir or base / "model"

    sensor_source = _build_source(args.source, args, base)

    component = EdgeInferenceComponent(
        model_dir=model_dir,
        sensor_source=sensor_source,
        interval=args.interval,
        local_mode=args.local,
    )

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        component.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    component.run()


def _build_source(source_type: str, args, base: Path) -> SensorSource:
    if source_type == "csv":
        csv_path = args.csv_path or _find_sensor_csv(base / "data")
        return CsvReplaySource(csv_path=csv_path, interval=args.interval)
    else:
        raise ValueError(
            f"Unknown sensor source: {source_type}. "
            f"Available: csv. See README.md for how to add custom sources."
        )


def _find_sensor_csv(data_dir: Path) -> Path:
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    if len(csv_files) == 1:
        return csv_files[0]
    for name in ("device_sensors.csv", "sensor_data.csv", "data.csv"):
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    return csv_files[0]


if __name__ == "__main__":
    main()
