"""Benchmark suite for the pdm library.

Run benchmarks:
    uv run python -m pdm.benchmarks.benchmark ./data all
    uv run python -m pdm.benchmarks.benchmark ./data cmapss

Download data:
    uv run python -m pdm.benchmarks.download ./data all
    uv run python -m pdm.benchmarks.download ./data cmapss

Extract single-device sensor data (for edge inference):
    uv run python -m pdm.benchmarks.extract_device --input data/raw_test.csv --model-dir ./model --output edge_component/data/device_sensors.csv
"""
