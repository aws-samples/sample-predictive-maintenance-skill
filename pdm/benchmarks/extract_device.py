"""Extract sensor data for a single device from a PdM dataset.

Produces a clean CSV containing only the feature columns the model expects,
suitable for feeding into the edge inference component's CsvReplaySource.

Usage:
    uv run python -m pdm.benchmarks.extract_device --input data/raw_test.csv --model-dir fault_prediction/baseline/model --output edge_component/data/device_sensors.csv
    uv run python -m pdm.benchmarks.extract_device --input data/raw_test.csv --model-dir anomaly_detection/baseline/model --output edge_component/data/device_sensors.csv --max-rows 200
    uv run python -m pdm.benchmarks.extract_device --input benchmark_data/cmapss --output edge_component/data/device_sensors.csv --device-id 3

For datasets with an entity column (e.g., unit_id), use --device-id to select
a specific device. For flat datasets (no entity column), all rows are treated
as belonging to a single device.
"""

import argparse
import json
import sys
from pathlib import Path


def extract_device_data(
    input_path: Path,
    model_dir: Path,
    output_path: Path,
    device_id: str | None = None,
    max_rows: int | None = None,
) -> None:
    import pandas as pd

    input_path, model_dir, output_path = Path(input_path), Path(model_dir), Path(output_path)

    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        meta_path = model_dir / "dataset_meta.json"
    if not meta_path.exists():
        print(f"Error: No metadata.json or dataset_meta.json found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    metadata = json.loads(meta_path.read_text())
    feature_names = metadata.get("feature_names") or metadata.get("feature_columns", [])
    entity_col = metadata.get("entity_column")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    if entity_col and entity_col in df.columns:
        devices = df[entity_col].unique()
        if device_id is not None:
            target_id = type(devices[0])(device_id)
            if target_id not in devices:
                print(f"Error: device_id={device_id} not found. Available: {devices[:10].tolist()}", file=sys.stderr)
                sys.exit(1)
            df = df[df[entity_col] == target_id]
        else:
            df = df[df[entity_col] == devices[0]]
        print(f"Selected device: {df[entity_col].iloc[0]} ({len(df)} rows)")

    available_features = [f for f in feature_names if f in df.columns]
    missing_features = [f for f in feature_names if f not in df.columns]

    if missing_features:
        print(f"Warning: {len(missing_features)} features not in input data: {missing_features[:5]}", file=sys.stderr)

    if not available_features:
        print("Error: No matching feature columns found between model and input data", file=sys.stderr)
        sys.exit(1)

    result = df[available_features].copy()

    if max_rows and len(result) > max_rows:
        result = result.head(max_rows)
        print(f"Capped to {max_rows} rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Wrote {len(result)} rows x {len(available_features)} features to {output_path}")
    print(f"  Features: {available_features}")
    size_kb = output_path.stat().st_size / 1024
    print(f"  File size: {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract single-device sensor data for edge inference"
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to raw test CSV (e.g., data/raw_test.csv)")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to model directory (for metadata.json)")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path (e.g., edge_component/data/device_sensors.csv)")
    parser.add_argument("--device-id", type=str, default=None, help="Device/unit ID to extract (default: first device)")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows (default: all)")
    args = parser.parse_args()

    extract_device_data(
        input_path=args.input,
        model_dir=args.model_dir,
        output_path=args.output,
        device_id=args.device_id,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
