#!/usr/bin/env python3
"""Download benchmark datasets to a local directory.

Usage:
    uv run python -m pdm.benchmarks.download <base_dir> <benchmark_name>

Examples:
    uv run python -m pdm.benchmarks.download ./benchmark_data all
    uv run python -m pdm.benchmarks.download ./benchmark_data cmapss
    uv run python -m pdm.benchmarks.download ./benchmark_data ai4i
    uv run python -m pdm.benchmarks.download ./benchmark_data battery

Datasets are downloaded into <base_dir>/<benchmark_name>/ and prepared
as raw_train.csv + raw_test.csv via the benchmark loaders.
"""
import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BENCHMARKS = ["cmapss", "ai4i", "battery", "smap"]

# Download URLs and extraction logic per dataset
DATASET_INFO = {
    "cmapss": {
        "url": "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
        "description": "NASA C-MAPSS Turbofan Engine Degradation (FD001)",
        "format": "zip",
        "files_needed": ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"],
    },
    "ai4i": {
        "url": "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        "description": "AI4I 2020 Predictive Maintenance Dataset (UCI)",
        "format": "zip",
        "files_needed": ["ai4i2020.csv"],
    },
    "battery": {
        "url": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
        "description": "NASA PCoE Li-Ion Battery Aging Dataset (charge/discharge cycles at different temperatures)",
        "format": "zip",
        "files_needed": [],  # All .mat files in the zip are relevant
    },
    "smap": {
        "description": "NASA SMAP Spacecraft Telemetry Anomaly Detection (KDD 2018)",
        "format": "npy",
        "base_url": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/SMAP",
        "files": ["SMAP_train.npy", "SMAP_test.npy", "SMAP_test_label.npy"],
    },
}


def download_file(url: str, dest: Path, description: str = "") -> None:
    """Download a file with progress indication."""
    print(f"  Downloading {description or url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Downloaded ({dest.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}\n  URL: {url}") from e


def download_cmapss(base_dir: Path) -> Path:
    """Download and prepare C-MAPSS FD001."""
    output_dir = base_dir / "cmapss"
    if _is_ready(output_dir):
        print(f"  ✓ cmapss already exists at {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    info = DATASET_INFO["cmapss"]

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "cmapss.zip"
        download_file(info["url"], zip_path, "C-MAPSS dataset")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Find the extracted files (may be in a subdirectory)
        raw_dir = Path(tmpdir)
        for needed in info["files_needed"]:
            candidates = list(raw_dir.rglob(needed))
            if candidates:
                shutil.copy2(candidates[0], output_dir / needed)

    # Run the loader to produce raw_train.csv, raw_test.csv
    from pdm.benchmarks.loaders import load_cmapss
    load_cmapss(output_dir, subset="FD001", output_dir=output_dir)
    print(f"  ✓ cmapss prepared at {output_dir}")
    return output_dir


def download_ai4i(base_dir: Path) -> Path:
    """Download and prepare AI4I 2020 dataset."""
    output_dir = base_dir / "ai4i"
    if _is_ready(output_dir):
        print(f"  ✓ ai4i already exists at {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    info = DATASET_INFO["ai4i"]

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "ai4i.zip"
        download_file(info["url"], zip_path, "AI4I 2020 dataset")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Find the CSV
        candidates = list(Path(tmpdir).rglob("*.csv"))
        if candidates:
            shutil.copy2(candidates[0], output_dir / "ai4i2020.csv")

    # Run the loader
    from pdm.benchmarks.loaders import load_ai4i
    load_ai4i(output_dir, output_dir=output_dir)
    print(f"  ✓ ai4i prepared at {output_dir}")
    return output_dir


def download_battery(base_dir: Path) -> Path:
    """Download and prepare the NASA PCoE Battery dataset.

    Downloads the zip from NASA's Prognostics Data Repository. The archive
    contains nested zips (one per battery group), each holding .mat files.
    We extract all .mat files and run the benchmark loader to produce
    survival-analysis format CSVs.
    """
    output_dir = base_dir / "battery"
    if _is_ready(output_dir):
        print(f"  ✓ battery already exists at {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    info = DATASET_INFO["battery"]

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "battery.zip"
        download_file(info["url"], zip_path, "NASA Battery dataset")

        # Outer zip contains nested zip files (one per battery group)
        with zipfile.ZipFile(zip_path, "r") as outer_zf:
            inner_zips = [m for m in outer_zf.namelist() if m.endswith(".zip")]
            if not inner_zips:
                raise RuntimeError("No inner zip files found in NASA Battery archive")

            print(f"  Extracting from {len(inner_zips)} battery group archives...")
            mat_count = 0
            for inner_name in inner_zips:
                # Extract inner zip to temp, then pull .mat files from it
                inner_path = Path(tmpdir) / Path(inner_name).name
                with outer_zf.open(inner_name) as src, open(inner_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                try:
                    with zipfile.ZipFile(inner_path, "r") as inner_zf:
                        mat_members = [m for m in inner_zf.namelist() if m.endswith(".mat")]
                        for member in mat_members:
                            filename = Path(member).name
                            target = output_dir / filename
                            if not target.exists():
                                with inner_zf.open(member) as msrc, open(target, "wb") as mdst:
                                    shutil.copyfileobj(msrc, mdst)
                                mat_count += 1
                except zipfile.BadZipFile:
                    print(f"  ⚠ Skipping corrupt archive: {Path(inner_name).name}")
                finally:
                    inner_path.unlink(missing_ok=True)

            if mat_count == 0:
                raise RuntimeError("No .mat files extracted from NASA Battery nested archives")
            print(f"  ✓ Extracted {mat_count} .mat battery files")

    # Remove any stale cache so the loader re-processes from .mat files
    cache = output_dir / "_battery_processed.csv"
    if cache.exists():
        cache.unlink()

    # Run the loader to parse .mat → survival CSVs
    from pdm.benchmarks.loaders import load_nasa_battery
    load_nasa_battery(output_dir, output_dir=output_dir)
    print(f"  ✓ battery prepared at {output_dir}")
    return output_dir


def download_smap(base_dir: Path) -> Path:
    """Download and prepare the NASA SMAP anomaly detection dataset.

    Downloads .npy files from Hugging Face (thuml/Time-Series-Library),
    then runs the loader to produce train/test CSVs with labels.
    """
    output_dir = base_dir / "smap"
    if _is_ready(output_dir):
        print(f"  ✓ smap already exists at {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    info = DATASET_INFO["smap"]
    base_url = info["base_url"]

    for filename in info["files"]:
        dest = output_dir / filename
        if not dest.exists():
            url = f"{base_url}/{filename}"
            download_file(url, dest, filename)

    # Run the loader to produce raw_train.csv, raw_test.csv
    from pdm.benchmarks.loaders import load_smap
    load_smap(output_dir, output_dir=output_dir)
    print(f"  ✓ smap prepared at {output_dir}")
    return output_dir


def _is_ready(output_dir: Path) -> bool:
    """Check if a benchmark dataset is already prepared."""
    return (output_dir / "raw_train.csv").exists() and (output_dir / "raw_test.csv").exists()


def is_available(base_dir: Path, name: str) -> bool:
    """Check if benchmark data exists at <base_dir>/<name>/."""
    return _is_ready(Path(base_dir) / name)


def ensure_available(base_dir: Path, name: str) -> Path:
    """Download benchmark data if not already present. Returns the data directory."""
    base_dir = Path(base_dir)
    downloaders = {
        "cmapss": download_cmapss,
        "ai4i": download_ai4i,
        "battery": download_battery,
        "smap": download_smap,
    }
    if name not in downloaders:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(downloaders.keys())}")
    return downloaders[name](base_dir)


def main():
    parser = argparse.ArgumentParser(description="Download PdM benchmark datasets")
    parser.add_argument("base_dir", type=Path, help="Base directory for benchmark data")
    parser.add_argument("benchmark", choices=BENCHMARKS + ["all"],
                       help="Which benchmark to download ('all' for everything)")
    args = parser.parse_args()

    args.base_dir.mkdir(parents=True, exist_ok=True)
    targets = BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    print(f"Download directory: {args.base_dir.resolve()}")
    print(f"Benchmarks: {targets}\n")

    for name in targets:
        print(f"[{name}]")
        try:
            ensure_available(args.base_dir, name)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        print()


if __name__ == "__main__":
    main()
