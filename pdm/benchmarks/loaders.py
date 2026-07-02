"""Benchmark dataset loaders — produce common format (CSVs + DatasetMeta).

Each loader reads a known PdM dataset from a local folder and writes:
  ./data/raw_train.csv, ./data/raw_test.csv, ./data/dataset_meta.json
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pdm.data.dataset_schema import DatasetMeta


# ---------------------------------------------------------------------------
# C-MAPSS
# ---------------------------------------------------------------------------

def _is_cmapss(data_dir: Path) -> bool:
    return any((data_dir / f"train_FD00{i}.txt").exists() for i in range(1, 5))


def load_cmapss(data_dir: Path, subset: str = "FD001", output_dir: Path = Path("./data")) -> DatasetMeta:
    """Load C-MAPSS dataset and write to common format.

    Produces one row per (unit × cycle) with RUL target capped at 125.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    cols = ["unit_id", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]

    train_df = pd.read_csv(data_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=cols)
    test_df = pd.read_csv(data_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=cols)
    rul_df = pd.read_csv(data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["RUL"])

    # Compute RUL for training (capped at 125)
    max_cycles = train_df.groupby("unit_id")["cycle"].max().reset_index(name="max_cycle")
    train_df = train_df.merge(max_cycles, on="unit_id")
    train_df["RUL"] = (train_df["max_cycle"] - train_df["cycle"]).clip(upper=125)
    train_df.drop(columns=["max_cycle"], inplace=True)

    # Test: attach RUL to last cycle per unit (keep full trajectories for windowing)
    test_max = test_df.groupby("unit_id")["cycle"].max().reset_index(name="max_cycle")
    test_df = test_df.merge(test_max, on="unit_id")
    rul_map = dict(zip(range(1, len(rul_df) + 1), rul_df["RUL"].values))
    test_df["RUL"] = test_df.apply(
        lambda r: rul_map[int(r["unit_id"])] + int(r["max_cycle"]) - int(r["cycle"]), axis=1
    ).clip(upper=125)
    test_df.drop(columns=["max_cycle"], inplace=True)

    # Drop near-constant sensors
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    std = train_df[sensor_cols].std()
    useful = std[std > 0.01].index.tolist()
    feature_cols = [f"op{i}" for i in range(1, 4)] + useful

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "raw_train.csv", index=False)
    test_df.to_csv(output_dir / "raw_test.csv", index=False)

    n_test_units = test_df["unit_id"].nunique()
    meta = DatasetMeta(
        name=f"C-MAPSS {subset}",
        source="benchmark",
        formulation="rul",
        target_columns=["RUL"],
        feature_columns=feature_cols,
        entity_column="unit_id",
        time_column="cycle",
        split_strategy="per_unit",
        n_train=len(train_df),
        n_test=n_test_units,
        n_features=len(feature_cols),
        evaluation_protocol={"metric": "rmse", "secondary": "nasa_score", "rul_cap": 125},
        reference={"title": "C-MAPSS Aircraft Engine Simulator Data", "source": "NASA PCoE"},
        data_path={"train": str(output_dir / "raw_train.csv"), "test": str(output_dir / "raw_test.csv")},
    )
    meta.save(output_dir / "dataset_meta.json")
    return meta


# ---------------------------------------------------------------------------
# AI4I 2020
# ---------------------------------------------------------------------------

def _is_ai4i(data_dir: Path) -> bool:
    return (data_dir / "ai4i2020.csv").exists() or any(data_dir.glob("ai4i*.csv"))


def load_ai4i(data_dir: Path, output_dir: Path = Path("./data")) -> DatasetMeta:
    """Load AI4I 2020 Predictive Maintenance dataset."""
    from sklearn.model_selection import train_test_split

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    csv_path = data_dir / "ai4i2020.csv" if (data_dir / "ai4i2020.csv").exists() else next(data_dir.glob("ai4i*.csv"))
    df = pd.read_csv(csv_path)

    # Standardize column names
    rename = {
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rot_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "machine_failure",
    }
    df = df.rename(columns=rename)
    if "Type" in df.columns:
        df["type_encoded"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    feature_cols = ["air_temp", "process_temp", "rot_speed", "torque", "tool_wear"]
    if "type_encoded" in df.columns:
        feature_cols.append("type_encoded")

    keep_cols = feature_cols + ["machine_failure"]
    df = df[[c for c in keep_cols if c in df.columns]]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["machine_failure"])

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "raw_train.csv", index=False)
    test_df.to_csv(output_dir / "raw_test.csv", index=False)

    meta = DatasetMeta(
        name="AI4I 2020",
        source="benchmark",
        formulation="classification",
        target_columns=["machine_failure"],
        feature_columns=feature_cols,
        split_strategy="random_stratified",
        n_train=len(train_df),
        n_test=len(test_df),
        n_features=len(feature_cols),
        evaluation_protocol={"metric": "f1", "secondary": "precision,recall"},
        reference={"title": "AI4I 2020 Predictive Maintenance Dataset", "source": "UCI ML Repository"},
        data_path={"train": str(output_dir / "raw_train.csv"), "test": str(output_dir / "raw_test.csv")},
    )
    meta.save(output_dir / "dataset_meta.json")
    return meta


# ---------------------------------------------------------------------------
# NASA Battery
# ---------------------------------------------------------------------------

def _is_nasa_battery(data_dir: Path) -> bool:
    data_dir = Path(data_dir)
    return (data_dir / "_battery_processed.csv").exists() or bool(list(data_dir.glob("*.mat"))) or bool(list(data_dir.glob("*Battery*.zip")))


def load_nasa_battery(data_dir: Path, output_dir: Path = Path("./data")) -> DatasetMeta:
    """Load NASA Battery dataset for survival analysis.

    Constructs (duration, event) pairs from capacity degradation curves.
    """
    from sklearn.model_selection import train_test_split

    data_dir, output_dir = Path(data_dir), Path(output_dir)

    cache = data_dir / "_battery_processed.csv"
    if cache.exists():
        raw = pd.read_csv(cache)
    else:
        # Try to extract .mat files from zips if present
        mat_files = list(data_dir.glob("*.mat"))
        if not mat_files:
            zips = list(data_dir.glob("*Battery*.zip"))
            if zips:
                import zipfile
                for z in zips:
                    with zipfile.ZipFile(z, "r") as zf:
                        for name in zf.namelist():
                            if name.endswith(".mat") and not (data_dir / name).exists():
                                zf.extract(name, data_dir)
                mat_files = list(data_dir.glob("*.mat"))

        if mat_files:
            # Parse .mat files for capacity degradation data
            import scipy.io as sio
            records_raw = []
            for mat_path in sorted(mat_files):
                bid = mat_path.stem
                mat = sio.loadmat(str(mat_path))
                # NASA battery .mat files store cycle data in nested structures
                if bid in mat:
                    cycles = mat[bid]["cycle"][0, 0].flatten()
                    for i, cyc in enumerate(cycles):
                        try:
                            cap = float(cyc["data"][0, 0]["Capacity"][0, 0].flatten()[0])
                            records_raw.append({"battery_id": bid, "cycle": i, "capacity": cap})
                        except (KeyError, IndexError, ValueError):
                            continue
            if not records_raw:
                raise RuntimeError(
                    f"No capacity data extracted from {len(mat_files)} .mat files in {data_dir}. "
                    "Ensure the NASA Battery dataset is properly downloaded."
                )
            raw = pd.DataFrame(records_raw)
            raw.to_csv(cache, index=False)
        else:
            raise FileNotFoundError(
                f"No .mat files found in {data_dir}. "
                "Download the NASA Battery dataset first:\n"
                "  uv run python -m pdm.benchmarks.download <base_dir> battery"
            )

    # Build per-battery features + survival target
    records = []
    for bid in raw["battery_id"].unique():
        bdf = raw[raw["battery_id"] == bid].sort_values("cycle")
        caps = bdf["capacity"].values
        if len(caps) < 5:
            continue

        initial = caps[0]
        threshold = initial * 0.80
        below = np.where(caps < threshold)[0]
        duration = int(below[0]) if len(below) > 0 else len(caps)
        event = 1 if len(below) > 0 else 0
        if duration < 2:
            continue

        early_n = max(3, int(len(caps) * 0.2))
        early = caps[:early_n]
        records.append({
            "battery_id": bid,
            "initial_capacity": initial,
            "early_mean_cap": float(early.mean()),
            "early_std_cap": float(early.std()),
            "early_slope": float(np.polyfit(range(len(early)), early, 1)[0]),
            "early_min_cap": float(early.min()),
            "n_early_cycles": early_n,
            "duration": duration,
            "event": event,
        })

    df = pd.DataFrame(records).dropna()
    feature_cols = ["initial_capacity", "early_mean_cap", "early_std_cap", "early_slope", "early_min_cap", "n_early_cycles"]

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "raw_train.csv", index=False)
    test_df.to_csv(output_dir / "raw_test.csv", index=False)

    meta = DatasetMeta(
        name="NASA Battery",
        source="benchmark",
        formulation="survival",
        target_columns=["duration", "event"],
        feature_columns=feature_cols,
        entity_column="battery_id",
        split_strategy="random",
        n_train=len(train_df),
        n_test=len(test_df),
        n_features=len(feature_cols),
        evaluation_protocol={"metric": "concordance_index", "secondary": "brier_score"},
        reference={"title": "NASA Battery Dataset", "source": "NASA PCoE"},
        data_path={"train": str(output_dir / "raw_train.csv"), "test": str(output_dir / "raw_test.csv")},
    )
    meta.save(output_dir / "dataset_meta.json")
    return meta


# ---------------------------------------------------------------------------
# FEMTO / PRONOSTIA Bearing (PHM 2012)
# ---------------------------------------------------------------------------

def _is_femto(data_dir: Path) -> bool:
    """Detect FEMTO by Learning_set/Test_set or Bearing* directories with acc_* files."""
    if (data_dir / "Learning_set").exists():
        return True
    return any(
        f.is_dir() and f.name.startswith("Bearing") and list(f.glob("acc_*.csv"))
        for f in data_dir.iterdir() if f.is_dir()
    )


def load_femto(data_dir: Path, output_dir: Path = Path("./data")) -> DatasetMeta:
    """Load FEMTO/PRONOSTIA bearing dataset.

    Expects:
        Learning_set/Bearing1_1/, Bearing1_2/, ...
        Test_set/Bearing1_3/, Bearing1_4/, ...
    Each folder contains acc_NNNNN.csv (2 cols: horizontal, vertical acceleration)
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load_bearing(bearing_dir: Path) -> pd.DataFrame:
        files = sorted(bearing_dir.glob("acc_*.csv"))
        records = []
        for i, f in enumerate(files):
            acc = pd.read_csv(f, header=None, names=["horiz", "vert"])
            records.append({
                "file_idx": i,
                "horiz_rms": float(np.sqrt((acc["horiz"] ** 2).mean())),
                "vert_rms": float(np.sqrt((acc["vert"] ** 2).mean())),
                "horiz_peak": float(acc["horiz"].abs().max()),
                "vert_peak": float(acc["vert"].abs().max()),
                "horiz_kurtosis": float(acc["horiz"].kurtosis()),
                "vert_kurtosis": float(acc["vert"].kurtosis()),
                "horiz_std": float(acc["horiz"].std()),
                "vert_std": float(acc["vert"].std()),
                "horiz_crest": float(acc["horiz"].abs().max() / (np.sqrt((acc["horiz"] ** 2).mean()) + 1e-10)),
                "vert_crest": float(acc["vert"].abs().max() / (np.sqrt((acc["vert"] ** 2).mean()) + 1e-10)),
            })
        return pd.DataFrame(records)

    # Load training bearings (full run-to-failure)
    train_dfs = []
    train_dir = data_dir / "Learning_set"
    if train_dir.exists():
        for bearing_dir in sorted(d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith("Bearing")):
            df = _load_bearing(bearing_dir)
            df["unit_id"] = bearing_dir.name
            df["cycle"] = df["file_idx"]
            max_cycle = df["cycle"].max()
            df["RUL"] = (max_cycle - df["cycle"]).clip(upper=125)
            train_dfs.append(df)

    # Load test bearings
    test_dfs = []
    test_dir = data_dir / "Test_set"
    if test_dir.exists():
        for bearing_dir in sorted(d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("Bearing")):
            df = _load_bearing(bearing_dir)
            df["unit_id"] = bearing_dir.name
            df["cycle"] = df["file_idx"]
            df["RUL"] = np.nan
            test_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    feature_cols = [c for c in train_df.columns if c not in ("unit_id", "cycle", "RUL", "file_idx")]

    train_df.to_csv(output_dir / "raw_train.csv", index=False)
    test_df.to_csv(output_dir / "raw_test.csv", index=False)

    meta = DatasetMeta(
        name="FEMTO/PRONOSTIA Bearing", source="benchmark", formulation="rul",
        target_columns=["RUL"], feature_columns=feature_cols,
        entity_column="unit_id", time_column="cycle", split_strategy="per_unit",
        n_train=len(train_df), n_test=len(test_df), n_features=len(feature_cols),
        evaluation_protocol={"metric": "rmse"},
    )
    meta.save(output_dir / "dataset_meta.json")
    return meta


# ---------------------------------------------------------------------------
# XJTU-SY Bearing
# ---------------------------------------------------------------------------

def _is_xjtu_sy(data_dir: Path) -> bool:
    """Detect XJTU-SY by numbered condition directories containing Bearing*.csv files."""
    for name in ("Condition1", "Condition2", "Condition3", "1", "2", "3"):
        cond_dir = data_dir / name
        if cond_dir.exists() and any(f.name.startswith("Bearing") for f in cond_dir.iterdir()):
            return True
    return False


def load_xjtu_sy(data_dir: Path, output_dir: Path = Path("./data"),
                  train_bearings_per_condition: int = 3) -> DatasetMeta:
    """Load XJTU-SY bearing dataset.

    Expects: Condition{1,2,3}/Bearing{condition}_{idx}.csv
    Each CSV: 2 columns (horiz_accel, vert_accel), 32768 rows per file at 25.6kHz.
    Standard split: first 3 bearings per condition = train, last 2 = test.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _agg_file(filepath: Path) -> dict:
        df = pd.read_csv(filepath, header=None, names=["horiz", "vert"])
        return {
            "horiz_rms": float(np.sqrt((df["horiz"] ** 2).mean())),
            "vert_rms": float(np.sqrt((df["vert"] ** 2).mean())),
            "horiz_peak": float(df["horiz"].abs().max()),
            "vert_peak": float(df["vert"].abs().max()),
            "horiz_kurtosis": float(df["horiz"].kurtosis()),
            "vert_kurtosis": float(df["vert"].kurtosis()),
        }

    train_dfs, test_dfs = [], []
    for cond_name in ("Condition1", "Condition2", "Condition3"):
        cond_dir = data_dir / cond_name
        if not cond_dir.exists():
            cond_dir = data_dir / cond_name[-1]  # Try "1", "2", "3"
        if not cond_dir.exists():
            continue
        bearing_dirs = sorted(d for d in cond_dir.iterdir() if d.is_dir())
        for i, b_dir in enumerate(bearing_dirs):
            files = sorted(b_dir.glob("*.csv"))
            records = [_agg_file(f) for f in files]
            df = pd.DataFrame(records)
            df["unit_id"] = b_dir.name
            df["cycle"] = range(len(df))
            max_c = len(df) - 1
            df["RUL"] = (max_c - df["cycle"]).clip(upper=125)
            if i < train_bearings_per_condition:
                train_dfs.append(df)
            else:
                test_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    feature_cols = [c for c in train_df.columns if c not in ("unit_id", "cycle", "RUL")]

    train_df.to_csv(output_dir / "raw_train.csv", index=False)
    test_df.to_csv(output_dir / "raw_test.csv", index=False)

    meta = DatasetMeta(
        name="XJTU-SY Bearing", source="benchmark", formulation="rul",
        target_columns=["RUL"], feature_columns=feature_cols,
        entity_column="unit_id", time_column="cycle", split_strategy="per_unit",
        n_train=len(train_df), n_test=len(test_df), n_features=len(feature_cols),
        evaluation_protocol={"metric": "rmse"},
    )
    meta.save(output_dir / "dataset_meta.json")
    return meta


# ---------------------------------------------------------------------------
# N-CMAPSS (New CMAPSS, 2021)
# ---------------------------------------------------------------------------

def _is_ncmapss(data_dir: Path) -> bool:
    """Detect N-CMAPSS by .h5 files with N-CMAPSS or DS0 in name."""
    return any(
        f.suffix == ".h5" and ("N-CMAPSS" in f.name or "DS0" in f.name)
        for f in data_dir.iterdir() if f.is_file()
    )


def load_ncmapss(data_dir: Path, output_dir: Path = Path("./data"),
                  subset: str = "DS01") -> DatasetMeta:
    """Load N-CMAPSS dataset from HDF5 format.

    Expects: *{subset}*.h5 file with groups: dev_data, test_data, vali_data
    Each group: W (op conditions), X_s (sensors), X_v (virtual), T (time), Y (RUL), A (unit IDs)
    """
    import h5py

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_candidates = [f for f in data_dir.glob("*.h5") if subset in f.name or "N-CMAPSS" in f.name]
    if not h5_candidates:
        raise FileNotFoundError(f"No HDF5 file found for subset '{subset}' in {data_dir}")
    h5_file = h5_candidates[0]

    with h5py.File(h5_file, "r") as f:
        def _load_group(group_name: str) -> pd.DataFrame:
            g = f[group_name]
            W = pd.DataFrame(g["W"][:], columns=[f"w{i}" for i in range(g["W"].shape[1])])
            X_s = pd.DataFrame(g["X_s"][:], columns=[f"xs{i}" for i in range(g["X_s"].shape[1])])
            Y = g["Y"][:].flatten()
            A = g["A"][:].flatten()
            df = pd.concat([W, X_s], axis=1)
            df["unit_id"] = A.astype(int)
            df["RUL"] = Y
            df["cycle"] = df.groupby("unit_id").cumcount()
            return df

        train_df = _load_group("dev_data")
        test_df = _load_group("test_data")

    feature_cols = [c for c in train_df.columns if c not in ("unit_id", "cycle", "RUL")]

    train_df.to_csv(output_dir / "raw_train.csv", index=False)
    test_df.to_csv(output_dir / "raw_test.csv", index=False)

    meta = DatasetMeta(
        name=f"N-CMAPSS {subset}", source="benchmark", formulation="rul",
        target_columns=["RUL"], feature_columns=feature_cols,
        entity_column="unit_id", time_column="cycle", split_strategy="per_unit",
        n_train=len(train_df), n_test=len(test_df), n_features=len(feature_cols),
        evaluation_protocol={"metric": "rmse", "scoring": "nasa_phm08"},
    )
    meta.save(output_dir / "dataset_meta.json")
    return meta


# ---------------------------------------------------------------------------
# SMAP (NASA Spacecraft Telemetry Anomaly Detection)
# ---------------------------------------------------------------------------

def _is_smap(data_dir: Path) -> bool:
    data_dir = Path(data_dir)
    return (data_dir / "SMAP_train.npy").exists() or (data_dir / "SMAP_test.npy").exists()


def load_smap(data_dir: Path, output_dir: Path = Path("./data")) -> DatasetMeta:
    """Load NASA SMAP anomaly detection dataset.

    Expects .npy files: SMAP_train.npy, SMAP_test.npy, SMAP_test_label.npy
    Produces raw_train.csv (all normal) and raw_test.csv (with 'label' column).

    The data is concatenated across all 55 telemetry channels (25 dims each).
    Training data is assumed entirely normal. Test labels are binary per-timestep.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.load(data_dir / "SMAP_train.npy")  # (N_train, 25)
    test_data = np.load(data_dir / "SMAP_test.npy")    # (N_test, 25)
    test_labels = np.load(data_dir / "SMAP_test_label.npy")  # (N_test,)

    feature_cols = [f"feat_{i}" for i in range(train_data.shape[1])]

    # Build DataFrames
    train_df = pd.DataFrame(train_data, columns=feature_cols)
    test_df = pd.DataFrame(test_data, columns=feature_cols)
    test_df["label"] = test_labels.astype(int)

    train_df.to_csv(output_dir / "raw_train.csv", index=False)
    test_df.to_csv(output_dir / "raw_test.csv", index=False)

    meta = DatasetMeta(
        name="NASA SMAP",
        source="benchmark",
        formulation="anomaly_detection",
        target_columns=["label"],
        feature_columns=feature_cols,
        split_strategy="temporal",
        n_train=len(train_df),
        n_test=len(test_df),
        n_features=len(feature_cols),
        evaluation_protocol={"metric": "f1", "secondary": "precision,recall", "protocol": "point_adjust"},
        reference={"title": "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding",
                   "source": "NASA JPL / KDD 2018", "arxiv": "1802.04431"},
        data_path={"train": str(output_dir / "raw_train.csv"), "test": str(output_dir / "raw_test.csv")},
    )
    meta.save(output_dir / "dataset_meta.json")
    return meta


# ---------------------------------------------------------------------------
# Backblaze Hard Drive Failure (hdfail)
# ---------------------------------------------------------------------------

def _is_hdfail(data_dir: Path) -> bool:
    data_dir = Path(data_dir)
    return (data_dir / "hdfail.csv").exists()


def load_hdfail(data_dir: Path, output_dir: Path = Path("./data")) -> DatasetMeta:
    """Load Backblaze Hard Drive Failure (hdfail) dataset for survival analysis.

    Source: frailtySurv R package (originally from Backblaze Drive Stats).
    52,422 hard drives tracked over ~2 years.
    - 2,885 failures (5.5%), 49,537 censored (94.5%)
    - Features: temp (temperature °C), rsc (reallocated sectors),
                rer (read error rate), psc (pending sector count)
    - Cluster variable: model (drive model, used as frailty/group)

    Survival formulation: time = days operational, event = drive failure.
    Split: temporal (first 70% by entry time → train, last 30% → test).
    """
    from sklearn.model_selection import train_test_split

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "hdfail.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"hdfail.csv not found in {data_dir}. "
            "Download with: uv run python -m pdm.benchmarks.download <base_dir> hdfail"
        )

    df = pd.read_csv(csv_path)

    # Standardize column names (from R: serial, model, time, status, temp, rsc, rer, psc)
    expected_cols = {"serial", "model", "time", "status", "temp", "rsc", "rer", "psc"}
    if not expected_cols.issubset(set(df.columns)):
        # Try lowercase
        df.columns = [c.lower() for c in df.columns]

    # Rename for consistency with our survival format
    df = df.rename(columns={"time": "duration", "status": "event"})

    # Encode model as integer (categorical → numeric for tree models)
    model_map = {m: i for i, m in enumerate(sorted(df["model"].unique()))}
    df["model_encoded"] = df["model"].map(model_map)

    feature_cols = ["temp", "rsc", "rer", "psc", "model_encoded"]

    # Keep only relevant columns
    keep_cols = ["serial", "duration", "event"] + feature_cols
    df = df[[c for c in keep_cols if c in df.columns]].dropna()

    # Ensure types
    df["duration"] = df["duration"].astype(float)
    df["event"] = df["event"].astype(int)

    # Split: random stratified by event (to preserve event ratio in both splits)
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["event"]
    )

    train_df.to_csv(output_dir / "raw_train.csv", index=False)
    test_df.to_csv(output_dir / "raw_test.csv", index=False)

    meta = DatasetMeta(
        name="Backblaze Hard Drive Failure (hdfail)",
        source="benchmark",
        formulation="survival",
        target_columns=["duration", "event"],
        feature_columns=feature_cols,
        entity_column="serial",
        split_strategy="random_stratified",
        n_train=len(train_df),
        n_test=len(test_df),
        n_features=len(feature_cols),
        evaluation_protocol={"metric": "concordance_index", "secondary": "brier_score"},
        reference={
            "title": "Backblaze Hard Drive Failure Dataset (hdfail)",
            "source": "frailtySurv R package / Backblaze Drive Stats",
            "url": "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data",
            "paper": "Ahmed & Green (2024), Neural Computing & Applications 37:1089-1104",
        },
        data_path={"train": str(output_dir / "raw_train.csv"), "test": str(output_dir / "raw_test.csv")},
    )
    meta.save(output_dir / "dataset_meta.json")
    return meta


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY = {
    "cmapss": {"detector": _is_cmapss, "loader": load_cmapss},
    "ai4i": {"detector": _is_ai4i, "loader": load_ai4i},
    "nasa_battery": {"detector": _is_nasa_battery, "loader": load_nasa_battery},
    "femto": {"detector": _is_femto, "loader": load_femto},
    "xjtu_sy": {"detector": _is_xjtu_sy, "loader": load_xjtu_sy},
    "ncmapss": {"detector": _is_ncmapss, "loader": load_ncmapss},
    "smap": {"detector": _is_smap, "loader": load_smap},
    "hdfail": {"detector": _is_hdfail, "loader": load_hdfail},
}


def detect_and_load(data_dir: Path, output_dir: Path = Path("./data")) -> Optional[DatasetMeta]:
    """Try to auto-detect a known benchmark from folder contents."""
    data_dir = Path(data_dir)
    for name, entry in BENCHMARK_REGISTRY.items():
        if entry["detector"](data_dir):
            return entry["loader"](data_dir, output_dir=output_dir)
    return None
