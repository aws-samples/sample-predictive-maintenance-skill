"""Dataset metadata schema — common format produced by both Path A and Path B."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DatasetMeta:
    """Machine-readable dataset metadata consumed by all downstream phases."""

    name: str
    source: str  # "s3" or "benchmark"
    formulation: str  # "rul", "classification", "multilabel", "survival", "anomaly_detection"
    target_columns: list[str]
    feature_columns: list[str]
    n_train: int
    n_test: int
    n_features: int

    # Optional structural info
    entity_column: Optional[str] = None
    time_column: Optional[str] = None
    split_strategy: str = "temporal"
    evaluation_protocol: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    data_path: dict = field(default_factory=lambda: {"train": "./data/raw_train.csv", "test": "./data/raw_test.csv"})
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # Path A extras
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    split_date: Optional[str] = None
    label_positive_rates: Optional[dict] = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "DatasetMeta":
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def validate(self) -> None:
        """Check consistency between metadata and actual CSV files."""
        train_path = Path(self.data_path["train"])
        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        train = pd.read_csv(train_path, nrows=5)
        for col in self.target_columns:
            if col not in train.columns:
                raise ValueError(f"Target column '{col}' not in train data. Found: {list(train.columns)}")
