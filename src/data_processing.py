from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_TIMESTAMP_COL = "measurement_timestamp"


def load_raw_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing raw data file: {file_path}")
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame, timestamp_col: str = DEFAULT_TIMESTAMP_COL) -> Tuple[pd.DataFrame, pd.Series]:
    working_df = df.copy()

    if timestamp_col in working_df.columns:
        working_df[timestamp_col] = pd.to_datetime(working_df[timestamp_col], errors="coerce")
        working_df.sort_values(timestamp_col, inplace=True)

    working_df.replace({"None": np.nan, "": np.nan, "NaN": np.nan}, inplace=True)
    working_df = working_df.select_dtypes(include=[np.number, "datetime64[ns]"])

    working_df = working_df.interpolate(limit_direction="both")
    working_df.dropna(inplace=True)

    if working_df.empty:
        raise ValueError("No usable data after preprocessing")

    target_col = "aqi"
    if target_col not in working_df.columns:
        raise KeyError("Expected 'aqi' column for target variable")

    features = working_df.drop(columns=[target_col])
    target = working_df[target_col]
    return features, target


def save_processed_data(features: pd.DataFrame, target: pd.Series, processed_dir: Path, prefix: str) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed_dir / f"{prefix}_features.csv", index=False)
    target.to_csv(processed_dir / f"{prefix}_target.csv", index=False)
