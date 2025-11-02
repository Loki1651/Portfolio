from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def load_features_and_target(processed_dir: Path, prefix: str) -> Dict[str, pd.DataFrame | pd.Series]:
    features_path = processed_dir / f"{prefix}_features.csv"
    target_path = processed_dir / f"{prefix}_target.csv"

    if not features_path.exists() or not target_path.exists():
        raise FileNotFoundError("Processed feature/target files not found")

    features = pd.read_csv(features_path)
    target = pd.read_csv(target_path, squeeze=True)
    if target.shape[1] == 1:
        target = target.iloc[:, 0]

    return {"features": features, "target": target}
