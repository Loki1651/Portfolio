from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


@dataclass
class DataIngestionConfig:
    api_url: str
    api_key: Optional[str]
    raw_dir: Path


class AirQualityDataIngestor:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_latest_data(self, city: str, limit: int = 1000) -> Path:
        params = {"city": city, "limit": limit, "key": self.config.api_key}
        response = requests.get(self.config.api_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        df = pd.json_normalize(data.get("results", data))
        if df.empty:
            raise ValueError("No data returned from API")

        file_path = self.config.raw_dir / f"{city.lower().replace(' ', '_')}_latest.csv"
        df.to_csv(file_path, index=False)
        return file_path
