from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG = {"data": {"raw_dir": "data/raw", "processed_dir": "data/processed"}, "model": {"path": "models/random_forest.pkl", "features": ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed"], "target": "aqi"}, "visualization": {"map_output": "reports/aqi_map.html", "timeseries_output": "reports/aqi_timeseries.html"}}


def load_config(path: Path | None = None) -> Dict[str, Any]:
    config_path = path or Path("config.yml")
    if not config_path.exists():
        config_path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False))
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
