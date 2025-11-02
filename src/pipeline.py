from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import load_config
from .data_ingestion import AirQualityDataIngestor, DataIngestionConfig
from .data_processing import load_raw_data, preprocess_data, save_processed_data
from .modeling import AQIPredictor, ModelTrainingConfig
from .visualization import create_choropleth_map, create_timeseries_plot


def run_pipeline(city: str, config_path: Optional[Path] = None, limit: int = 1000) -> dict:
    config = load_config(config_path)

    ingest_config = DataIngestionConfig(
        api_url=config["data"].get("api_url", "https://api.openaq.org/v2/latest"),
        api_key=config["data"].get("api_key"),
        raw_dir=Path(config["data"]["raw_dir"]),
    )
    ingestor = AirQualityDataIngestor(ingest_config)
    raw_file = ingestor.fetch_latest_data(city=city, limit=limit)

    raw_df = load_raw_data(raw_file)
    features, target = preprocess_data(raw_df)
    save_processed_data(features, target, Path(config["data"]["processed_dir"]), city.lower().replace(" ", "_"))

    model_config = ModelTrainingConfig(model_path=Path(config["model"]["path"]))
    predictor = AQIPredictor(model_config)
    metrics = predictor.train(features, target)

    predictions = predictor.predict(features)
    result_df = pd.concat([features.reset_index(drop=True), predictions.rename("predicted_aqi")], axis=1)

    map_output = create_choropleth_map(
        result_df.assign(latitude=raw_df.get("coordinates.latitude"), longitude=raw_df.get("coordinates.longitude")),
        lat_col="latitude",
        lon_col="longitude",
        value_col="predicted_aqi",
        output_path=Path(config["visualization"]["map_output"]),
    )

    if "measurement_timestamp" in raw_df.columns:
        timeseries_df = pd.DataFrame(
            {
                "measurement_timestamp": pd.to_datetime(raw_df["measurement_timestamp"], errors="coerce"),
                "aqi": raw_df["aqi"],
            }
        ).dropna()
        timeseries_output = create_timeseries_plot(
            timeseries_df,
            timestamp_col="measurement_timestamp",
            value_col="aqi",
            output_path=Path(config["visualization"]["timeseries_output"]),
        )
    else:
        timeseries_output = None

    return {
        "metrics": metrics,
        "map": str(map_output),
        "timeseries": str(timeseries_output) if timeseries_output else None,
        "model_path": str(model_config.model_path),
    }
