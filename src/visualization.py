from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import folium
import pandas as pd
import plotly.express as px


def create_choropleth_map(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    value_col: str,
    output_path: Path,
    markers: Optional[Iterable[dict]] = None,
) -> Path:
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6,
            fill=True,
            fill_opacity=0.7,
            color="blue",
            popup=f"AQI: {row[value_col]:.2f}",
        ).add_to(fmap)

    if markers:
        for marker in markers:
            folium.Marker(location=marker["location"], popup=marker.get("popup", ""), icon=marker.get("icon")).add_to(fmap)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    return output_path


def create_timeseries_plot(df: pd.DataFrame, timestamp_col: str, value_col: str, output_path: Path) -> Path:
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    df_sorted = df.sort_values(timestamp_col)
    fig = px.line(df_sorted, x=timestamp_col, y=value_col, title=f"{value_col} over time", markers=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return output_path
