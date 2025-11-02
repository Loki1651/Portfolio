# Urban Air Quality Prediction & Visualization

This project ingests urban air quality measurements, trains a machine learning model to predict AQI values, and produces interactive visualizations.

## Quick Start

1. **Install dependencies** (ideally in a virtual environment):
   ```bash
   python3 -m pip install --user -r requirements.txt
   ```
2. **Configure** API details in `config.yml` if needed.
3. **Run the pipeline**:
   ```bash
   python3 -m src.pipeline run_pipeline --city "Delhi"
   ```
4. Explore outputs in the `reports/` directory.

## Repository Structure

```
src/
  config.py            # Config loader with default fallback
  data_ingestion.py    # Fetches data from OpenAQ API
  data_processing.py   # Cleans and prepares features/targets
  modeling.py          # RandomForest-based AQI predictor
  visualization.py     # Folium map + Plotly timeseries
  pipeline.py          # Orchestrates end-to-end workflow
notebooks/
  aqi_analysis.ipynb   # Example notebook triggering the pipeline
config.yml              # Default configuration values
requirements.txt        # Python dependencies
```

## Next Steps
- Add automated tests for data processing and modeling modules.
- Set up CI to validate data schema changes.
- Integrate dashboard framework (e.g., Dash or Streamlit) for live monitoring.
