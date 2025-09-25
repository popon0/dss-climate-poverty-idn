# dss/io.py
from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from config import STATIC_DATASET_CSV, PREDICTED_COMBINED_CSV

# Required columns for any dataset loaded into the dashboard
REQUIRED_COLS = {
    "year", "province", "province_code",
    "Emissions_Tons", "Government_Revenue_Trillions",
    "Poverty_Rate_Percent", "Tax_Rate"
}


@st.cache_data(show_spinner=False)
def load_data(path: str | None = None) -> pd.DataFrame:
    """
    Load dataset for the dashboard with caching.

    Priority:
    1. Use provided path (if given).
    2. Otherwise, use LATEST_COMBINED_CSV (predicted dataset).
    3. If not available, fallback to DEFAULT_INPUT_CSV (final static dataset).

    Raises
    ------
    ValueError
        If required columns are missing from the dataset.
    """
    if path is None:
        path = PREDICTED_COMBINED_CSV if os.path.exists(PREDICTED_COMBINED_CSV) else STATIC_DATASET_CSV

    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in file: {os.path.basename(path)}")

    return df


def clear_cache() -> None:
    """Clear Streamlit cache for load_data()."""
    try:
        load_data.clear()  # type: ignore[attr-defined]
    except Exception:
        pass
