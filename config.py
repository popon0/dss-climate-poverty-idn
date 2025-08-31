"""
config.py

Centralized configuration file for dataset paths, model artifacts,
and utility functions. All modules should import paths/functions
from here to ensure consistency across the project.

Structure:
- Base directories (data, outputs)
- Dataset paths (raw, processed, final)
- Model artifact paths
- Loader utilities
"""

from __future__ import annotations

import os
import pandas as pd


# === Base paths ===
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Dataset directories
DATASET_DIR: str = os.path.join(BASE_DIR, "data")
RAW_DIR: str = os.path.join(DATASET_DIR, "raw")
PROCESSED_DIR: str = os.path.join(DATASET_DIR, "processed")
FINAL_DIR: str = os.path.join(DATASET_DIR, "final")

# Output directory (for model artifacts and projections)
OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")


# === Dataset paths ===
DEFAULT_INPUT_CSV: str = os.path.join(
    FINAL_DIR, "dss_dataset_final_2010_2030.csv"
)
LATEST_COMBINED_CSV: str = os.path.join(
    OUTPUT_DIR, "historical_and_predicted_2010_2030.csv"
)

COM_CSV: str = os.path.join(PROCESSED_DIR, "training_data_scaled.csv")
EMISI_CSV: str = os.path.join(RAW_DIR, "ghg_emission_by_province_2010_2020.csv")
POV_CSV: str = os.path.join(RAW_DIR, "poverty_rate_by_province_2010_2020.csv")


# === Model artifact paths ===
LSTM_MODEL_PATH: str = os.path.join(OUTPUT_DIR, "lstm_model.pt")
ANFIS_MODEL_PATH: str = os.path.join(OUTPUT_DIR, "anfis_model.pt")


# === Default prediction years ===
DEFAULT_START_YEAR: int = 2021
DEFAULT_END_YEAR: int = 2030


# === Loader utilities ===
def load_csv_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load main CSV files into pandas DataFrames.

    Returns:
        tuple:
            - df_com: Processed/normalized dataset (for training)
            - df_emisi: Raw emission dataset (historical)
            - df_pov: Raw poverty dataset (historical)
    """
    df_com = pd.read_csv(COM_CSV)
    df_emisi = pd.read_csv(EMISI_CSV)
    df_pov = pd.read_csv(POV_CSV)
    return df_com, df_emisi, df_pov


def get_min_max(
    data_emisi: pd.Series, data_pov: pd.Series
) -> tuple[float, float, float, float]:
    """
    Calculate min and max values for emissions and poverty.

    Args:
        data_emisi (pd.Series): Column of emission values.
        data_pov (pd.Series): Column of poverty values.

    Returns:
        tuple:
            - min_emisi (float): Minimum emission value
            - max_emisi (float): Maximum emission value
            - min_poverty (float): Minimum poverty rate
            - max_poverty (float): Maximum poverty rate
    """
    min_emisi, max_emisi = data_emisi.min(), data_emisi.max()
    min_poverty, max_poverty = data_pov.min(), data_pov.max()
    return min_emisi, max_emisi, min_poverty, max_poverty
