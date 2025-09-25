"""
config.py

Centralized configuration module for the Decision Support System (DSS) project.

This module provides a unified configuration interface for dataset paths, 
model artifacts, and utility functions. All modules should import paths
and functions from this configuration module to ensure consistency and 
maintainability across the entire project architecture.

Module Structure:
    - Base directory definitions (data, outputs)
    - Dataset path configurations (raw, processed, final)
    - Model artifact path specifications  
    - Data loading utility functions
    - Default parameter configurations

Dependencies:
    - pandas: For data manipulation and CSV loading operations
    - os: For cross-platform file path operations

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
"""

from __future__ import annotations

import os
import pandas as pd


# === Base directory configurations ===
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Dataset directory structure
DATASET_DIR: str = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR: str = os.path.join(DATASET_DIR, "raw")
PROCESSED_DATA_DIR: str = os.path.join(DATASET_DIR, "processed")
FINAL_DATA_DIR: str = os.path.join(DATASET_DIR, "final")

# Model output directory for artifacts and predictions
MODEL_OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")


# === Dataset file path configurations ===
STATIC_DATASET_CSV: str = os.path.join(
    FINAL_DATA_DIR, "dss_dataset_final_2010_2030.csv"
)
PREDICTED_COMBINED_CSV: str = os.path.join(
    MODEL_OUTPUT_DIR, "historical_and_predicted_2010_2030.csv"
)

PROCESSED_TRAINING_CSV: str = os.path.join(PROCESSED_DATA_DIR, "training_data_scaled.csv")
RAW_EMISSIONS_CSV: str = os.path.join(RAW_DATA_DIR, "ghg_emission_by_province_2010_2020.csv")
RAW_POVERTY_CSV: str = os.path.join(RAW_DATA_DIR, "poverty_rate_by_province_2010_2020.csv")


# === Trained model artifact paths ===
TRAINED_LSTM_MODEL_PATH: str = os.path.join(MODEL_OUTPUT_DIR, "lstm_model.pt")
TRAINED_ANFIS_MODEL_PATH: str = os.path.join(MODEL_OUTPUT_DIR, "anfis_model.pt")


# === Default prediction time range configurations ===
DEFAULT_PREDICTION_START_YEAR: int = 2021
DEFAULT_PREDICTION_END_YEAR: int = 2030


# === Data loading utility functions ===
def load_dataset_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load primary CSV datasets into pandas DataFrames for model training and analysis.
    
    This function provides a centralized method for loading the three core datasets
    used throughout the DSS pipeline: processed training data, raw emissions data,
    and raw poverty rate data.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - processed_df: Normalized/scaled dataset optimized for machine learning training
            - emissions_df: Historical greenhouse gas emission data by province (2010-2020)  
            - poverty_df: Historical poverty rate data by province (2010-2020)
            
    Raises:
        FileNotFoundError: If any of the required CSV files cannot be located
        pandas.errors.EmptyDataError: If any CSV file is empty or corrupted
        
    Example:
        >>> processed_data, emissions_data, poverty_data = load_dataset_files()
        >>> print(f"Processed dataset shape: {processed_data.shape}")
    """
    processed_df = pd.read_csv(PROCESSED_TRAINING_CSV)
    emissions_df = pd.read_csv(RAW_EMISSIONS_CSV)
    poverty_df = pd.read_csv(RAW_POVERTY_CSV)
    return processed_df, emissions_df, poverty_df


def calculate_normalization_bounds(
    emissions_data: pd.Series, poverty_data: pd.Series
) -> tuple[float, float, float, float]:
    """
    Calculate minimum and maximum boundary values for data normalization procedures.
    
    This function computes the statistical bounds required for min-max normalization
    of emissions and poverty data, which is essential for ensuring consistent
    feature scaling across the machine learning pipeline.
    
    Args:
        emissions_data (pd.Series): Time series of greenhouse gas emission values
        poverty_data (pd.Series): Time series of poverty rate percentage values
        
    Returns:
        tuple[float, float, float, float]: A tuple containing normalization bounds:
            - min_emissions (float): Minimum observed emission value across all data
            - max_emissions (float): Maximum observed emission value across all data  
            - min_poverty_rate (float): Minimum observed poverty percentage
            - max_poverty_rate (float): Maximum observed poverty percentage
            
    Note:
        These boundary values are crucial for maintaining consistency between
        training and prediction phases, ensuring that scaled features remain
        within the expected [0, 1] range for optimal model performance.
        
    Example:
        >>> min_e, max_e, min_p, max_p = calculate_normalization_bounds(
        ...     emissions_series, poverty_series
        ... )
        >>> print(f"Emission range: [{min_e:.2f}, {max_e:.2f}]")
    """
    min_emissions, max_emissions = emissions_data.min(), emissions_data.max()
    min_poverty_rate, max_poverty_rate = poverty_data.min(), poverty_data.max()
    return min_emissions, max_emissions, min_poverty_rate, max_poverty_rate
