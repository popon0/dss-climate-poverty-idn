"""
preprocessor.py

Data preprocessing utilities for the Decision Support System (DSS).

Includes:
- normalize(): scale raw emission & poverty data using min-max normalization
- create_lstm_sequences(): build sequential data for LSTM training
"""

from __future__ import annotations

import numpy as np
import torch
import pandas as pd


def normalize(
    df: pd.DataFrame,
    min_emisi: float,
    max_emisi: float,
    min_poverty: float,
    max_poverty: float,
) -> pd.DataFrame:
    """
    Apply min-max normalization to emission and poverty columns.

    Args:
        df (pd.DataFrame): Input dataframe with raw columns
            - "Emisi (Ton)"
            - "Kemiskinan (%)"
        min_emisi (float): Minimum emission in dataset
        max_emisi (float): Maximum emission in dataset
        min_poverty (float): Minimum poverty rate in dataset
        max_poverty (float): Maximum poverty rate in dataset

    Returns:
        pd.DataFrame: DataFrame with two new columns
            - emisi_bersih_scaled
            - poverty_scaled
    """
    df["emisi_bersih_scaled"] = (
        (df["Emisi (Ton)"] - min_emisi) / (max_emisi - min_emisi)
    )
    df["poverty_scaled"] = (
        (df["Kemiskinan (%)"] - min_poverty) / (max_poverty - min_poverty)
    )
    return df


def create_lstm_sequences(
    df_sorted: pd.DataFrame, sequence_length: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build sequences for LSTM training from normalized data.

    Args:
        df_sorted (pd.DataFrame): DataFrame sorted by [province_id, year]
            - must contain "emisi_bersih_scaled" and "poverty_scaled"
        sequence_length (int): Number of timesteps in each input sequence

    Returns:
        tuple:
            - X (torch.Tensor): Shape (N, sequence_length, 2)
            - y (torch.Tensor): Shape (N, 2)
    """
    X, y = [], []

    for pid in df_sorted["province_id"].unique():
        sub = df_sorted[df_sorted["province_id"] == pid]
        values = sub[["emisi_bersih_scaled", "poverty_scaled"]].values

        for i in range(len(values) - sequence_length):
            X.append(values[i : i + sequence_length])
            y.append(values[i + sequence_length])

    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32)

    return torch.from_numpy(X_np), torch.from_numpy(y_np)
