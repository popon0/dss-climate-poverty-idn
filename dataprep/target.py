"""
target.py

Utility functions for computing target values
used in the ANFIS model.
"""

from __future__ import annotations
import pandas as pd


def calculate_policy_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate policy score (skor_kebijakan) as a weighted combination
    of emission and poverty indices.

    Formula:
        skor_kebijakan = 0.6 * emisi_bersih_scaled
                         + 0.4 * (1 - poverty_scaled)

    Args:
        df (pd.DataFrame): Input DataFrame containing:
            - "emisi_bersih_scaled" (float): normalized emission [0–1]
            - "poverty_scaled" (float): normalized poverty [0–1]

    Returns:
        pd.DataFrame: DataFrame with new column "skor_kebijakan"
    """
    df["skor_kebijakan"] = (
        0.6 * df["emisi_bersih_scaled"] + 0.4 * (1 - df["poverty_scaled"])
    )
    return df
