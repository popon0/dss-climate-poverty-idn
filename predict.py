# Copyright 2025 Teuku Hafiez Ramadhan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
predict.py

Generate historical + future predictions (emission, poverty, tax score, revenue).

Pipeline:
1. Load raw & processed dataset
2. Load trained LSTM & ANFIS models
3. Compute historical scores (2010–2020)
4. Forecast future years (default: 2021–2030)
5. Merge historical + future, compute revenue & contributions
6. Save to CSV
"""

from __future__ import annotations
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch

from config import (
    TRAINED_LSTM_MODEL_PATH,
    TRAINED_ANFIS_MODEL_PATH,
    PREDICTED_COMBINED_CSV,
    MODEL_OUTPUT_DIR,
    load_dataset_files,
)

from dataprep.province_reference import get_province_code
from models.lstm_model import LSTMForecast
from models.anfis_model import ANFIS


# === Utilities ===
def ensure_outputs_dir() -> None:
    """Ensure outputs/ directory exists."""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# === Data Preparation ===
def load_minmax_from_raw() -> tuple[pd.DataFrame, tuple[float, float, float, float]]:
    """
    Load historical raw data (emission + poverty) and compute min/max ranges.

    Returns:
        tuple:
            - DataFrame (raw merged data 2010–2020)
            - (min_emissions, max_emissions, min_poverty, max_poverty)
    """
    _, df_emissions, df_poverty = load_dataset_files()
    df_raw = pd.merge(
        df_emissions, df_poverty,
        on=["province_id", "year"],
        how="inner",
        suffixes=("_emissions", "_poverty")
    )

    df_raw["province"] = df_raw["province_emissions"]
    df_raw = df_raw.rename(columns={
        "Net_Emissions_Tons": "Emissions_Tons",
        "Poverty_Rate_Percent": "Poverty_Rate_Percent"
    })
    df_raw = df_raw[["province_id", "province", "year", "Emissions_Tons", "Poverty_Rate_Percent"]]

    return df_raw, (
        df_raw["Emissions_Tons"].min(),
        df_raw["Emissions_Tons"].max(),
        df_raw["Poverty_Rate_Percent"].min(),
        df_raw["Poverty_Rate_Percent"].max(),
    )


def get_scaled_base_dataframe(
    df_com: pd.DataFrame, df_raw: pd.DataFrame, minmax: tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Return scaled dataframe with columns:
    [province_id, province, year, scaled_net_emissions, scaled_poverty_rate].

    If processed data is available, use that.
    Otherwise normalize from raw with given minmax.
    """
    if df_com is not None and not df_com.empty:
        required = {"province_id", "province", "year", "scaled_net_emissions", "scaled_poverty_rate"}
        missing = required - set(df_com.columns)
        if missing:
            raise ValueError(f"Missing columns in processed CSV: {missing}")
        return df_com[list(required)].copy()

    min_emissions, max_emissions, min_poverty, max_poverty = minmax
    df_base = df_raw.copy()
    df_base["scaled_net_emissions"] = (df_base["Emissions_Tons"] - min_emissions) / (max_emissions - min_emissions)
    df_base["scaled_poverty_rate"] = (df_base["Poverty_Rate_Percent"] - min_poverty) / (max_poverty - min_poverty)
    return df_base[["province_id", "province", "year", "scaled_net_emissions", "scaled_poverty_rate"]]


def build_last_sequence(df_base: pd.DataFrame, pid: int, seq_len=3):
    """
    Build the last sequence of scaled values for a given province.

    Args:
        df_base: Scaled dataframe
        pid: Province ID
        seq_len: Sequence length

    Returns:
        (last_seq, province_name) or (None, None) if insufficient data
    """
    sub = df_base[df_base["province_id"] == pid].sort_values("year")
    if sub.empty:
        return None, None
    
    values = sub[["scaled_net_emissions", "scaled_poverty_rate"]].values
    if len(values) < seq_len:
        return None, sub["province"].iloc[0] if "province" in sub.columns else None
    
    last_seq = values[-seq_len:]
    province_name = sub["province"].iloc[0] if "province" in sub.columns else None
    return last_seq, province_name


def inverse_scale(pred_emissions: float, pred_poverty: float, minmax: tuple[float, float, float, float]) -> tuple[float, float]:
    """Inverse scale predictions back to original units."""
    min_emissions, max_emissions, min_poverty, max_poverty = minmax
    return (
        pred_emissions * (max_emissions - min_emissions) + min_emissions,
        pred_poverty * (max_poverty - min_poverty) + min_poverty,
    )


def compute_tax_from_score(score: float) -> float:
    """Compute tax tariff based on policy score (Sugeno rule)."""
    return max(30.0, float(score) * 150.0)


# === Prediction Pipeline ===
def run_prediction(
    start_year: int,
    end_year: int,
    lstm_path: str = TRAINED_LSTM_MODEL_PATH,
    anfis_path: str = TRAINED_ANFIS_MODEL_PATH,
    output_csv: str = PREDICTED_COMBINED_CSV,
    seq_len: int = 3,
) -> None:
    """Run full prediction pipeline (historical + future)."""
    ensure_outputs_dir()
    set_seed(42)

    # 1) Load data
    df_com, _, _ = load_dataset_files()
    df_raw, minmax = load_minmax_from_raw()
    df_base = get_scaled_base_dataframe(df_com, df_raw, minmax)

    # 2) Load models
    device = torch.device("cpu")
    lstm_model = LSTMForecast().to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()

    anfis_model = ANFIS().to(device)
    anfis_model.load_state_dict(torch.load(anfis_path, map_location=device))
    anfis_model.eval()

    # 3) Historical scores
    hist = df_raw.copy()
    hist["province_code"] = hist["province"].apply(get_province_code)

    min_emissions, max_emissions, min_poverty, max_poverty = minmax
    hist["scaled_net_emissions"] = (hist["Emissions_Tons"] - min_emissions) / (max_emissions - min_emissions)
    hist["scaled_poverty_rate"] = (hist["Poverty_Rate_Percent"] - min_poverty) / (max_poverty - min_poverty)

    X_hist = torch.tensor(hist[["scaled_net_emissions", "scaled_poverty_rate"]].values, dtype=torch.float32)
    with torch.no_grad():
        score_hist = anfis_model(X_hist).cpu().numpy().squeeze()
    hist["Tax_Score"] = score_hist
    hist["Tax_Rate"] = np.maximum(30.0, score_hist * 150.0)

    # 4) Future predictions
    future_rows = []
    for pid in df_base["province_id"].unique():
        last_seq, province_name = build_last_sequence(df_base, pid, seq_len=seq_len)
        if last_seq is None:
            continue
        input_seq = torch.tensor([last_seq], dtype=torch.float32).to(device)

        for yr in range(start_year, end_year + 1):
            with torch.no_grad():
                pred_scaled = lstm_model(input_seq).cpu().numpy().squeeze()

            emissions_scaled, poverty_scaled = float(np.clip(pred_scaled[0], 0.0, 1.0)), float(np.clip(pred_scaled[1], 0.0, 1.0))
            emissions_pred, poverty_pred = inverse_scale(emissions_scaled, poverty_scaled, minmax)

            with torch.no_grad():
                score = float(anfis_model(torch.tensor([[emissions_scaled, poverty_scaled]], dtype=torch.float32).to(device)).cpu().item())
            tax_rate = compute_tax_from_score(score)

            future_rows.append({
                "province_code": get_province_code(province_name),
                "province": province_name,
                "province_id": pid,
                "year": yr,
                "Emissions_Tons": emissions_pred,
                "Poverty_Rate_Percent": poverty_pred,
                "Tax_Score": score,
                "Tax_Rate": tax_rate,
            })

            # rolling update
            new_input = np.vstack([input_seq[0, 1:].cpu().numpy(), np.array([emissions_scaled, poverty_scaled])])
            input_seq = torch.tensor([new_input], dtype=torch.float32).to(device)

    df_future = pd.DataFrame(future_rows)

    # 5) Combine + revenue calculation
    combined = pd.concat([
        hist[["province_code", "province", "province_id", "year", "Emissions_Tons", "Poverty_Rate_Percent", "Tax_Score", "Tax_Rate"]],
        df_future,
    ], ignore_index=True).sort_values(["province_code", "year"]).reset_index(drop=True)

    combined["Tax_Rate_Rp_Per_Ton"] = combined["Tax_Rate"] * 1000.0
    combined["Government_Revenue_Rp"] = combined["Tax_Rate_Rp_Per_Ton"] * combined["Emissions_Tons"]
    combined["Government_Revenue_Trillions"] = combined["Government_Revenue_Rp"] / 1_000_000_000_000
    combined["National_Total_Trillions"] = combined.groupby("year")["Government_Revenue_Trillions"].transform("sum")
    combined["Contribution_Percent"] = (combined["Government_Revenue_Trillions"] / combined["National_Total_Trillions"]) * 100.0

    # 6) Save
    combined.to_csv(output_csv, index=False)

    # 7) Summary log
    print(f"✅ Combined file saved: {output_csv}")
    print(f"Unique provinces: {combined['province_code'].nunique()}")
    print(f"Data years: {combined['year'].min()} - {combined['year'].max()}")


# === CLI ===
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DSS future prediction pipeline")
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2030)
    parser.add_argument("--lstm_path", type=str, default=TRAINED_LSTM_MODEL_PATH)
    parser.add_argument("--anfis_path", type=str, default=TRAINED_ANFIS_MODEL_PATH)
    parser.add_argument("--output_csv", type=str, default=PREDICTED_COMBINED_CSV)
    parser.add_argument("--sequence_length", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(
        start_year=args.start_year,
        end_year=args.end_year,
        lstm_path=args.lstm_path,
        anfis_path=args.anfis_path,
        output_csv=args.output_csv,
        seq_len=args.sequence_length,
    )
