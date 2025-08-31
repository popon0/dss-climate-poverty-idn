"""
main.py

Training pipeline for DSS models (LSTM & ANFIS).

Modes:
- from-raw: build dataset from raw CSVs (normalize + merge)
- from-com: use preprocessed/computed dataset (scaled, normalized)

Outputs:
- Trained model artifacts (outputs/lstm_model.pt, outputs/anfis_model.pt)
- Training summary metrics (outputs/training_summary.csv)
"""

from __future__ import annotations

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch

from config import (
    LSTM_MODEL_PATH,
    ANFIS_MODEL_PATH,
    OUTPUT_DIR,
)

from dataprep.preprocessor import normalize, create_lstm_sequences
from dataprep.target import calculate_policy_score
from dataprep.province_reference import province_code_map  # optional if needed
from config import load_csv_files

from models.lstm_model import (
    LSTMForecast,
    train_lstm_model,
    evaluate_lstm_model,
)
from models.anfis_model import ANFIS, train_anfis_model


# === Utilities ===
def ensure_outputs_dir() -> None:
    """Ensure outputs/ directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# === Data preparation ===
def load_and_prepare_from_raw() -> pd.DataFrame:
    """
    Load and prepare dataset from raw CSVs.

    Steps:
        - Load raw emission & poverty data
        - Merge by [province_id, year]
        - Rename columns to standardized format
        - Normalize using historical min/max
        - Compute policy score (ANFIS target)
    """
    print("📥 Load data mentah...")
    _, df_emisi, df_pov = load_csv_files()

    # Merge on province_id + year
    df = pd.merge(
        df_emisi, df_pov,
        on=["province_id", "year"],
        how="inner",
        suffixes=("_grk", "_pov")
    )

    # Standardize columns
    df["province"] = df["province_grk"]
    df = df.rename(columns={
        "Emisi_Bersih_Ton": "Emisi (Ton)",
        "Persentase_kemiskinan": "Kemiskinan (%)"
    })
    df = df[["province_id", "province", "year", "Emisi (Ton)", "Kemiskinan (%)"]]

    # Normalize
    min_e, max_e = df["Emisi (Ton)"].min(), df["Emisi (Ton)"].max()
    min_p, max_p = df["Kemiskinan (%)"].min(), df["Kemiskinan (%)"].max()
    df = normalize(df, min_e, max_e, min_p, max_p)

    # Policy score
    df = calculate_policy_score(df)
    return df


def load_and_prepare_from_com() -> pd.DataFrame:
    """
    Load and prepare dataset from processed CSV (scaled).

    - Ensure required columns exist
    - Add policy score if missing
    """
    print("📥 Load data dari processed/training_data_scaled.csv...")
    df, _, _ = load_csv_files()

    required = {
        "province_id", "province", "year",
        "emisi_bersih_scaled", "poverty_scaled"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom hilang: {missing}. Harus ada {required}")

    if "skor_kebijakan" not in df.columns:
        df = calculate_policy_score(df)

    return df


# === Training functions ===
def train_and_eval_lstm(df: pd.DataFrame):
    """Train and evaluate LSTM model."""
    df_sorted = df.sort_values(["province_id", "year"])
    X_seq, y_seq = create_lstm_sequences(df_sorted)

    model = LSTMForecast()
    print("🔁 Training model LSTM...")
    model = train_lstm_model(model, X_seq, y_seq)

    print("✅ Evaluasi LSTM:")
    metrics = evaluate_lstm_model(model, X_seq, y_seq)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, metrics


def train_and_eval_anfis(df: pd.DataFrame):
    """Train and evaluate ANFIS model."""
    X = torch.tensor(
        df[["emisi_bersih_scaled", "poverty_scaled"]].values,
        dtype=torch.float32
    )
    if "skor_kebijakan" not in df.columns:
        df = calculate_policy_score(df)
    y = torch.tensor(
        df["skor_kebijakan"].values.reshape(-1, 1),
        dtype=torch.float32
    )

    model = ANFIS()
    print("🔁 Training model ANFIS...")
    model = train_anfis_model(model, X, y)

    with torch.no_grad():
        pred = model(X).numpy().squeeze()
        y_np = y.numpy().squeeze()
        mse = float(np.mean((pred - y_np) ** 2))
        mae = float(np.mean(np.abs(pred - y_np)))

    print("✅ Evaluasi ANFIS:")
    print(f"  mse: {mse:.6f}")
    print(f"  mae: {mae:.6f}")

    return model, {"mse": mse, "mae": mae}, df


# === Main ===
def main(mode: str) -> None:
    ensure_outputs_dir()
    set_seed(42)

    # Data preparation
    if mode == "from-raw":
        df = load_and_prepare_from_raw()
    elif mode == "from-com":
        df = load_and_prepare_from_com()
    else:
        raise ValueError("Mode tidak dikenal. Gunakan 'from-raw' atau 'from-com'.")

    # Train LSTM
    lstm_model, lstm_metrics = train_and_eval_lstm(df)

    # Train ANFIS
    anfis_model, anfis_metrics, _ = train_and_eval_anfis(df)

    # Save models
    torch.save(lstm_model.state_dict(), LSTM_MODEL_PATH)
    torch.save(anfis_model.state_dict(), ANFIS_MODEL_PATH)
    print(f"💾 Model disimpan: {LSTM_MODEL_PATH}, {ANFIS_MODEL_PATH}")

    # Save training summary
    summary = {
        "mode": mode,
        **{f"lstm_{k}": v for k, v in lstm_metrics.items()},
        **{f"anfis_{k}": v for k, v in anfis_metrics.items()},
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(OUTPUT_DIR, "training_summary.csv"), index=False
    )
    print("📑 Ringkasan metrik disimpan: outputs/training_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="from-com",
        help="Data loading mode: 'from-com' (processed) / 'from-raw' (raw merge)"
    )
    args = parser.parse_args()
    main(args.mode)
