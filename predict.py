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
    LSTM_MODEL_PATH,
    ANFIS_MODEL_PATH,
    LATEST_COMBINED_CSV,
    OUTPUT_DIR,
    load_csv_files,
)

from dataprep.province_reference import get_province_code
from models.lstm_model import LSTMForecast
from models.anfis_model import ANFIS


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


# === Data Preparation ===
def load_minmax_from_raw() -> tuple[pd.DataFrame, tuple[float, float, float, float]]:
    """
    Load historical raw data (emission + poverty) and compute min/max ranges.

    Returns:
        tuple:
            - DataFrame (raw merged data 2010–2020)
            - (min_emisi, max_emisi, min_pov, max_pov)
    """
    _, df_emisi, df_pov = load_csv_files()
    df_raw = pd.merge(
        df_emisi, df_pov,
        on=["province_id", "year"],
        how="inner",
        suffixes=("_grk", "_pov")
    )

    df_raw["province"] = df_raw["province_grk"]
    df_raw = df_raw.rename(columns={
        "Emisi_Bersih_Ton": "Emisi (Ton)",
        "Persentase_kemiskinan": "Kemiskinan (%)"
    })
    df_raw = df_raw[["province_id", "province", "year", "Emisi (Ton)", "Kemiskinan (%)"]]

    return df_raw, (
        df_raw["Emisi (Ton)"].min(),
        df_raw["Emisi (Ton)"].max(),
        df_raw["Kemiskinan (%)"].min(),
        df_raw["Kemiskinan (%)"].max(),
    )


def get_scaled_base_dataframe(
    df_com: pd.DataFrame, df_raw: pd.DataFrame, minmax: tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Return scaled dataframe with columns:
    [province_id, province, year, emisi_bersih_scaled, poverty_scaled].

    If processed data is available, use that.
    Otherwise normalize from raw with given minmax.
    """
    if df_com is not None and not df_com.empty:
        required = {"province_id", "province", "year", "emisi_bersih_scaled", "poverty_scaled"}
        missing = required - set(df_com.columns)
        if missing:
            raise ValueError(f"Kolom hilang di processed CSV: {missing}")
        return df_com[list(required)].copy()

    min_e, max_e, min_p, max_p = minmax
    df_base = df_raw.copy()
    df_base["emisi_bersih_scaled"] = (df_base["Emisi (Ton)"] - min_e) / (max_e - min_e)
    df_base["poverty_scaled"] = (df_base["Kemiskinan (%)"] - min_p) / (max_p - min_p)
    return df_base[["province_id", "province", "year", "emisi_bersih_scaled", "poverty_scaled"]]


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
    
    values = sub[["emisi_bersih_scaled", "poverty_scaled"]].values
    if len(values) < seq_len:
        return None, sub["province"].iloc[0] if "province" in sub.columns else None
    
    last_seq = values[-seq_len:]
    province_name = sub["province"].iloc[0] if "province" in sub.columns else None
    return last_seq, province_name


def inverse_scale(pred_e: float, pred_p: float, minmax: tuple[float, float, float, float]) -> tuple[float, float]:
    """Inverse scale predictions back to original units."""
    min_e, max_e, min_p, max_p = minmax
    return (
        pred_e * (max_e - min_e) + min_e,
        pred_p * (max_p - min_p) + min_p,
    )


def compute_tax_from_score(score: float) -> float:
    """Compute tax tariff based on policy score (Sugeno rule)."""
    return max(30.0, float(score) * 150.0)


# === Prediction Pipeline ===
def run_prediction(
    start_year: int,
    end_year: int,
    lstm_path: str = LSTM_MODEL_PATH,
    anfis_path: str = ANFIS_MODEL_PATH,
    output_csv: str = LATEST_COMBINED_CSV,
    seq_len: int = 3,
) -> None:
    """Run full prediction pipeline (historical + future)."""
    ensure_outputs_dir()
    set_seed(42)

    # 1) Load data
    df_com, _, _ = load_csv_files()
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

    min_e, max_e, min_p, max_p = minmax
    hist["emisi_bersih_scaled"] = (hist["Emisi (Ton)"] - min_e) / (max_e - min_e)
    hist["poverty_scaled"] = (hist["Kemiskinan (%)"] - min_p) / (max_p - min_p)

    X_hist = torch.tensor(hist[["emisi_bersih_scaled", "poverty_scaled"]].values, dtype=torch.float32)
    with torch.no_grad():
        skor_hist = anfis_model(X_hist).cpu().numpy().squeeze()
    hist["Skor Pajak"] = skor_hist
    hist["Tarif Pajak"] = np.maximum(30.0, skor_hist * 150.0)

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

            e_scaled, p_scaled = float(np.clip(pred_scaled[0], 0.0, 1.0)), float(np.clip(pred_scaled[1], 0.0, 1.0))
            emisi_pred, pov_pred = inverse_scale(e_scaled, p_scaled, minmax)

            with torch.no_grad():
                skor = float(anfis_model(torch.tensor([[e_scaled, p_scaled]], dtype=torch.float32).to(device)).cpu().item())
            tarif = compute_tax_from_score(skor)

            future_rows.append({
                "province_code": get_province_code(province_name),
                "province": province_name,
                "province_id": pid,
                "year": yr,
                "Emisi (Ton)": emisi_pred,
                "Kemiskinan (%)": pov_pred,
                "Skor Pajak": skor,
                "Tarif Pajak": tarif,
            })

            # rolling update
            new_input = np.vstack([input_seq[0, 1:].cpu().numpy(), np.array([e_scaled, p_scaled])])
            input_seq = torch.tensor([new_input], dtype=torch.float32).to(device)

    df_future = pd.DataFrame(future_rows)

    # 5) Combine + revenue calculation
    combined = pd.concat([
        hist[["province_code", "province", "province_id", "year", "Emisi (Ton)", "Kemiskinan (%)", "Skor Pajak", "Tarif Pajak"]],
        df_future,
    ], ignore_index=True).sort_values(["province_code", "year"]).reset_index(drop=True)

    combined["Tarif Pajak (Rp/Ton)"] = combined["Tarif Pajak"] * 1000.0
    combined["Penerimaan Negara (Rp)"] = combined["Tarif Pajak (Rp/Ton)"] * combined["Emisi (Ton)"]
    combined["Penerimaan Negara (T)"] = combined["Penerimaan Negara (Rp)"] / 1_000_000_000_000
    combined["Total Nasional (T)"] = combined.groupby("year")["Penerimaan Negara (T)"].transform("sum")
    combined["Kontribusi (%)"] = (combined["Penerimaan Negara (T)"] / combined["Total Nasional (T)"]) * 100.0

    # 6) Save
    combined.to_csv(output_csv, index=False)

    # 7) Summary log
    print(f"✅ File gabungan disimpan: {output_csv}")
    print(f"Provinsi unik: {combined['province_code'].nunique()}")
    print(f"Tahun data: {combined['year'].min()} - {combined['year'].max()}")


# === CLI ===
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DSS future prediction pipeline")
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2030)
    parser.add_argument("--lstm_path", type=str, default=LSTM_MODEL_PATH)
    parser.add_argument("--anfis_path", type=str, default=ANFIS_MODEL_PATH)
    parser.add_argument("--output_csv", type=str, default=LATEST_COMBINED_CSV)
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
