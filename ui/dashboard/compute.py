# dss/compute.py
from __future__ import annotations
import pandas as pd
from typing import Iterable, Literal

# === Constants ===
FLAT_TARIFF: float = 30.0
HIST_CUTOFF_YEAR: int = 2025
TARGET_YEAR: int = 2030

Mode = Literal["Nasional", "Provinsi", "Perbandingan Provinsi"]


def slice_by_context(
    df: pd.DataFrame, year: int, mode: Mode, provinces: list[str] | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (df_year, df_prev) sesuai konteks tampilan:
    - Nasional: seluruh provinsi
    - Provinsi: hanya 1 provinsi
    - Perbandingan: subset multiselect
    """
    if mode in ("Provinsi", "Perbandingan Provinsi") and provinces:
        cur = df[(df["year"] == year) & (df["province"].isin(provinces))].copy()
        prv = df[(df["year"] == year - 1) & (df["province"].isin(provinces))].copy()
    else:
        cur = df[df["year"] == year].copy()
        prv = df[df["year"] == year - 1].copy()
    return cur, prv


def compute_yoy_deltas(
    df: pd.DataFrame, year: int, mode: Mode, provinces: list[str] | None
) -> dict[str, float | bool]:
    """
    Hitung Year-on-Year % untuk Emisi, Penerimaan, Kemiskinan sesuai konteks.

    Returns
    -------
    dict
        Keys: emission_yoy, revenue_yoy, poverty_yoy, has_prev
    """
    cur, prv = slice_by_context(df, year, mode, provinces)
    cur_kpi = kpi_snapshot(cur)
    prv_kpi = kpi_snapshot(prv) if not prv.empty else {
        "emissions_ton": 0.0,
        "revenue_T": 0.0,
        "poverty_pct": 0.0
    }

    def pct(now: float, before: float) -> float:
        return ((now - before) / before * 100.0) if before not in (0, None) else 0.0

    return {
        "emission_yoy": pct(cur_kpi["emissions_ton"], prv_kpi["emissions_ton"]),
        "revenue_yoy": pct(cur_kpi["revenue_T"], prv_kpi["revenue_T"]),
        "poverty_yoy": pct(cur_kpi["poverty_pct"], prv_kpi["poverty_pct"]),
        "has_prev": not prv.empty,
    }


def filter_year_range(
    df: pd.DataFrame, start: int, end: int, provinces: Iterable[str] | None
) -> pd.DataFrame:
    """Filter data sesuai range tahun dan subset provinsi (opsional)."""
    out = df[(df["year"] >= start) & (df["year"] <= end)].copy()
    if provinces:
        out = out[out["province"].isin(provinces)]
    return out


def weighted_tariff(df: pd.DataFrame) -> float:
    """Hitung tarif rata-rata tertimbang berdasarkan emisi (Ton)."""
    w = df["Emisi (Ton)"].fillna(0)
    num = (df["Tarif Pajak"].fillna(0) * w).sum()
    den = w.sum()
    return float(num / den) if den else 0.0


def kpi_snapshot(df_year: pd.DataFrame) -> dict[str, float]:
    """Ambil KPI dasar: total emisi, total penerimaan (T), rata-rata kemiskinan (%)."""
    return {
        "emissions_ton": float(df_year["Emisi (Ton)"].sum()),
        "revenue_T": float(df_year["Penerimaan Negara (T)"].sum()),
        "poverty_pct": float(df_year["Kemiskinan (%)"].mean()),
    }


def scaled_targets(
    df_all: pd.DataFrame, df_view: pd.DataFrame,
    tgt_emission: float, tgt_revenue: float, tgt_poverty: float,
    scale_on: bool
) -> dict[str, float]:
    """
    Skala target nasional ke subset data yang sedang ditampilkan.
    Jika scale_on=False → target tetap target nasional.
    """
    base_all = df_all[df_all["year"] == TARGET_YEAR]
    base_view = df_view[df_view["year"] == TARGET_YEAR] if scale_on else base_all
    share = base_view["Emisi (Ton)"].sum() / max(base_all["Emisi (Ton)"].sum(), 1.0)
    share = share if scale_on else 1.0

    return {
        "emission": tgt_emission * share,
        "revenue": tgt_revenue * share,
        "poverty": tgt_poverty,
    }
