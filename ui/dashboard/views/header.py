# dss/views/header.py
"""
Komponen header untuk DSS dashboard (Streamlit).

Menampilkan KPI utama di bagian atas dashboard:
- Penerimaan negara (T)
- Emisi (Ton)
- Kemiskinan (%)
- Tarif adaptif (Rp/kg)

Jika tahun == 2030 → dibandingkan dengan target.
Jika tahun != 2030 → dibandingkan dengan YoY.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd

from dashboard.theme import Theme
from dashboard.compute import FLAT_TARIFF, compute_yoy_deltas, kpi_snapshot


def _pill(ok: bool, text: str, th: Theme) -> None:
    """
    Render pill indikator dengan warna hijau/merah.

    Parameters
    ----------
    ok : bool
        Kondisi (True = hijau, False = merah).
    text : str
        Isi teks di dalam pill.
    th : Theme
        Tema (tidak dipakai untuk sekarang, tapi bisa dipakai konsistensi).
    """
    color = "#16a34a" if ok else "#e11d48"
    st.markdown(
        f"<span style='color:{color};border:1px solid {color};padding:4px 8px;"
        f"border-radius:999px;font-size:12px'>{text}</span>",
        unsafe_allow_html=True,
    )


def render_header_kpis(
    df: pd.DataFrame,
    year: int,
    mode: str,
    provinces: list[str] | None,
    TH: Theme,
    targets: dict,
    tariff_rpkg: float,
) -> None:
    """
    Render header KPIs utama.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset gabungan.
    year : int
        Tahun aktif.
    mode : str
        Mode tampilan ("Nasional" | "Provinsi" | "Perbandingan Provinsi").
    provinces : list[str] | None
        Daftar provinsi aktif (jika ada).
    TH : Theme
        Tema visualisasi.
    targets : dict
        Target nasional (emission, revenue, poverty).
    tariff_rpkg : float
        Tarif adaptif rata-rata (Rp/kg).
    """
    # KPI saat ini (sesuai konteks)
    cur, _ = _slice(df, year, mode, provinces)
    kpi = kpi_snapshot(cur)

    # Delta YoY
    yoy = compute_yoy_deltas(df, year, mode, provinces)

    # Delta vs Flat (30 Rp/kg)
    delta_vs_flat = ((tariff_rpkg - FLAT_TARIFF) / FLAT_TARIFF * 100.0) if FLAT_TARIFF else 0.0

    c1, c2, c3, c4 = st.columns(4)

    if year == 2030 and targets:
        # === Perbandingan terhadap target 2030 ===
        d_rev = _pct(kpi["revenue_T"], targets["revenue"])
        d_emi = _pct(kpi["emissions_ton"], targets["emission"])
        d_pov = _pct(kpi["poverty_pct"], targets["poverty"])

        with c1:
            st.metric(
                "Penerimaan Negara (T)",
                f"{kpi['revenue_T']:.2f}",
                f"{d_rev:+.1f}% vs target",
                help=f"Target minimum: {targets['revenue']:.2f} T",
            )
        with c2:
            st.metric(
                "Emisi (Ton)",
                f"{kpi['emissions_ton']:,.0f}",
                f"{d_emi:+.1f}% vs target",
                delta_color="inverse",
                help=f"Target maksimum: {targets['emission']:,.0f} Ton",
            )
        with c3:
            st.metric(
                "Kemiskinan (%)",
                f"{kpi['poverty_pct']:.2f}%",
                f"{d_pov:+.1f}% vs target",
                delta_color="inverse",
                help=f"Target maksimum: {targets['poverty']:.2f}%",
            )
        with c4:
            st.metric(
                "Tarif Adaptif (ANFIS) — Rp/kg",
                f"{tariff_rpkg:,.2f}",
                f"{delta_vs_flat:+.1f}% vs 30",
                help="Tarif rata-rata tertimbang emisi (Rp/kg) dibanding tarif flat 30.",
            )

        # === Gap pills ===
        g1, g2, g3 = st.columns(3)
        with g1:
            _pill(kpi["revenue_T"] >= targets["revenue"],
                  f"Gap Penerimaan: {kpi['revenue_T'] - targets['revenue']:+.2f} T", TH)
        with g2:
            _pill(kpi["emissions_ton"] <= targets["emission"],
                  f"Gap Emisi: {kpi['emissions_ton'] - targets['emission']:+,.0f} Ton", TH)
        with g3:
            _pill(kpi["poverty_pct"] <= targets["poverty"],
                  f"Gap Kemiskinan: {kpi['poverty_pct'] - targets['poverty']:+.2f} %", TH)

    else:
        # === fallback YoY ===
        with c1:
            st.metric("Penerimaan Negara (T)",
                      f"{kpi['revenue_T']:.2f}",
                      f"{yoy['revenue_yoy']:+.1f}% vs {year-1}")
        with c2:
            st.metric("Emisi (Ton)",
                      f"{kpi['emissions_ton']:,.0f}",
                      f"{yoy['emission_yoy']:+.1f}% vs {year-1}",
                      delta_color="inverse")
        with c3:
            st.metric("Kemiskinan (%)",
                      f"{kpi['poverty_pct']:.2f}%",
                      f"{yoy['poverty_yoy']:+.1f}% vs {year-1}",
                      delta_color="inverse")
        with c4:
            st.metric("Tarif Adaptif (ANFIS) — Rp/kg",
                      f"{tariff_rpkg:,.2f}",
                      f"{delta_vs_flat:+.1f}% vs 30")


def _slice(df: pd.DataFrame, year: int, mode: str, provinces: list[str] | None):
    """
    Ambil subset data untuk tahun aktif & tahun sebelumnya sesuai mode/provinsi.
    """
    if mode in ("Provinsi", "Perbandingan Provinsi") and provinces:
        return (
            df[(df["year"] == year) & (df["province"].isin(provinces))].copy(),
            df[(df["year"] == year - 1) & (df["province"].isin(provinces))].copy(),
        )
    return (
        df[df["year"] == year].copy(),
        df[df["year"] == year - 1].copy(),
    )


def _pct(now: float, base: float) -> float:
    """Hitung % perubahan terhadap base."""
    return ((now - base) / base * 100.0) if base not in (0, None) else 0.0
