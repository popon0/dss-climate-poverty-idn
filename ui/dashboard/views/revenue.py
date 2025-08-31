# dss/views/revenue.py
"""
View: Revenue Comparison (Streamlit)

Membandingkan penerimaan negara dari dua skema:
- ANFIS (tarif adaptif hasil model)
- Flat 30 (tarif tetap Rp30/kg)

Ditampilkan dalam bentuk:
- KPI total revenue nasional
- Bar chart per provinsi
- Tabel detail per provinsi
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.theme import Theme, compact
from dashboard.compute import FLAT_TARIFF


def render_revenue_compare(
    df: pd.DataFrame,
    year: int,
    TH: Theme,
    provinces: list[str] | None,
) -> None:
    """
    Render perbandingan revenue ANFIS vs Flat 30 untuk tahun aktif.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset gabungan.
    year : int
        Tahun aktif.
    TH : Theme
        Tema visualisasi.
    provinces : list[str] | None
        Subset provinsi (opsional). Jika None → seluruh provinsi.
    """
    # === Scope data tahun aktif ===
    scope = df[df["year"] == year].copy()
    if provinces:
        scope = scope[scope["province"].isin(provinces)]

    # === Hitung revenue (Triliun) ===
    scope["rev_anfis_T"] = (
        scope["Tarif Pajak"] * 1000.0 * scope["Emisi (Ton)"]
    ) / 1_000_000_000_000
    scope["rev_flat_T"] = (
        FLAT_TARIFF * 1000.0 * scope["Emisi (Ton)"]
    ) / 1_000_000_000_000

    # === Agregasi per provinsi ===
    agg_by_prov = (
        scope.groupby("province", as_index=False)[["rev_anfis_T", "rev_flat_T"]]
        .sum()
        .sort_values("rev_anfis_T", ascending=False)
    )

    # === KPI total nasional ===
    total_anfis = float(agg_by_prov["rev_anfis_T"].sum())
    total_flat = float(agg_by_prov["rev_flat_T"].sum())
    delta_pct = ((total_anfis - total_flat) / total_flat * 100.0) if total_flat else 0.0

    st.subheader("📈 Perbandingan Tarif: ANFIS vs Flat 30")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Revenue — ANFIS (T)", f"{total_anfis:.2f}")
    k2.metric("Total Revenue — Flat 30 (T)", f"{total_flat:.2f}")
    k3.metric("Kenaikan vs Flat", f"{delta_pct:+.1f} %")

    # === Bar revenue per provinsi ===
    melted = agg_by_prov.melt(
        id_vars="province", var_name="Skema", value_name="Revenue (T)"
    )
    fig = px.bar(
        melted,
        x="Revenue (T)",
        y="province",
        color="Skema",
        barmode="group",
        orientation="h",
        color_discrete_map={"rev_anfis_T": "#22c55e", "rev_flat_T": "#94a3b8"},
        labels={"province": "Provinsi", "Skema": ""},
    )
    st.plotly_chart(compact(fig, TH, h=380), use_container_width=True)

    # === Tabel detail ===
    with st.expander("📋 Tabel Tarif per Provinsi"):
        st.dataframe(
            agg_by_prov.rename(
                columns={
                    "province": "Provinsi",
                    "rev_anfis_T": "ANFIS (T)",
                    "rev_flat_T": "Flat 30 (T)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
