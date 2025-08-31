# dss/views/national.py
"""
Tampilan nasional untuk Decision Support System (Streamlit dashboard).

Menampilkan:
- Sebaran emisi & kemiskinan nasional (peta choropleth).
- Tren nasional: emisi & penerimaan negara (dual-axis).
- Tren tarif pajak rata-rata adaptif.
- Top 5 movers (perubahan YoY emisi terbesar).
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.theme import Theme, compact
from dashboard.charts import choropleth, dual_axis


def render_national(
    df: pd.DataFrame,
    df_year: pd.DataFrame,
    df_range: pd.DataFrame,
    year: int,
    TH: Theme,
    kpi: dict,
    tariff_rpkg: float,
    targets: dict
) -> None:
    """
    Render panel tampilan nasional.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset lengkap (historis + prediksi).
    df_year : pd.DataFrame
        Subset data untuk tahun yang dipilih.
    df_range : pd.DataFrame
        Subset data sesuai rentang tahun.
    year : int
        Tahun aktif (dipilih di UI).
    TH : Theme
        Objek tema visualisasi (light/dark).
    kpi : dict
        KPI snapshot (emisi, revenue, poverty).
    tariff_rpkg : float
        Tarif rata-rata tertimbang (Rp/kg).
    targets : dict
        Target skala nasional (emission, revenue, poverty).
    """
    left, right = st.columns([2, 1])

    # === Left: Peta sebaran emisi & kemiskinan ===
    with left:
        st.subheader("Sebaran Emisi & Kemiskinan Nasional")

        st.caption("Emisi (Ton)")
        fig_emisi = choropleth(df_year, "Emisi (Ton)", TH)
        st.plotly_chart(compact(fig_emisi, TH, h=380, hide_cbar=False), use_container_width=True)

        st.caption("Kemiskinan (%)")
        fig_kem = choropleth(df_year, "Kemiskinan (%)", TH)
        st.plotly_chart(compact(fig_kem, TH, h=380, hide_cbar=False), use_container_width=True)

    # === Right: Tren & movers ===
    with right:
        st.subheader("Tren — Nasional")

        # Emisi & penerimaan negara (dual-axis)
        agg = (
            df.groupby("year", as_index=False)
              .agg({"Emisi (Ton)": "sum", "Penerimaan Negara (T)": "sum"})
        )
        fig_trend = dual_axis(agg, TH, "Emisi (Ton)", "Penerimaan Negara (T)")
        st.plotly_chart(compact(fig_trend, TH, h=260, legend_top=True), use_container_width=True)

        # Tren tarif rata-rata adaptif
        st.subheader("Tren Tarif (Adaptif)")
        grp = (
            df.groupby("year", as_index=False)
              .apply(lambda g: pd.Series({
                  "tarif_aw": (
                      (g["Tarif Pajak"].fillna(0) * g["Emisi (Ton)"].fillna(0)).sum()
                      / max(g["Emisi (Ton)"].fillna(0).sum(), 1e-9)
                  )
              }))
              .reset_index(drop=True)
              .sort_values("year")
        )
        fig_tarif = px.line(
            grp, x="year", y="tarif_aw", markers=True,
            labels={"tarif_aw": "Rp/kg", "year": "Tahun"}
        )
        st.plotly_chart(compact(fig_tarif, TH, h=220), use_container_width=True)

        # Top 5 movers emisi YoY
        st.subheader("Top 5 Movers (Emisi YoY)")
        prev = (
            df[df["year"] == year - 1][["province", "Emisi (Ton)"]]
            .rename(columns={"Emisi (Ton)": "prev"})
        )
        now = (
            df[df["year"] == year][["province", "Emisi (Ton)"]]
            .rename(columns={"Emisi (Ton)": "now"})
        )
        yoy = now.merge(prev, on="province", how="inner").query("prev != 0")
        yoy["pct"] = (yoy["now"] - yoy["prev"]) / yoy["prev"] * 100
        top5 = yoy.sort_values("pct", ascending=False).head(5).sort_values("pct")

        fig_mv = px.bar(
            top5, x="pct", y="province", orientation="h",
            color="pct", color_continuous_scale="Greens",
            labels={"pct": "%", "province": ""}
        )
        st.plotly_chart(compact(fig_mv, TH, h=230), use_container_width=True)
