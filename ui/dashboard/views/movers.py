# dss/views/movers.py
"""
View: Top Movers (Streamlit)

Menampilkan provinsi dengan kenaikan dan penurunan terbesar
dari tahun sebelumnya untuk indikator tertentu (misalnya Emisi).
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.theme import Theme, compact


def render_top_movers(df: pd.DataFrame, year: int, indikator: str, TH: Theme) -> None:
    """
    Render grafik provinsi dengan kenaikan & penurunan terbesar YoY.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset gabungan.
    year : int
        Tahun aktif.
    indikator : str
        Nama kolom indikator yang dianalisis (misalnya "Emisi (Ton)").
    TH : Theme
        Tema visualisasi.
    """
    st.subheader(
        f"Top Movers: Kenaikan & Penurunan Tahunan per Provinsi — {year} vs {year-1}"
    )

    # === Filter tahun aktif & sebelumnya ===
    df_curr = df[df["year"] == year]
    df_prev = df[df["year"] == year - 1]

    # Pastikan hanya provinsi yang ada di dua tahun
    common_provs = set(df_curr["province"]) & set(df_prev["province"])
    df_curr = df_curr[df_curr["province"].isin(common_provs)]
    df_prev = df_prev[df_prev["province"].isin(common_provs)]

    # === Gabungkan & hitung perubahan (%) ===
    merged_change = pd.merge(
        df_curr[["province", indikator]],
        df_prev[["province", indikator]],
        on="province",
        suffixes=("", "_prev"),
    )
    merged_change["pct_change"] = (
        (merged_change[indikator] - merged_change[f"{indikator}_prev"])
        / merged_change[f"{indikator}_prev"].replace(0, 1e-9)
        * 100.0
    )

    # === Ambil top 10 naik & turun ===
    top_up = merged_change.sort_values("pct_change", ascending=False).head(10)
    top_down = merged_change.sort_values("pct_change", ascending=True).head(10)

    col_up, col_down = st.columns(2)

    # ---- Grafik Top 10 Kenaikan ----
    with col_up:
        fig_up = px.bar(
            top_up,
            x="pct_change",
            y="province",
            orientation="h",
            color="pct_change",
            color_continuous_scale="Greens",
            labels={"province": "Provinsi", "pct_change": "Perubahan (%)"},
            range_color=[top_up["pct_change"].min(), top_up["pct_change"].max()],
        )
        fig_up.update_layout(title=f"Top 10 Kenaikan ({indikator}) — {year} vs {year-1}")
        st.plotly_chart(compact(fig_up, TH, h=360), use_container_width=True)

    # ---- Grafik Top 10 Penurunan ----
    with col_down:
        fig_down = px.bar(
            top_down,
            x="pct_change",
            y="province",
            orientation="h",
            color="pct_change",
            color_continuous_scale="Reds_r",
            labels={"province": "Provinsi", "pct_change": "Perubahan (%)"},
            range_color=[top_down["pct_change"].min(), top_down["pct_change"].max()],
        )
        fig_down.update_layout(
            title=f"Top 10 Penurunan ({indikator}) — {year} vs {year-1}"
        )
        st.plotly_chart(compact(fig_down, TH, h=360), use_container_width=True)

    st.caption("Catatan: Perubahan dihitung per provinsi dibandingkan tahun sebelumnya.")
