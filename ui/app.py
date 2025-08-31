# app.py
from __future__ import annotations
import os
import streamlit as st
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    LATEST_COMBINED_CSV, DEFAULT_INPUT_CSV,
)
from dashboard.io import load_data
from dashboard.theme import get_theme
from dashboard.compute import (
    filter_year_range, kpi_snapshot, weighted_tariff, scaled_targets
)
from dashboard.views.national import render_national
from dashboard.views.province import render_province
from dashboard.views.compare import render_compare
from dashboard.views.extras import render_heatmap_and_optimizer
from dashboard.views.revenue import render_revenue_compare
from dashboard.views.movers import render_top_movers
from dashboard.views.header import render_header_kpis


# --- Page setup ---
st.set_page_config(layout="wide")
st.markdown(
    "<style>.stPlotlyChart,.plot-container,.js-plotly-plot{overflow:visible!important}"
    "div[data-testid='stAppViewContainer'] .block-container{padding-top:1.2rem!important;}</style>",
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.header("📊 Decision Support Panel")
mode = st.sidebar.radio(
    "Mode Tampilan", ["Nasional", "Provinsi", "Perbandingan Provinsi"], index=0
)

# Fixed theme
TH = get_theme(light=False)  # default dark theme

# Dataset path (prefer predicted → fallback final static)
data_path = LATEST_COMBINED_CSV if os.path.exists(LATEST_COMBINED_CSV) else DEFAULT_INPUT_CSV
df = load_data(data_path)

if not os.path.exists(LATEST_COMBINED_CSV):
    st.sidebar.warning("⚠️ Data prediksi belum ada. Menampilkan dataset statis final.")

# --- Tahun & wilayah ---
if mode == "Perbandingan Provinsi":
    year_min, year_max = int(df.year.min()), int(df.year.max())
    start, end = st.sidebar.slider("Pilih Rentang Tahun", year_min, year_max, (2015, 2025))
    year = end
else:
    year = st.sidebar.slider("Pilih Tahun", int(df.year.min()), int(df.year.max()), 2022)
    start, end = year, year

prov_list: list[str] | None = None
if mode == "Provinsi":
    prov = st.sidebar.selectbox("Pilih Provinsi", sorted(df.province.unique()))
    prov_list = [prov]
elif mode == "Perbandingan Provinsi":
    prov_list = st.sidebar.multiselect(
        "Pilih Beberapa Provinsi",
        sorted(df.province.unique()),
        default=sorted(df.province.unique())[:2],
    )

# --- Target Nasional 2030 ---
st.sidebar.markdown("### 🎯 Target Nasional (2030)")
target_emission = st.sidebar.number_input("Target Emisi 2030 (Ton)", min_value=0, value=1_500_000_000)
target_poverty = st.sidebar.number_input("Target Kemiskinan 2030 (%)", min_value=0.0, value=7.5, step=0.1)
target_revenue = st.sidebar.number_input("Target Penerimaan 2030 (T)", min_value=0.0, value=180.0, step=1.0)
scale_on = st.sidebar.checkbox("🔗 Skala target ke tampilan saat ini", value=True)
st.sidebar.caption("Jika aktif: target diproporsikan sesuai kontribusi wilayah tampilan (baseline 2030).")

# --- Download filtered data ---
with st.sidebar.expander("💾 Unduh Data (sesuai filter)"):
    dlf = df[(df["year"] >= start) & (df["year"] <= end)].copy()
    if prov_list:
        dlf = dlf[dlf["province"].isin(prov_list)]
    st.download_button(
        "Unduh CSV",
        data=dlf.to_csv(index=False).encode("utf-8"),
        file_name="data_filtered.csv",
        mime="text/csv",
    )


# --- KPI & Targets ---
df_range = filter_year_range(df, start, end, prov_list)
df_year = df[(df.year == year) & (df.province.isin(prov_list) if prov_list else True)].copy()
kpi = kpi_snapshot(df_year)
tariff = weighted_tariff(df_year)

targets = scaled_targets(
    df_all=df,
    df_view=(df[df.province.isin(prov_list)] if prov_list else df),
    tgt_emission=target_emission,
    tgt_revenue=target_revenue,
    tgt_poverty=target_poverty,
    scale_on=scale_on,
)

# --- Header ---
title = "DSS Nasional"
subtitle = (
    "Sistem Pendukung Keputusan — Ringkasan Nasional" if mode == "Nasional"
    else f"Sistem Pendukung Keputusan — {prov_list[0]}" if prov_list and len(prov_list) == 1
    else "Sistem Pendukung Keputusan — Perbandingan Provinsi"
)
st.markdown(f"## {title}")
st.caption(f"{subtitle} • **Tahun {year}**")

# --- KPIs Header ---
render_header_kpis(
    df=df, year=year, mode=mode, provinces=prov_list,
    TH=TH, targets=targets, tariff_rpkg=tariff
)

# --- View Rendering ---
if mode == "Nasional":
    render_national(df, df_year, df_range, year, TH, kpi, tariff, targets)
    render_revenue_compare(df, year, TH, provinces=None)
    render_heatmap_and_optimizer(df, TH)
    render_top_movers(df, year, indikator="Emisi (Ton)", TH=TH)

elif mode == "Provinsi":
    render_province(df, df_year, df_range, year, TH, prov_list[0], kpi, tariff, targets)
    render_revenue_compare(df[df['province'] == prov_list[0]], year, TH, provinces=prov_list)
    render_heatmap_and_optimizer(df, TH)

else:  # Perbandingan
    render_compare(df, df_year, df_range, (start, end), year, TH, prov_list, kpi, tariff, targets)
    render_revenue_compare(df, year, TH, provinces=prov_list)
    render_heatmap_and_optimizer(df, TH)
