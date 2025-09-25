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

# app.py
from __future__ import annotations
import os
import streamlit as st
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))
from config import (
    PREDICTED_COMBINED_CSV, STATIC_DATASET_CSV,
)
from dashboard.io import load_data
from dashboard.theme import get_theme
from dashboard.compute import (
    filter_temporal_range, generate_kpi_snapshot, calculate_emission_weighted_tariff, compute_proportional_targets
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

# --- Sidebar Navigation Interface ---
st.sidebar.header("📊 Decision Support Panel")
display_mode = st.sidebar.radio(
    "Display Mode", ["National", "Provincial", "Provincial Comparison"], index=0
)

# Fixed theme configuration
dashboard_theme = get_theme(light=False)  # Default dark theme for optimal visibility

# Dataset path prioritization (predicted data → fallback to static dataset)
dataset_path = PREDICTED_COMBINED_CSV if os.path.exists(PREDICTED_COMBINED_CSV) else STATIC_DATASET_CSV
national_dataset = load_data(dataset_path)

if not os.path.exists(PREDICTED_COMBINED_CSV):
    st.sidebar.warning("⚠️ Predicted data not available. Displaying static final dataset.")

# --- Temporal and Geographical Selection Interface ---
if display_mode == "Provincial Comparison":
    year_range_min, year_range_max = int(national_dataset.year.min()), int(national_dataset.year.max())
    start_year, end_year = st.sidebar.slider("Select Year Range", year_range_min, year_range_max, (2015, 2025))
    selected_year = end_year
else:
    selected_year = st.sidebar.slider("Select Analysis Year", int(national_dataset.year.min()), int(national_dataset.year.max()), 2022)
    start_year, end_year = selected_year, selected_year

selected_provinces_list: list[str] | None = None
if display_mode == "Provincial":
    selected_province = st.sidebar.selectbox("Select Province", sorted(national_dataset.province.unique()))
    selected_provinces_list = [selected_province]
elif display_mode == "Provincial Comparison":
    selected_provinces_list = st.sidebar.multiselect(
        "Select Multiple Provinces",
        sorted(national_dataset.province.unique()),
        default=sorted(national_dataset.province.unique())[:2],
    )

# --- National Strategic Targets 2030 ---
st.sidebar.markdown("### 🎯 National Strategic Targets (2030)")
national_emission_target = st.sidebar.number_input("Emission Target 2030 (Tons)", min_value=0, value=1_500_000_000)
national_poverty_target = st.sidebar.number_input("Poverty Target 2030 (%)", min_value=0.0, value=7.5, step=0.1)
national_revenue_target = st.sidebar.number_input("Revenue Target 2030 (T)", min_value=0.0, value=180.0, step=1.0)
enable_proportional_scaling = st.sidebar.checkbox("🔗 Scale targets to current view", value=True)
st.sidebar.caption("When enabled: targets are proportionally scaled based on regional contribution (baseline 2030).")

# --- Data Export Functionality ---
with st.sidebar.expander("💾 Download Filtered Data"):
    filtered_download_data = national_dataset[(national_dataset["year"] >= start_year) & (national_dataset["year"] <= end_year)].copy()
    if selected_provinces_list:
        filtered_download_data = filtered_download_data[filtered_download_data["province"].isin(selected_provinces_list)]
    st.download_button(
        "Download CSV",
        data=filtered_download_data.to_csv(index=False).encode("utf-8"),
        file_name="filtered_dataset.csv",
        mime="text/csv",
    )


# --- KPI Calculations and Target Configuration ---
temporal_filtered_data = filter_temporal_range(national_dataset, start_year, end_year, selected_provinces_list)
current_year_data = national_dataset[(national_dataset.year == selected_year) & (national_dataset.province.isin(selected_provinces_list) if selected_provinces_list else True)].copy()
current_kpi_metrics = generate_kpi_snapshot(current_year_data)
weighted_tariff_rate = calculate_emission_weighted_tariff(current_year_data)

strategic_targets = compute_proportional_targets(
    complete_dataset=national_dataset,
    view_specific_dataset=(national_dataset[national_dataset.province.isin(selected_provinces_list)] if selected_provinces_list else national_dataset),
    national_emission_target=national_emission_target,
    national_revenue_target=national_revenue_target,
    national_poverty_target=national_poverty_target,
    enable_proportional_scaling=enable_proportional_scaling,
)

# --- Application Header Configuration ---
application_title = "Climate-Poverty DSS"
application_subtitle = (
    "Decision Support System — National Overview" if display_mode == "National"
    else f"Decision Support System — {selected_provinces_list[0]}" if selected_provinces_list and len(selected_provinces_list) == 1
    else "Decision Support System — Provincial Comparison"
)
st.markdown(f"## {application_title}")
st.caption(f"{application_subtitle} • **Analysis Year {selected_year}**")

# --- Key Performance Indicators Header ---
render_header_kpis(
    df=national_dataset, year=selected_year, mode=display_mode, provinces=selected_provinces_list,
    TH=dashboard_theme, targets=strategic_targets, tariff_rpkg=weighted_tariff_rate
)

# --- Dashboard View Rendering Logic ---
if display_mode == "National":
    render_national(national_dataset, current_year_data, temporal_filtered_data, selected_year, dashboard_theme, current_kpi_metrics, weighted_tariff_rate, strategic_targets)
    render_revenue_compare(national_dataset, selected_year, dashboard_theme, provinces=None)
    render_heatmap_and_optimizer(national_dataset, dashboard_theme)
    render_top_movers(national_dataset, selected_year, indikator="Emissions_Tons", TH=dashboard_theme)

elif display_mode == "Provincial":
    render_province(national_dataset, current_year_data, temporal_filtered_data, selected_year, dashboard_theme, selected_provinces_list[0], current_kpi_metrics, weighted_tariff_rate, strategic_targets)
    render_revenue_compare(national_dataset[national_dataset['province'] == selected_provinces_list[0]], selected_year, dashboard_theme, provinces=selected_provinces_list)
    render_heatmap_and_optimizer(national_dataset, dashboard_theme)

else:  # Provincial Comparison
    render_compare(national_dataset, current_year_data, temporal_filtered_data, (start_year, end_year), selected_year, dashboard_theme, selected_provinces_list, current_kpi_metrics, weighted_tariff_rate, strategic_targets)
    render_revenue_compare(national_dataset, selected_year, dashboard_theme, provinces=selected_provinces_list)
    render_heatmap_and_optimizer(national_dataset, dashboard_theme)

