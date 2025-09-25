# dashboard/views/national.py
"""
National Overview Panel for Climate-Poverty Decision Support System Dashboard.

This module provides comprehensive national-level visualization and analysis capabilities
for Indonesia's climate-poverty nexus, featuring interactive mapping, trend analysis,
and policy impact assessment tools.

Key Visualization Components:
- Provincial emission and poverty distribution maps (choropleth visualization)
- National aggregate trends: emissions and government revenue (dual-axis time series)
- Adaptive tariff rate evolution and policy effectiveness indicators
- Top regional contributors analysis with year-over-year emission changes

Technical Features:
- Interactive choropleth maps using official Indonesian provincial boundaries
- Historical vs. predicted data visualization with clear temporal demarcation
- Emission-weighted tariff calculation for policy impact assessment
- Dynamic top movers analysis highlighting provincial performance variations

Academic Framework:
The module implements evidence-based policy visualization principles, providing
stakeholders with scientifically rigorous tools for national climate policy
evaluation and strategic planning within the context of poverty alleviation goals.

Dependencies:
    - streamlit: Interactive web dashboard framework
    - pandas: Data manipulation and aggregation
    - plotly: Advanced interactive visualization
    - Custom dashboard components for thematic consistency

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.theme import Theme, compact
from dashboard.charts import create_choropleth_map, create_dual_axis_timeseries


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
    Render comprehensive national climate-poverty analysis dashboard panel.

    This function creates an integrated visualization environment for national-level
    policy analysis, combining spatial distribution maps, temporal trend analysis,
    and policy effectiveness indicators to support evidence-based decision making.

    Parameters:
        df (pd.DataFrame): Complete historical and predicted dataset spanning analysis period
        df_year (pd.DataFrame): Year-specific provincial data subset for spatial analysis
        df_range (pd.DataFrame): Multi-year data subset for temporal trend computation
        year (int): Current analysis year selected for detailed examination
        TH (Theme): Dashboard theme configuration for consistent visual presentation
        kpi (dict): Key Performance Indicators snapshot for current year metrics
        tariff_rpkg (float): Emission-weighted average carbon tariff rate (Rp/kg)
        targets (dict): National strategic targets for emissions, revenue, and poverty

    Technical Implementation:
        - Implements dual-column layout optimizing information density
        - Integrates choropleth mapping with time series visualization
        - Applies weighted aggregation for national-level indicator computation
        - Provides comparative analysis framework for policy impact assessment

    Academic Significance:
        The visualization framework supports multi-criteria decision analysis
        principles, enabling policymakers to evaluate trade-offs between
        environmental objectives and socioeconomic development goals.
    """
    left, right = st.columns([2, 1])

    # === Left Panel: Spatial Distribution Analysis ===
    with left:
        st.subheader("National Emission & Poverty Distribution")

        st.caption("Greenhouse Gas Emissions (Tons)")
        fig_emisi = create_choropleth_map(df_year, "Emissions_Tons", TH)
        st.plotly_chart(compact(fig_emisi, TH, h=380, hide_cbar=False), use_container_width=True)

        st.caption("Poverty Rate Distribution (%)")
        fig_kem = create_choropleth_map(df_year, "Poverty_Rate_Percent", TH)
        st.plotly_chart(compact(fig_kem, TH, h=380, hide_cbar=False), use_container_width=True)

    # === Right Panel: Temporal Trends & Performance Analysis ===
    with right:
        st.subheader("National Aggregate Trends")

        # National emissions & government revenue trend analysis (dual-axis visualization)
        agg = (
            df.groupby("year", as_index=False)
              .agg({"Emissions_Tons": "sum", "Government_Revenue_Trillions": "sum"})
        )
        fig_trend = create_dual_axis_timeseries(agg, TH, "Emissions_Tons", "Government_Revenue_Trillions")
        st.plotly_chart(compact(fig_trend, TH, h=260, legend_top=True), use_container_width=True)

        # Adaptive carbon tariff evolution analysis
        st.subheader("Adaptive Tariff Rate Evolution")
        grp = (
            df.groupby("year", as_index=False)
              .apply(lambda g: pd.Series({
                  "tarif_aw": (
                      (g["Tax_Rate"].fillna(0) * g["Emissions_Tons"].fillna(0)).sum()
                      / max(g["Emissions_Tons"].fillna(0).sum(), 1e-9)
                  )
              }), include_groups=False)
              .reset_index(drop=True)
              .sort_values("year")
        )
        fig_tarif = px.line(
            grp, x="year", y="tarif_aw", markers=True,
            labels={"tarif_aw": "Carbon Tax Rate (Rp/kg)", "year": "Year"}
        )
        st.plotly_chart(compact(fig_tarif, TH, h=220), use_container_width=True)

        # Top regional emission performance indicators (year-over-year analysis)
        st.subheader("Top Regional Performance Indicators")
        prev = (
            df[df["year"] == year - 1][["province", "Emissions_Tons"]]
            .rename(columns={"Emissions_Tons": "prev_emissions"})
        )
        now = (
            df[df["year"] == year][["province", "Emissions_Tons"]]
            .rename(columns={"Emissions_Tons": "current_emissions"})
        )
        merged = pd.merge(prev, now, on="province", how="inner")
        
        if not merged.empty:
            merged["emission_change_pct"] = (
                (merged["current_emissions"] - merged["prev_emissions"])
                / merged["prev_emissions"].replace(0, 1e-9) * 100.0
            )
            # Display top 5 provinces with highest emission increases
            top_increases = merged.nlargest(5, "emission_change_pct")[
                ["province", "emission_change_pct"]
            ]
            
            if not top_increases.empty:
                st.write(f"**Highest Emission Increases ({year} vs {year-1})**")
                for _, row in top_increases.iterrows():
                    st.write(f"• {row['province']}: **{row['emission_change_pct']:+.1f}%**")
            else:
                st.write("*Insufficient data for year-over-year comparison*")
