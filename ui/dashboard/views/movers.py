# dss/views/movers.py
"""
Provincial Performance Change Analysis View (Streamlit):

This module provides comprehensive year-over-year comparative analysis of provincial
performance indicators within the climate-poverty decision support system.

Key functionalities:
- Identification of provinces with highest performance improvements and deteriorations
- Year-over-year percentage change calculation and visualization
- Top movers analysis for strategic policy focus and intervention prioritization

The module supports data-driven identification of regional performance patterns
and enables targeted policy intervention based on provincial change dynamics.

Functions:
    render_top_movers: Main rendering function for provincial change analysis
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.theme import Theme, compact


def render_top_movers(df: pd.DataFrame, year: int, indikator: str, TH: Theme) -> None:
    """
    Render comprehensive provincial performance change analysis with top movers identification.

    This function analyzes year-over-year changes in provincial indicators and identifies
    the provinces with the most significant improvements and deteriorations. The analysis
    supports strategic policy focus by highlighting regional performance dynamics.

    Parameters
    ----------
    df : pd.DataFrame
        Integrated climate-poverty dataset containing multi-year provincial data
        Must include columns: province, year, and the specified indicator column
    year : int
        Current analysis year for comparison calculation
    indikator : str
        Target indicator column name for analysis (e.g., "GHG Emissions (Tons)")
    TH : Theme
        Visualization theme configuration for consistent dashboard styling
        
    Returns
    -------
    None
        Renders interactive Streamlit visualization components
        
    Notes
    -----
    The analysis focuses on provinces present in both current and previous years
    to ensure valid comparative assessment. Results highlight intervention priorities.
    """
    st.subheader(
        f"Provincial Performance Analysis: Year-over-Year Changes — {year} vs {year-1}"
    )

    # === Filter current year & previous year data ===
    df_curr = df[df["year"] == year]
    df_prev = df[df["year"] == year - 1]

    # Ensure analysis includes only provinces present in both years
    common_provs = set(df_curr["province"]) & set(df_prev["province"])
    df_curr = df_curr[df_curr["province"].isin(common_provs)]
    df_prev = df_prev[df_prev["province"].isin(common_provs)]

    # === Calculate year-over-year percentage changes ===
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

    # === Identify top 10 improvers and deteriorators ===
    top_up = merged_change.sort_values("pct_change", ascending=False).head(10)
    top_down = merged_change.sort_values("pct_change", ascending=True).head(10)

    col_up, col_down = st.columns(2)

    # ---- Top 10 Performance Improvers Visualization ----
    with col_up:
        fig_up = px.bar(
            top_up,
            x="pct_change",
            y="province",
            orientation="h",
            color="pct_change",
            color_continuous_scale="Greens",
            labels={"province": "Province", "pct_change": "Change (%)"},
            range_color=[top_up["pct_change"].min(), top_up["pct_change"].max()],
        )
        fig_up.update_layout(title=f"Top 10 Performance Improvements ({indikator}) — {year} vs {year-1}")
        st.plotly_chart(compact(fig_up, TH, h=360), use_container_width=True)

    # ---- Top 10 Performance Deteriorators Visualization ----
    with col_down:
        fig_down = px.bar(
            top_down,
            x="pct_change",
            y="province",
            orientation="h",
            color="pct_change",
            color_continuous_scale="Reds_r",
            labels={"province": "Province", "pct_change": "Change (%)"},
            range_color=[top_down["pct_change"].min(), top_down["pct_change"].max()],
        )
        fig_down.update_layout(
            title=f"Top 10 Performance Deteriorations ({indikator}) — {year} vs {year-1}"
        )
        st.plotly_chart(compact(fig_down, TH, h=360), use_container_width=True)

    st.caption("Analysis Note: Changes calculated per province compared to previous year baseline.")
