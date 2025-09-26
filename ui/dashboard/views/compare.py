# dashboard/views/compare.py
"""
Multi-Provincial Comparative Analysis Panel for Climate-Poverty Decision Support System.

This module provides sophisticated comparative analysis capabilities for evaluating
multiple provinces simultaneously, featuring multi-dimensional trend analysis,
correlation assessment, and comprehensive performance benchmarking tools.

Key Analytical Components:
- Multi-province emission trend visualization with temporal analysis
- Inter-indicator correlation analysis using Pearson correlation coefficients
- Year-over-year performance indicators across selected provincial set
- Comprehensive provincial comparison matrices with tabular and graphical formats
- Multi-dimensional radar profile analysis for holistic provincial assessment

Comparative Framework:
- Statistical correlation analysis supporting policy decision-making
- Dynamic provincial selection with real-time comparative updates
- Performance ranking systems with year-over-year change detection
- Comprehensive tabular summaries for detailed provincial benchmarking

Technical Features:
- Interactive multi-line time series supporting up to N provinces
- Advanced correlation heatmaps with statistical significance indicators
- Dynamic color coding for performance differentiation
- Responsive layout supporting variable provincial set sizes

Academic Significance:
The module implements multi-criteria comparative analysis principles,
enabling evidence-based evaluation of regional policy effectiveness
and identification of best practices across Indonesia's diverse provincial contexts.

Dependencies:
    - streamlit: Interactive web dashboard framework
    - pandas: Advanced data manipulation for multi-provincial analysis
    - plotly: Interactive visualization with multi-series support
    - Custom dashboard components ensuring thematic consistency

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.theme import Theme, compact


def render_compare(
    df: pd.DataFrame,
    df_year: pd.DataFrame,
    df_range: pd.DataFrame,
    year_range: tuple[int, int],
    year: int,
    TH: Theme,
    provinces: list[str],
    kpi: dict,
    tariff_rpkg: float,
    targets: dict
) -> None:
    """
    Render comprehensive multi-provincial comparative analysis dashboard panel.

    This function creates an integrated comparative analysis environment supporting
    simultaneous evaluation of multiple provinces across key climate-poverty indicators,
    enabling stakeholders to identify patterns, correlations, and performance differentials.

    Parameters:
        df (pd.DataFrame): Complete historical and predicted dataset for comparative analysis
        df_year (pd.DataFrame): Current year data subset for cross-sectional provincial comparison
        df_range (pd.DataFrame): Multi-year dataset subset for temporal comparative analysis
        year_range (tuple[int, int]): Temporal range boundaries for aggregation and correlation analysis
        year (int): Current analysis year for detailed comparative examination
        TH (Theme): Dashboard theme configuration ensuring visual consistency across comparisons
        provinces (list[str]): Selected provincial set for focused comparative analysis
        kpi (dict): Aggregated Key Performance Indicators for comparative context
        tariff_rpkg (float): Weighted average carbon tariff rate across selected provinces
        targets (dict): Strategic targets providing benchmarking context for comparative assessment

    Technical Implementation:
        - Implements dual-panel layout optimizing comparative information presentation
        - Integrates multi-series time visualization with correlation analysis
        - Applies statistical methods for inter-indicator relationship assessment
        - Provides comprehensive tabular summaries supporting detailed comparison

    Analytical Framework:
        The visualization supports comparative policy analysis principles,
        enabling identification of best practices and performance patterns
        across diverse provincial contexts within Indonesia's federal system.
    """
    left, right = st.columns([1.6, 1])

    # === Left Panel: Multi-Provincial Trend & Correlation Analysis ===
    with left:
        head = ", ".join(provinces[:4]) + ("…" if len(provinces) > 4 else "")
        st.subheader(f"Multi-Provincial Trends — {head}")

        # Multi-series emission trend analysis (historical and predicted)
        d = df[df["province"].isin(provinces)].copy()
        fig_line = px.line(d, x="year", y="Emissions_Tons", color="province")
        st.plotly_chart(compact(fig_line, TH, h=400, legend_top=True), use_container_width=True)

        # Inter-indicator correlation analysis (Pearson correlation matrix)
        st.subheader("Inter-Indicator Correlation Analysis (Pearson)")
        scope = df[
            (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
            & (df["province"].isin(provinces))
        ].copy()
        
        # Use raw data (province-year) without aggregation
        filtered = scope[
            ["Emissions_Tons", "Government_Revenue_Trillions", "Poverty_Rate_Percent", "Tax_Rate"]
        ].dropna()

        if len(filtered) >= 2:
            corr = filtered.corr(method="pearson")
        else:
            st.warning("Not enough data points to compute Pearson correlation.")
            corr = pd.DataFrame()  # Empty placeholder

        fig_corr = px.imshow(
            corr, text_auto=".2f", zmin=-1, zmax=1,
            color_continuous_scale=TH.diverging, labels={"color": "Correlation Coefficient (ρ)"}
        )
        st.plotly_chart(compact(fig_corr, TH, h=360, hide_cbar=False), use_container_width=True)

    # === Right Panel: Performance Indicators & Comparative Assessment ===
    with right:
        # Top movers
        st.subheader("Top Provincial Movers (Emission YoY Analysis)")
        prev = (
            df[df["year"] == year - 1][["province", "Emissions_Tons"]]
            .rename(columns={"Emissions_Tons": "prev"})
        )
        now = (
            df[df["year"] == year][["province", "Emissions_Tons"]]
            .rename(columns={"Emissions_Tons": "now"})
        )
        yoy = (
            now.merge(prev, on="province", how="inner")
            .query("prev != 0 and province in @provinces")
        )
        yoy["pct"] = (yoy["now"] - yoy["prev"]) / yoy["prev"] * 100
        sel = yoy.sort_values("pct", ascending=False).head(7).sort_values("pct")

        fig_mv = px.bar(
            sel, x="pct", y="province", orientation="h",
            color="pct", color_continuous_scale="Greens",
            labels={"pct": "%", "province": ""}
        )
        st.plotly_chart(compact(fig_mv, TH, h=260), use_container_width=True)

        # Perbandingan ringkas
        st.subheader("Provincial Performance Comparison — Current Year Analysis")
        cmp = df[(df["year"] == year) & (df["province"].isin(provinces))].copy()
        grp = (
            cmp.groupby("province", as_index=False)
            .agg({
                "Emissions_Tons": "sum",
                "Government_Revenue_Trillions": "sum",
                "Poverty_Rate_Percent": "mean"
            })
        )
        wtar = (
            cmp.groupby("province")
            .apply(lambda g: (
                (g["Tax_Rate"] * g["Emissions_Tons"]).sum()
                / max(g["Emissions_Tons"].sum(), 1e-9)
            ), include_groups=False)
            .reset_index(name="Tax Rate (Rp/kg)")
        )
        grp = grp.merge(wtar, on="province", how="left")
        total_nas = float(df[df["year"] == year]["Emissions_Tons"].sum())
        grp["National Share (%)"] = grp["Emissions_Tons"] / max(total_nas, 1e-9) * 100
        grp["Rev Density (T per Mt)"] = (
            grp["Government_Revenue_Trillions"] * 1_000_000
        ) / grp["Emissions_Tons"]

        st.dataframe(
            grp.rename(columns={
                "province": "Provinsi",
                "Government_Revenue_Trillions": "Revenue (T)"
            }),
            hide_index=True, use_container_width=True
        )

        # Radar profil
        st.subheader("Multi-Dimensional Performance Radar Analysis — Emissions / Revenue / Tariff / Poverty")

        def _norm(s: pd.Series) -> pd.Series:
            mn, mx = float(s.min()), float(s.max())
            return (s - mn) / (mx - mn) if mx > mn else s * 0

        rb = grp.copy()
        rb["E"] = _norm(rb["Emissions_Tons"])
        rb["P"] = _norm(
            rb["Revenue (T)"] if "Revenue (T)" in rb else rb["Government_Revenue_Trillions"]
        )
        rb["T"] = _norm(rb["Tax Rate (Rp/kg)"])
        rb["K"] = 1 - _norm(rb["Poverty_Rate_Percent"])
        cats = ["Emissions", "Revenue", "Tax Rate", "Poverty(↓)"]

        fig_r = go.Figure()
        for _, r in rb.iterrows():
            fig_r.add_trace(go.Scatterpolar(
                r=[r["E"], r["P"], r["T"], r["K"], r["E"]],
                theta=cats + ["Emissions"], name=r["province"], line=dict(width=2)
            ))
        fig_r.update_layout(
            height=320, margin=dict(l=0, r=0, t=25, b=25),
            polar=dict(radialaxis=dict(visible=True, range=[0, 1]))
        )
        st.plotly_chart(compact(fig_r, TH, h=320), use_container_width=True)

