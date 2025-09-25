# dashboard/views/province.py
"""
Provincial Analysis Panel for Climate-Poverty Decision Support System Dashboard.

This module provides comprehensive provincial-level analysis capabilities for detailed
examination of regional climate-poverty dynamics, featuring trend analysis, contribution
assessment, and comparative performance evaluation within the national context.

Key Visualization Components:
- Provincial emission and revenue trend analysis (dual-axis time series)
- Adaptive carbon tariff evolution for specific regional contexts
- Year-over-year performance indicators and regional ranking analysis
- Provincial contribution assessment relative to national totals and peer provinces

Analytical Framework:
- Provincial KPI computation with national context integration
- Emission contribution ranking with top-tier comparative analysis
- Regional performance benchmarking using donut charts and horizontal bar rankings
- Multi-dimensional assessment combining environmental and economic indicators

Technical Features:
- Dynamic provincial selection with real-time data filtering
- Consistent color theming with provincial highlighting for visual emphasis
- Interactive visualization supporting drill-down analysis capabilities
- Responsive layout optimization for comprehensive provincial assessment

Academic Significance:
The module implements regional development analysis principles, enabling
evidence-based evaluation of provincial climate policy effectiveness
and socioeconomic impact within Indonesia's decentralized governance framework.

Dependencies:
    - streamlit: Interactive web dashboard framework
    - pandas: Advanced data manipulation and aggregation
    - plotly: Interactive visualization with provincial highlighting
    - Custom dashboard components for thematic consistency

Author: Teuku Hafiez Ramadhan  
License: Apache License 2.0
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.theme import Theme, compact
from dashboard.charts import create_dual_axis_timeseries

# === Consistent Color Scheme for Provincial Highlighting ===
HIGHLIGHT_COLOR = "#16a34a"   # Emerald green for selected province emphasis
NEUTRAL_COLOR   = "#94a3b8"   # Slate gray for comparative provinces  
ACCENT_COLOR    = "#60a5fa"   # Sky blue for "Top-10 Others" category

# Shorter aliases for readability in visualizations
HIGHLIGHT = HIGHLIGHT_COLOR
NEUTRAL = NEUTRAL_COLOR
ACCENT = ACCENT_COLOR


def render_province(
    df: pd.DataFrame,
    df_year: pd.DataFrame,
    df_range: pd.DataFrame,
    year: int,
    TH: Theme,
    province: str,
    kpi: dict,
    tariff_rpkg: float,
    targets: dict
) -> None:
    """
    Render comprehensive provincial climate-poverty analysis dashboard panel.

    This function creates a detailed analytical environment for provincial-level
    policy assessment, combining temporal trend analysis with national context
    comparison to support regional decision-making and performance evaluation.

    Parameters:
        df (pd.DataFrame): Complete historical and predicted dataset for provincial analysis
        df_year (pd.DataFrame): Current year provincial data for cross-sectional analysis
        df_range (pd.DataFrame): Multi-year provincial dataset for trend computation
        year (int): Current analysis year for detailed provincial examination
        TH (Theme): Dashboard theme configuration ensuring visual consistency
        province (str): Selected province name for focused analysis and highlighting
        kpi (dict): Provincial Key Performance Indicators for current year metrics
        tariff_rpkg (float): Provincial emission-weighted carbon tariff rate (Rp/kg)
        targets (dict): Strategic targets scaled to provincial context when applicable

    Technical Implementation:
        - Implements dual-panel layout optimizing provincial data presentation
        - Integrates provincial trends with national comparative context
        - Applies dynamic provincial highlighting in visualization components
        - Provides comprehensive contribution analysis relative to national totals

    Analytical Framework:
        The visualization supports sub-national policy analysis principles,
        enabling regional policymakers to assess provincial performance within
        the broader national climate-poverty strategy context.
    """
    left, right = st.columns([1.6, 1])

    # === Left Panel: Provincial Trend Analysis ===
    with left:
        st.subheader(f"Provincial Trends — {province}")
        agg = (
            df[df["province"] == province]
            .groupby("year", as_index=False)
            .agg({"Emissions_Tons": "sum", "Government_Revenue_Trillions": "sum"})
        )
        fig_trend = create_dual_axis_timeseries(agg, TH, "Emissions_Tons", "Government_Revenue_Trillions")
        st.plotly_chart(
            compact(fig_trend, TH, h=400, legend_top=True),
            use_container_width=True
        )

        st.subheader(f"Adaptive Carbon Tariff Evolution — {province}")
        grp = (
            df[df["province"] == province]
            .groupby("year", as_index=False)
            .apply(lambda d: pd.Series({
                "tarif_aw": (
                    (d["Tax_Rate"].fillna(0) * d["Emissions_Tons"].fillna(0)).sum()
                    / max(d["Emissions_Tons"].fillna(0).sum(), 1e-9)
                )
            }), include_groups=False)
            .reset_index(drop=True)
            .sort_values("year")
        )
        fig_tarif = px.line(
            grp, x="year", y="tarif_aw", markers=True,
            labels={"tarif_aw": "Carbon Tax Rate (Rp/kg)", "year": "Year"}
        )
        st.plotly_chart(compact(fig_tarif, TH, h=400), use_container_width=True)

    # === Right Panel: Performance Indicators & Provincial Contribution ===
    with right:
        st.subheader("Regional Performance Indicators")
        prev = (
            df[df["year"] == year - 1][["province", "Emissions_Tons"]]
            .rename(columns={"Emissions_Tons": "previous_emissions"})
        )
        now = (
            df[df["year"] == year][["province", "Emissions_Tons"]]
            .rename(columns={"Emissions_Tons": "current_emissions"})
        )
        yoy = now.merge(prev, on="province", how="inner").query("previous_emissions != 0")
        yoy["emission_change_pct"] = (yoy["current_emissions"] - yoy["previous_emissions"]) / yoy["previous_emissions"] * 100
        sel = yoy.sort_values("emission_change_pct", ascending=False).head(7).sort_values("emission_change_pct")

        fig_mv = px.bar(
            sel, x="emission_change_pct", y="province", orientation="h",
            color="emission_change_pct", color_continuous_scale="Greens",
            labels={"emission_change_pct": "YoY Change (%)", "province": "Province"}
        )
        st.plotly_chart(compact(fig_mv, TH, h=240), use_container_width=True)

        st.subheader("Provincial Emission Contribution & Ranking Analysis")
        render_province_contribution(df, year, province, TH)


def render_province_contribution(
    df: pd.DataFrame,
    year: int,
    province: str,
    TH: Theme
) -> None:
    """
    Render comprehensive provincial contribution analysis relative to national context.

    This function provides detailed assessment of provincial performance within the
    national emission landscape, featuring ranking analysis, contribution metrics,
    and comparative visualization with peer provinces.

    Visualization Components:
    - Key Performance Indicators: national ranking, contribution percentage, and top-tier comparison
    - Donut chart visualization showing provincial contribution within national total
    - Horizontal bar ranking of top-10 provinces with selected province highlighting

    Analytical Framework:
        The visualization implements comparative performance analysis principles,
        enabling stakeholders to assess provincial impact within broader national
        climate policy context and identify peer provinces for benchmarking analysis.
    """
    yr = df[df["year"] == year].copy()
    if yr.empty:
        st.info("⚠️ No data available for the selected year.")
        return

    # === National aggregation and provincial ranking analysis ===
    agg = (
        yr.groupby("province", as_index=False)["Emissions_Tons"]
        .sum().sort_values("Emissions_Tons", ascending=False).reset_index(drop=True)
    )
    total = float(agg["Emissions_Tons"].sum())
    if total <= 0 or province not in agg["province"].values:
        st.info("⚠️ Emission data unavailable or province not found in dataset.")
        return

    row = agg[agg["province"] == province].iloc[0]
    val = float(row["Emissions_Tons"])
    rank = int(row.name) + 1
    nprov = len(agg)
    contrib = val / total * 100.0

    # === Top-tier comparative analysis (Top-10 provinces) ===
    top10 = agg.head(10).copy()
    top10_sum = float(top10["Emissions_Tons"].sum())
    share_vs_top10 = (val / top10_sum * 100.0) if top10_sum else 0.0

    # === Provincial Performance Summary Metrics ===
    k1, k2, k3 = st.columns(3)
    k1.metric("Emission Ranking", f"#{rank}", help=f"Out of {nprov} provinces nationwide")
    k2.metric("National Contribution", f"{contrib:.2f}%")
    k3.metric("Top-10 Share", f"{share_vs_top10:.2f}%", help="Contribution relative to top-10 provinces combined")

    # === Provincial Contribution Donut Chart Visualization ===
    if province in top10["province"].values:
        top10_others = float(top10[top10["province"] != province]["Emissions_Tons"].sum())
        others_rest = max(total - (val + top10_others), 0.0)
        ddf = pd.DataFrame({
            "Component": [province, "Other Top-10", "Outside Top-10"],
            "Emissions": [val, top10_others, others_rest],
        })
    else:
        others_rest = max(total - (val + top10_sum), 0.0)
        ddf = pd.DataFrame({
            "Component": [province, "Top-10 (excluding selected)", "Other Provinces"],
            "Emissions": [val, top10_sum, others_rest],
        })
    ddf["Percentage"] = ddf["Emissions"] / total * 100.0

    col_map = {
        province: HIGHLIGHT,
        "Top-10 (excluding selected)": ACCENT,
        "Other Provinces": NEUTRAL,
    }
    fig_pie = px.pie(
        ddf, names="Component", values="Percentage",
        hole=0.6, color="Component", color_discrete_map=col_map
    )
    fig_pie.update_traces(
        texttemplate="%{label}<br>%{percent:.1%}",
        textposition="inside",
        pull=[0.08 if n == province else 0 for n in ddf["Component"]],
    )
    fig_pie.update_layout(
        title=f"Provincial Emission Contribution Analysis: {province} ({contrib:.2f}% of total)",
        height=240, margin=dict(l=6, r=6, t=30, b=6), showlegend=False
    )
    st.plotly_chart(compact(fig_pie, TH, h=240, hide_cbar=True), use_container_width=True)

    # === Bar chart top-10 ===
    rank10 = agg.head(10).copy()
    if province not in rank10["province"].values:
        keep = rank10.tail(9)
        sel = agg[agg["province"] == province]
        rank10 = (
            pd.concat([keep, sel], ignore_index=True)
            .sort_values("Emissions_Tons", ascending=True)
        )
    else:
        rank10 = rank10.sort_values("Emissions_Tons", ascending=True)

    rank10["__col"] = rank10["province"].apply(
        lambda x: HIGHLIGHT if x == province else NEUTRAL
    )
    fig_bar = px.bar(
        rank10, x="Emissions_Tons", y="province", orientation="h",
        color="__col", color_discrete_map="identity",
        labels={"province": "", "Emissions_Tons": "Emissions (Tons)"}
    )
    fig_bar.update_layout(
        title="Emission Ranking (Top-10)",
        height=260, margin=dict(l=6, r=6, t=30, b=6), showlegend=False
    )
    fig_bar.add_annotation(
        xref="paper", yref="paper", x=0, y=-0.14, showarrow=False,
        text=f"#{rank} of {nprov} provinces • {contrib:.2f}% national share • {share_vs_top10:.2f}% of Top-10 total",
        font=dict(size=11, color="#9aa4b2"),
    )
    st.plotly_chart(compact(fig_bar, TH, h=260, hide_cbar=True), use_container_width=True)
