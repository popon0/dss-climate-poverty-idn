# dashboard/views/revenue.py
"""
Revenue Policy Comparison Analysis Panel for Climate-Poverty Decision Support System.

This module provides comprehensive revenue analysis capabilities comparing adaptive
carbon taxation policies (ANFIS-based) with fixed flat-rate taxation schemes,
enabling evidence-based evaluation of fiscal policy effectiveness and revenue optimization.

Key Comparative Components:
- Adaptive tariff system (ANFIS model-based) vs. Fixed flat-rate comparison
- Provincial revenue generation analysis with aggregated national totals
- Detailed tabular breakdown showing per-province revenue differentials
- Performance metrics highlighting percentage improvements and fiscal impact

Policy Analysis Framework:
- Quantitative assessment of revenue generation under different taxation scenarios
- Provincial-level fiscal impact evaluation supporting decentralized policy decisions
- Comparative visualization enabling identification of revenue optimization opportunities
- Statistical analysis of taxation policy effectiveness across diverse regional contexts

Technical Features:
- Real-time revenue calculation based on provincial emission data and tariff rates
- Dynamic provincial filtering supporting focused regional analysis
- Interactive bar chart visualization with dual-scheme comparison
- Expandable detailed tables for comprehensive fiscal planning support

Academic Significance:
The module implements public finance analysis principles specific to environmental
taxation, providing policymakers with quantitative tools for evaluating the
fiscal implications of carbon pricing mechanisms within poverty alleviation contexts.

Dependencies:
    - streamlit: Interactive web dashboard framework
    - pandas: Financial data manipulation and aggregation
    - plotly: Revenue visualization with comparative analysis
    - Custom computation modules for tariff calculations

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
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
    Render comprehensive revenue policy comparison analysis for adaptive vs. flat taxation schemes.

    This function creates detailed fiscal impact assessment comparing ANFIS-based adaptive
    carbon taxation with fixed flat-rate alternatives, providing quantitative evidence
    for policy decision-making and revenue optimization strategies.

    Parameters:
        df (pd.DataFrame): Complete provincial dataset containing emission and tariff information
        year (int): Analysis year for focused fiscal impact assessment
        TH (Theme): Dashboard theme configuration ensuring visual consistency
        provinces (list[str] | None): Optional provincial subset for targeted analysis

    Revenue Calculation Methodology:
        - ANFIS Revenue: Applies model-derived adaptive tariff rates per province
        - Flat Revenue: Applies uniform 30 Rp/kg baseline tariff across all provinces
        - Comparative Analysis: Computes percentage differentials and fiscal implications

    Technical Implementation:
        - Implements trillions-scale revenue computation for national policy context
        - Provides provincial-level disaggregation supporting sub-national planning
        - Applies dynamic filtering based on selected provincial scope
        - Generates interactive visualizations with expandable detail tables

    Analytical Framework:
        The visualization supports public finance analysis principles, enabling
        policymakers to assess fiscal trade-offs between adaptive taxation complexity
        and revenue generation effectiveness within environmental policy frameworks.
    """
    # === Analytical Scope: Current Year Provincial Data ===
    scope = df[df["year"] == year].copy()
    if provinces:
        scope = scope[scope["province"].isin(provinces)]

    # === Revenue Computation: Trillion-Scale Fiscal Analysis ===
    scope["adaptive_revenue_T"] = (
        scope["Tax_Rate"] * 1000.0 * scope["Emissions_Tons"]
    ) / 1_000_000_000_000
    scope["flat_rate_revenue_T"] = (
        FLAT_TARIFF * 1000.0 * scope["Emissions_Tons"]
    ) / 1_000_000_000_000

    # === Provincial Revenue Aggregation and Ranking ===
    agg_by_prov = (
        scope.groupby("province", as_index=False)[["adaptive_revenue_T", "flat_rate_revenue_T"]]
        .sum()
        .sort_values("adaptive_revenue_T", ascending=False)
    )

    # === National Revenue Performance Indicators ===
    total_adaptive = float(agg_by_prov["adaptive_revenue_T"].sum())
    total_flat = float(agg_by_prov["flat_rate_revenue_T"].sum())
    improvement_percentage = ((total_adaptive - total_flat) / total_flat * 100.0) if total_flat else 0.0

    st.subheader("📈 Carbon Tax Revenue Policy Comparison: Adaptive vs. Flat-Rate")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Revenue — Adaptive System (T)", f"{total_adaptive:.2f}")
    k2.metric("Total Revenue — Flat-Rate System (T)", f"{total_flat:.2f}")
    k3.metric("Revenue Enhancement vs. Flat-Rate", f"{improvement_percentage:+.1f}%")

    # === Provincial Revenue Comparison Visualization ===
    melted = agg_by_prov.melt(
        id_vars="province", var_name="Taxation_Scheme", value_name="Revenue_Trillions"
    )
    fig = px.bar(
        melted,
        x="Revenue_Trillions",
        y="province",
        color="Taxation_Scheme",
        barmode="group",
        orientation="h",
        color_discrete_map={"adaptive_revenue_T": "#22c55e", "flat_rate_revenue_T": "#94a3b8"},
        labels={"province": "Province", "Taxation_Scheme": "Policy Scheme", "Revenue_Trillions": "Revenue (Trillions Rp)"},
    )
    st.plotly_chart(compact(fig, TH, h=380), use_container_width=True)

    # === Detailed Provincial Revenue Analysis Table ===
    with st.expander("📋 Detailed Provincial Revenue Analysis"):
        st.dataframe(
            agg_by_prov.rename(
                columns={
                    "province": "Province",
                    "adaptive_revenue_T": "Adaptive System (T)",
                    "flat_rate_revenue_T": "Flat-Rate System (T)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
