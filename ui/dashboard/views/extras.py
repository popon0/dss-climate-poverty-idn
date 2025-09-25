# dss/views/extras.py
"""
Supplementary Analysis Panel for DSS Climate-Poverty Dashboard (Streamlit):

This module provides advanced analytical tools for comprehensive policy assessment:
- Provincial-level heatmap visualization for multi-year indicator comparison
- Scenario optimization interface for 2030 strategic target evaluation

The module supports interactive exploration of spatiotemporal patterns and scenario-based
policy optimization within the climate-poverty decision support system framework.

Functions:
    render_heatmap_and_optimizer: Main rendering function for supplementary analytical views
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.theme import Theme, compact


def render_heatmap_and_optimizer(df: pd.DataFrame, TH: Theme) -> None:
    """
    Render comprehensive analytical panel combining Provincial Heatmap & 2030 Scenario Optimizer.

    This function provides advanced visualization and optimization tools for policy analysis:
    1. Multi-dimensional provincial heatmap for indicator comparison across years
    2. Interactive scenario optimizer for 2030 strategic target assessment

    Parameters
    ----------
    df : pd.DataFrame
        Integrated dataset containing historical data and model predictions
        Must include columns: province, year, emissions, poverty_rate, government_revenue
    TH : Theme
        Visualization theme object supporting light/dark mode configurations
        
    Returns
    -------
    None
        Renders interactive Streamlit components directly to the application interface
        
    Notes
    -----
    The heatmap visualization enables identification of spatiotemporal patterns
    while the scenario optimizer supports policy target feasibility assessment.
    """
    st.markdown("## Provincial Heatmap Analysis & 2030 Scenario Optimization")

    left, right = st.columns([2, 1], vertical_alignment="top")

    # === Provincial Performance Heatmap ===
    with left:
        st.markdown("### 🔍 Provincial Performance Heatmap")
        indikator = st.selectbox(
            "Performance Indicator",
            ["Emissions_Tons", "Poverty_Rate_Percent", "Government_Revenue_Trillions"],
            index=0, key="hm_indikator"
        )
        pvt = df.pivot_table(
            index="province", columns="year", values=indikator, aggfunc="sum"
        )
        fig = px.imshow(
            pvt, aspect="auto", color_continuous_scale="RdYlGn_r",
            labels={"color": indikator}
        )
        st.plotly_chart(compact(fig, TH, h=430, hide_cbar=False), use_container_width=True)

    # === 2030 Strategic Scenario Optimizer ===
    with right:
        st.markdown("### 2030 Strategic Scenario Optimizer")
        st.caption("Policy configuration → Adjust parameters → Calculate impact assessment.")

        base_2030 = df[df["year"] == 2030].copy()
        baseline_em = base_2030["Emissions_Tons"].sum()
        baseline_kem = base_2030["Poverty_Rate_Percent"].mean()
        baseline_rev = base_2030["Government_Revenue_Trillions"].sum()

        def eval_sken(pajak_up: float, emisi_down: float, kem_down: float) -> tuple[float, float, float]:
            """
            Calculate scenario impact based on percentage changes in taxation, emissions, and poverty.
            
            Parameters
            ----------
            pajak_up : float
                Tax policy adjustment percentage (increase)
            emisi_down : float  
                Emission reduction percentage target
            kem_down : float
                Poverty reduction percentage target
                
            Returns
            -------
            tuple[float, float, float]
                New emission level, poverty rate, and government revenue
            """
            em_new = baseline_em * (1 - emisi_down / 100)
            kem_new = baseline_kem * (1 - kem_down / 100)
            rev_new = baseline_rev * (1 + pajak_up / 100) * (em_new / max(baseline_em, 1e-9))
            return em_new, kem_new, rev_new

        # --- Strategic Policy Preset Templates ---
        cA, cB, cC = st.columns(3)
        if cA.button("🌿 Green Transition"):
            st.session_state.update({"_pajak": 10, "_emisi": 25, "_kem": 5})
        if cB.button("⚖️ Balanced Approach"):
            st.session_state.update({"_pajak": 20, "_emisi": 15, "_kem": 8})
        if cC.button("💰 Revenue Optimization"):
            st.session_state.update({"_pajak": 40, "_emisi": 8, "_kem": 2})

        # --- Policy Parameter Configuration ---
        tax_pct = st.slider("Tax Policy Adjustment (%)", 0, 200, st.session_state.get("_pajak", 20), key="opt_tax")
        em_pct = st.slider("Emission Reduction Target (%)", 0, 100, st.session_state.get("_emisi", 10), key="opt_em")
        kem_pct = st.slider("Poverty Reduction Target (%)", 0, 100, st.session_state.get("_kem", 5), key="opt_kem")

        # --- Strategic Impact Assessment ---
        if st.button("▶️ Calculate Policy Impact", key="opt_run"):
            em_b, km_b, rv_b = eval_sken(tax_pct, em_pct, kem_pct)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("2030 Government Revenue (T)", f"{rv_b:.2f}",
                          delta=f"{(rv_b / baseline_rev - 1) * 100:+.1f}% vs baseline")
            with c2:
                st.metric("2030 GHG Emissions (Tons)", f"{em_b:,.0f}",
                          delta=f"{(em_b / baseline_em - 1) * 100:+.1f}% vs baseline")
            with c3:
                st.metric("2030 Poverty Rate (%)", f"{km_b:.2f}",
                          delta=f"{(km_b / baseline_kem - 1) * 100:+.1f}% vs baseline")

        # === Revenue Policy Sensitivity Analysis ===
        st.markdown("### 🔺 Revenue Policy Sensitivity Analysis (2030)")
        _, _, baseP = eval_sken(0, 0, 0)
        _, _, plus_tax = eval_sken(1, 0, 0)
        _, _, plus_em = eval_sken(0, 1, 0)

        drivers = pd.DataFrame({
            "Policy Driver": ["Tax Policy (+1%)", "Emission Reduction (+1%)"],
            "Revenue Impact (T)": [plus_tax - baseP, plus_em - baseP]
        }).sort_values("Revenue Impact (T)")

        fig_drv = px.bar(
            drivers, x="Revenue Impact (T)", y="Policy Driver", orientation="h",
            color="Revenue Impact (T)", color_continuous_scale="Blues",
            labels={"Policy Driver": "", "Revenue Impact (T)": "Revenue Impact (Trillion Rupiah)"}
        )
        st.plotly_chart(compact(fig_drv, TH, h=260), use_container_width=True)
