# dashboard/views/header.py
"""
Key Performance Indicator Header Component for Climate-Poverty Decision Support System Dashboard.

This module provides comprehensive KPI visualization at the dashboard header level,
featuring real-time metrics display with contextual comparison against strategic targets
or year-over-year performance indicators, depending on the temporal analysis context.

Primary KPI Components:
- Government Revenue (Trillions Rp): National or regional fiscal performance metrics
- Greenhouse Gas Emissions (Tons): Environmental impact indicators with trend analysis
- Poverty Rate (%): Socioeconomic development metrics with regional context
- Adaptive Carbon Tariff Rate (Rp/kg): Policy effectiveness indicators with emission weighting

Comparative Analysis Framework:
- Target-based evaluation for strategic years (2030 strategic targets)
- Year-over-year performance analysis for intermediate years
- Gap analysis visualization using color-coded performance pills
- Contextual help information supporting policy interpretation

Technical Features:
- Dynamic metric computation based on dashboard mode and provincial selection
- Responsive layout supporting various screen sizes and information density
- Theme-aware visualization with consistent color coding for performance indicators
- Interactive help tooltips providing contextual information for decision-makers

Academic Significance:
The module implements performance dashboard design principles specific to
multi-criteria policy evaluation, enabling real-time assessment of climate-poverty
nexus indicators within evidence-based decision-making frameworks.

Dependencies:
    - streamlit: Interactive web dashboard framework for metric display
    - pandas: Data aggregation for contextual KPI computation
    - Custom computation modules for weighted averages and delta calculations

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
"""

from __future__ import annotations
import streamlit as st
import pandas as pd

from dashboard.theme import Theme
from dashboard.compute import FLAT_TARIFF, compute_yoy_deltas, kpi_snapshot


def _pill(ok: bool, text: str, th: Theme) -> None:
    """
    Render performance indicator pill with color-coded status visualization.

    This utility function creates visually distinctive performance indicators
    using color psychology principles to communicate achievement status effectively.

    Parameters:
        ok (bool): Achievement status (True = target achieved/positive, False = gap exists/negative)
        text (str): Descriptive text content for the performance indicator
        th (Theme): Theme configuration (reserved for future theme-responsive enhancements)

    Design Principles:
        - Green indicators (#16a34a) signal achievement or positive performance
        - Red indicators (#e11d48) highlight gaps requiring policy attention
        - Rounded border design ensures professional appearance across devices
    """
    color = "#16a34a" if ok else "#e11d48"
    st.markdown(
        f"<span style='color:{color};border:1px solid {color};padding:4px 8px;"
        f"border-radius:999px;font-size:12px'>{text}</span>",
        unsafe_allow_html=True,
    )


def render_header_kpis(
    df: pd.DataFrame,
    year: int,
    mode: str,
    provinces: list[str] | None,
    TH: Theme,
    targets: dict,
    tariff_rpkg: float,
) -> None:
    """
    Render header KPIs utama.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset gabungan.
    year : int
        Tahun aktif.
    mode : str
        Mode tampilan ("Nasional" | "Provinsi" | "Perbandingan Provinsi").
    provinces : list[str] | None
        Daftar provinsi aktif (jika ada).
    TH : Theme
        Tema visualisasi.
    targets : dict
        Target nasional (emission, revenue, poverty).
    tariff_rpkg : float
        Tarif adaptif rata-rata (Rp/kg).
    """
    # KPI saat ini (sesuai konteks)
    cur, _ = _slice(df, year, mode, provinces)
    kpi = kpi_snapshot(cur)

    # Delta YoY
    yoy = compute_yoy_deltas(df, year, mode, provinces)

    # Delta vs Flat (30 Rp/kg)
    delta_vs_flat = ((tariff_rpkg - FLAT_TARIFF) / FLAT_TARIFF * 100.0) if FLAT_TARIFF else 0.0

    c1, c2, c3, c4 = st.columns(4)

    if year == 2030 and targets:
        # === Strategic Target Performance Assessment (2030) ===
        d_rev = _pct(kpi["total_revenue_trillions"], targets["scaled_revenue_target"])
        d_emi = _pct(kpi["total_emissions_tons"], targets["scaled_emission_target"])
        d_pov = _pct(kpi["average_poverty_rate"], targets["scaled_poverty_target"])

        with c1:
            st.metric(
                "Government Revenue (T)",
                f"{kpi['total_revenue_trillions']:.2f}",
                f"{d_rev:+.1f}% vs target",
                help=f"Minimum strategic target: {targets['scaled_revenue_target']:.2f} T",
            )
        with c2:
            st.metric(
                "GHG Emissions (Tons)",
                f"{kpi['total_emissions_tons']:,.0f}",
                f"{d_emi:+.1f}% vs target",
                delta_color="inverse",
                help=f"Maximum emission target: {targets['scaled_emission_target']:,.0f} Tons",
            )
        with c3:
            st.metric(
                "Poverty Rate (%)",
                f"{kpi['average_poverty_rate']:.2f}%",
                f"{d_pov:+.1f}% vs target",
                delta_color="inverse",
                help=f"Maximum poverty target: {targets['scaled_poverty_target']:.2f}%",
            )
        with c4:
            st.metric(
                "Adaptive Carbon Tariff — Rp/kg",
                f"{tariff_rpkg:,.2f}",
                f"{delta_vs_flat:+.1f}% vs baseline",
                help="Emission-weighted average tariff rate compared to baseline flat rate (30 Rp/kg).",
            )

        # === Strategic Target Performance Gap Indicators ===
        g1, g2, g3 = st.columns(3)
        with g1:
            _pill(kpi["total_revenue_trillions"] >= targets["scaled_revenue_target"],
                  f"Revenue Gap: {kpi['total_revenue_trillions'] - targets['scaled_revenue_target']:+.2f} T", TH)
        with g2:
            _pill(kpi["total_emissions_tons"] <= targets["scaled_emission_target"],
                  f"Emission Gap: {kpi['total_emissions_tons'] - targets['scaled_emission_target']:+,.0f} Tons", TH)
        with g3:
            _pill(kpi["average_poverty_rate"] <= targets["scaled_poverty_target"],
                  f"Poverty Gap: {kpi['average_poverty_rate'] - targets['scaled_poverty_target']:+.2f} %", TH)

    else:
        # === Year-over-Year Performance Analysis ===
        with c1:
            st.metric("Government Revenue (T)",
                      f"{kpi['total_revenue_trillions']:.2f}",
                      f"{yoy['revenue_yoy_change']:+.1f}% vs {year-1}")
        with c2:
            st.metric("GHG Emissions (Tons)",
                      f"{kpi['total_emissions_tons']:,.0f}",
                      f"{yoy['emission_yoy_change']:+.1f}% vs {year-1}",
                      delta_color="inverse")
        with c3:
            st.metric("Poverty Rate (%)",
                      f"{kpi['average_poverty_rate']:.2f}%",
                      f"{yoy['poverty_yoy_change']:+.1f}% vs {year-1}",
                      delta_color="inverse")
        with c4:
            st.metric("Adaptive Carbon Tariff — Rp/kg",
                      f"{tariff_rpkg:,.2f}",
                      f"{delta_vs_flat:+.1f}% vs baseline")


def _slice(df: pd.DataFrame, year: int, mode: str, provinces: list[str] | None):
    """
    Extract data subset for current year and previous year based on analysis mode and province selection.
    
    This function performs temporal and spatial data filtering to support different analysis contexts
    within the decision support system's dashboard interface.
    
    Parameters
    ----------
    df : pd.DataFrame
        Master dataset containing climate-poverty indicators across all years and provinces
    year : int
        Target analysis year for data extraction
    mode : str
        Analysis scope mode ("National", "Provincial", or "Provincial Comparison")
    provinces : list[str] | None
        List of province names for filtering (None for national analysis)
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Current year data and previous year data for comparative analysis
    """
    if mode in ("Provinsi", "Perbandingan Provinsi") and provinces:
        return (
            df[(df["year"] == year) & (df["province"].isin(provinces))].copy(),
            df[(df["year"] == year - 1) & (df["province"].isin(provinces))].copy(),
        )
    return (
        df[df["year"] == year].copy(),
        df[df["year"] == year - 1].copy(),
    )


def _pct(now: float, base: float) -> float:
    """Hitung % perubahan terhadap base."""
    return ((now - base) / base * 100.0) if base not in (0, None) else 0.0
