# dss/charts.py
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .theme import Theme, style_figure
from .compute import HIST_CUTOFF_YEAR

# GeoJSON provinsi Indonesia (kode BPS di kolom province_code)
GEOJSON_URL = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province-simple.json"


def _scale_for(value_col: str, th: Theme) -> list[str]:
    """
    Choose an appropriate color scale depending on the indicator.

    - Emisi → OrRd (light) / YlOrRd (dark)
    - Kemiskinan → Blues (drop lightest for light mode)
    - Default → Viridis
    """
    col = value_col.lower()
    if col.startswith("emisi"):
        return px.colors.sequential.OrRd if th.is_light else px.colors.sequential.YlOrRd
    if "kemiskin" in col:
        return px.colors.sequential.Blues[2:] if th.is_light else px.colors.sequential.Blues
    return px.colors.sequential.Viridis


def choropleth(df_year: pd.DataFrame, value_col: str, th: Theme) -> go.Figure:
    """
    Choropleth map per provinsi Indonesia untuk indikator tertentu.
    """
    fig = px.choropleth(
        df_year,
        locations="province_code",
        geojson=GEOJSON_URL,
        featureidkey="properties.kode",
        color=value_col,
        color_continuous_scale=_scale_for(value_col, th),
    )
    # Perjelas batas provinsi dan latar geografi
    fig.update_traces(
        marker_line_color=("#9aa4b2" if th.is_light else "#334155"),
        marker_line_width=0.6,
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor=th.plot,  # supaya colormap tidak "hilang"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), title_text="")
    return style_figure(fig, th)


def _thicken_lines(fig: go.Figure) -> None:
    """Make all line traces at least width 2.2."""
    for tr in fig.data:
        if isinstance(tr, go.Scatter):
            tr.line.width = max(tr.line.width or 2, 2.2)


def dual_axis(
    agg: pd.DataFrame,
    th: Theme,
    col_y1: str = "Emisi (Ton)",
    col_y2: str = "Penerimaan Negara (T)",
) -> go.Figure:
    """
    Dual-axis line chart: historis vs prediksi untuk dua indikator.
    """
    hist = agg[agg["year"] <= HIST_CUTOFF_YEAR]
    pred = agg[agg["year"] > HIST_CUTOFF_YEAR]

    fig = go.Figure()

    # Historis
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["year"], y=hist[col_y1], name=f"{col_y1} (Historis)",
            line=dict(color=th.col_emission, width=2), yaxis="y1"
        ))
        fig.add_trace(go.Scatter(
            x=hist["year"], y=hist[col_y2], name=f"{col_y2} (Historis)",
            line=dict(color=th.col_revenue, width=2), yaxis="y2"
        ))

    # Prediksi
    if not pred.empty:
        fig.add_trace(go.Scatter(
            x=pred["year"], y=pred[col_y1], name=f"{col_y1} (Prediksi)",
            line=dict(color=th.col_emission, width=2, dash="dot"), yaxis="y1"
        ))
        fig.add_trace(go.Scatter(
            x=pred["year"], y=pred[col_y2], name=f"{col_y2} (Prediksi)",
            line=dict(color=th.col_revenue, width=2, dash="dot"), yaxis="y2"
        ))

    # Styling
    _thicken_lines(fig)
    fig.update_layout(
        title_text="",
        xaxis=dict(title="Tahun"),
        yaxis=dict(
            title=dict(text=col_y1, font=dict(color=th.col_emission)),
            tickfont=dict(color=th.col_emission),
        ),
        yaxis2=dict(
            title=dict(text=col_y2, font=dict(color=th.col_revenue)),
            tickfont=dict(color=th.col_revenue),
            anchor="x",
            overlaying="y",
            side="right",
        ),
    )
    return style_figure(fig, th)
