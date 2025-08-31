# dss/viz.py
"""
Utility plotting untuk dashboard DSS.

Berisi fungsi-fungsi visualisasi tambahan:
- Choropleth emisi per provinsi
- Choropleth kemiskinan per provinsi
- Heatmap provinsi vs tahun untuk indikator tertentu
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .theme import get_theme, compact
from .charts import _scale_for, GEOJSON_URL


def map_emisi(df_year: pd.DataFrame) -> go.Figure:
    """
    Choropleth map emisi per provinsi untuk tahun tertentu.

    Parameters
    ----------
    df_year : pd.DataFrame
        DataFrame filter untuk 1 tahun (kolom province_code & Emisi (Ton)).

    Returns
    -------
    go.Figure
        Plotly choropleth.
    """
    th = get_theme()
    fig = px.choropleth(
        df_year,
        geojson=GEOJSON_URL,
        featureidkey="properties.kode",
        locations="province_code",
        color="Emisi (Ton)",
        color_continuous_scale=_scale_for("Emisi (Ton)", th),
    )
    fig.update_geos(fitbounds="locations", visible=False)
    return compact(fig, th, h=380, hide_cbar=False)


def map_kemiskinan(df_year: pd.DataFrame) -> go.Figure:
    """
    Choropleth map kemiskinan per provinsi untuk tahun tertentu.

    Parameters
    ----------
    df_year : pd.DataFrame
        DataFrame filter untuk 1 tahun (kolom province_code & Kemiskinan (%)).

    Returns
    -------
    go.Figure
        Plotly choropleth.
    """
    th = get_theme()
    fig = px.choropleth(
        df_year,
        geojson=GEOJSON_URL,
        featureidkey="properties.kode",
        locations="province_code",
        color="Kemiskinan (%)",
        color_continuous_scale=_scale_for("Kemiskinan (%)", th),
    )
    fig.update_geos(fitbounds="locations", visible=False)
    return compact(fig, th, h=380, hide_cbar=False)


def heatmap_prov(df: pd.DataFrame, indikator: str) -> go.Figure:
    """
    Heatmap provinsi (baris) vs tahun (kolom) untuk indikator tertentu.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset gabungan (harus punya kolom province, year, indikator).
    indikator : str
        Kolom indikator yang divisualisasikan.

    Returns
    -------
    go.Figure
        Plotly heatmap.
    """
    th = get_theme()
    pvt = df.pivot_table(
        index="province",
        columns="year",
        values=indikator,
        aggfunc="sum",
    )
    fig = px.imshow(
        pvt,
        color_continuous_scale=_scale_for(indikator, th),
        aspect="auto",
        labels=dict(color=indikator),
    )
    return compact(fig, th, h=430, hide_cbar=False)
