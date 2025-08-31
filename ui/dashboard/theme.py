# dss/theme.py
from __future__ import annotations
from dataclasses import dataclass
import plotly.graph_objects as go


@dataclass(frozen=True)
class Theme:
    """
    Plotly theme configuration for consistent styling.
    """
    is_light: bool
    paper: str
    plot: str
    text: str
    grid: str
    template: str
    qual: tuple[str, ...]
    col_emission: str
    col_revenue: str
    col_poverty: str
    diverging: str


# === Preset themes ===
LIGHT = Theme(
    is_light=True,
    paper="#ffffff", plot="#ffffff", text="#111827", grid="#d0d5dd",
    template="plotly_white",
    qual=("#0072B2", "#E69F00", "#009E73", "#CC79A7",
          "#D55E00", "#56B4E9", "#F0E442", "#000000"),
    col_emission="#0072B2",
    col_revenue="#009E73",
    col_poverty="#E69F00",
    diverging="RdBu"
)

DARK = Theme(
    is_light=False,
    paper="#0e1117", plot="#0e1117", text="#e5e7eb", grid="#2d3648",
    template="plotly_dark",
    qual=("#56B4E9", "#F0E442", "#009E73", "#CC79A7",
          "#D55E00", "#0072B2", "#E69F00", "#ffffff"),
    col_emission="#56B4E9",
    col_revenue="#22C55E",
    col_poverty="#F59E0B",
    diverging="RdBu_r"
)


def get_theme(light: bool = False) -> Theme:
    """
    Return the current theme.
    By default, use DARK theme (light=False).
    """
    return LIGHT if light else DARK


def style_figure(fig: go.Figure, th: Theme) -> go.Figure:
    """
    Apply theme styling to a Plotly figure.
    """
    fig.update_layout(
        template=th.template,
        paper_bgcolor=th.paper,
        plot_bgcolor=th.plot,
        font_color=th.text,
    )
    fig.update_xaxes(gridcolor=th.grid, automargin=True, title_standoff=8)
    fig.update_yaxes(gridcolor=th.grid, automargin=True, title_standoff=8)
    return fig


def compact(
    fig: go.Figure,
    th: Theme,
    h: int = 260,
    legend_top: bool = False,
    hide_cbar: bool = True
) -> go.Figure:
    """
    Compact layout utility for Plotly figures.
    - h: height
    - legend_top: put legend above
    - hide_cbar: hide color bar if True
    """
    fig.update_layout(
        height=h,
        margin=dict(l=48, r=24, t=28, b=24),
        title_text=""
    )
    if legend_top:
        fig.update_layout(
            legend=dict(
                orientation="h", y=1.12, x=0.5, xanchor="center",
                font=dict(size=9)
            )
        )
    if hide_cbar:
        fig.update_layout(coloraxis_showscale=False)

    return style_figure(fig, th)
