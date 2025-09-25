# dashboard/charts.py
"""
Advanced Visualization Components for Decision Support System Dashboard.

This module provides sophisticated charting capabilities for environmental and
socioeconomic data visualization, featuring interactive choropleth maps and
dual-axis time series charts with historical and predictive data support.

Key Components:
    - choropleth(): Interactive Indonesian provincial maps with thematic coloring
    - dual_axis(): Time series visualization with historical vs predicted data
    - _scale_for(): Intelligent color scale selection based on data indicators
    - _thicken_lines(): Visual enhancement utilities for improved readability

Technical Features:
    - GeoJSON integration for accurate Indonesian provincial boundaries
    - Dynamic color scaling based on indicator types (emissions, poverty, etc.)
    - Theme-aware styling for consistent visual presentation
    - Responsive design optimized for dashboard integration

Dependencies:
    - plotly: For interactive visualization components
    - pandas: For data manipulation and aggregation
    - Custom theme system for consistent styling

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
"""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .theme import Theme, style_figure
from .compute import HISTORICAL_DATA_CUTOFF_YEAR

# GeoJSON data source for Indonesian provincial boundaries (BPS-compliant province codes)
INDONESIA_GEOJSON_URL = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province-simple.json"


def _select_color_scale_for_indicator(indicator_column: str, theme: Theme) -> list[str]:
    """
    Select an optimal color scale based on the environmental or socioeconomic indicator type.
    
    This function implements intelligent color scale selection that enhances data interpretation
    by applying perceptually appropriate color schemes for different indicator categories.
    
    Args:
        indicator_column (str): Name of the data column/indicator being visualized
        theme (Theme): Current theme configuration (light/dark mode)
        
    Returns:
        list[str]: Plotly-compatible color scale optimized for the specific indicator
        
    Color Scale Logic:
        - Emissions indicators: Orange-Red gradients (OrRd/YlOrRd) to emphasize environmental concern
        - Poverty indicators: Blue gradients (Blues) with enhanced contrast for light themes  
        - Default indicators: Viridis scale for general-purpose visualization
        
    Note:
        The function automatically adjusts color intensity and contrast based on the current
        theme to ensure optimal visibility in both light and dark interface modes.
        
    Example:
        >>> scale = _select_color_scale_for_indicator("Emissions (Tons)", dark_theme)
        >>> print(scale)  # ['#440154', '#21908c', '#fde725', ...]
    """
    normalized_column = indicator_column.lower()
    
    if normalized_column.startswith("emissions") or "emission" in normalized_column:
        return px.colors.sequential.OrRd if theme.is_light else px.colors.sequential.YlOrRd
    elif "poverty" in normalized_column or "kemiskin" in normalized_column:
        return px.colors.sequential.Blues[2:] if theme.is_light else px.colors.sequential.Blues
    else:
        return px.colors.sequential.Viridis


def create_choropleth_map(yearly_data: pd.DataFrame, indicator_column: str, theme: Theme) -> go.Figure:
    """
    Generate interactive choropleth map visualization for Indonesian provincial data analysis.
    
    This function creates publication-quality thematic maps that visualize environmental and
    socioeconomic indicators across Indonesian provinces using official BPS administrative
    boundaries and intelligent color scaling systems.
    
    Args:
        yearly_data (pd.DataFrame): Provincial dataset for a specific year containing:
            - province_code: Official BPS province identification codes
            - [indicator_column]: Numerical values to be visualized spatially
        indicator_column (str): Name of the column containing values for choropleth coloring
        theme (Theme): Visual theme configuration for consistent dashboard styling
        
    Returns:
        go.Figure: Interactive Plotly choropleth figure with the following features:
            - Province boundary highlighting with adaptive line styling
            - Theme-responsive color schemes optimized for indicator types
            - Automatic geographic bounds fitting for optimal viewing
            - Clean layout optimized for dashboard integration
            
    Technical Implementation:
        - Utilizes official Indonesian GeoJSON boundaries with BPS province codes
        - Implements adaptive color scaling based on indicator characteristics
        - Applies theme-aware styling for optimal visibility in light/dark modes
        - Removes unnecessary UI elements for clean dashboard presentation
        
    Example:
        >>> provincial_emissions_2022 = df[df['year'] == 2022]
        >>> emissions_map = create_choropleth_map(
        ...     provincial_emissions_2022, 
        ...     "Emissions_Tons", 
        ...     dark_theme
        ... )
        >>> emissions_map.show()
    """
    choropleth_figure = px.choropleth(
        yearly_data,
        locations="province_code",
        geojson=INDONESIA_GEOJSON_URL,
        featureidkey="properties.kode",
        color=indicator_column,
        color_continuous_scale=_select_color_scale_for_indicator(indicator_column, theme),
    )
    
    # Enhanced provincial boundary visualization with theme-adaptive styling
    choropleth_figure.update_traces(
        marker_line_color=("#9aa4b2" if theme.is_light else "#334155"),
        marker_line_width=0.6,
    )
    
    # Geographic visualization optimization
    choropleth_figure.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor=theme.plot,  # Ensures proper color map visibility
    )
    
    # Clean dashboard-optimized layout
    choropleth_figure.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        title_text=""
    )
    
    return style_figure(choropleth_figure, theme)


def _enhance_line_visibility(figure: go.Figure) -> None:
    """
    Enhance line trace visibility by ensuring minimum line width for improved readability.
    
    This utility function automatically adjusts line widths in Plotly figures to ensure
    optimal visibility and professional presentation in dashboard environments.
    
    Args:
        figure (go.Figure): Plotly figure containing line traces to be enhanced
        
    Side Effects:
        Modifies the input figure in-place by updating line width properties
        of all Scatter traces to a minimum readable thickness.
        
    Technical Details:
        - Applies minimum line width of 2.2 pixels for optimal visibility
        - Preserves existing line styling while enhancing thickness
        - Only affects Scatter-type traces (line and scatter plots)
        
    Note:
        This function is typically used internally by chart creation functions
        to ensure consistent line visibility across different visualization themes.
    """
    for trace in figure.data:
        if isinstance(trace, go.Scatter) and trace.line is not None:
            current_width = getattr(trace.line, 'width', 2)
            trace.line.width = max(current_width, 2.2)


def create_dual_axis_timeseries(
    aggregated_data: pd.DataFrame,
    theme: Theme,
    primary_indicator: str = "Emissions_Tons",
    secondary_indicator: str = "Government_Revenue_Trillions",
) -> go.Figure:
    """
    Create sophisticated dual-axis time series visualization with historical and predictive data.
    
    This function generates publication-quality time series charts that effectively display
    two related indicators on separate y-axes, with clear visual distinction between
    historical observations and model predictions.
    
    Args:
        aggregated_data (pd.DataFrame): Time series dataset containing:
            - year: Temporal dimension for x-axis
            - primary_indicator: First indicator values (left y-axis)
            - secondary_indicator: Second indicator values (right y-axis)
        theme (Theme): Visual theme configuration for consistent styling
        primary_indicator (str): Column name for left y-axis indicator (default: emissions)
        secondary_indicator (str): Column name for right y-axis indicator (default: revenue)
        
    Returns:
        go.Figure: Interactive dual-axis time series with the following features:
            - Separate y-axes with indicator-specific color coding
            - Visual distinction between historical (solid) and predicted (dashed) data
            - Enhanced line visibility with automatic thickness adjustment
            - Theme-responsive styling for optimal dashboard integration
            
    Technical Implementation:
        - Uses HIST_CUTOFF_YEAR to separate historical from predictive periods
        - Applies indicator-specific color schemes from theme configuration
        - Implements dual y-axis layout with proper axis positioning
        - Enhances line visibility for improved chart readability
        
    Example:
        >>> annual_summary = df.groupby('year').agg({
        ...     'Emissions_Tons': 'sum',
        ...     'Government_Revenue_Trillions': 'sum'
        ... }).reset_index()
        >>> chart = create_dual_axis_timeseries(annual_summary, dark_theme)
        >>> chart.show()
    """
    # Separate historical and predictive data periods
    historical_data = aggregated_data[aggregated_data["year"] <= HISTORICAL_DATA_CUTOFF_YEAR]
    predictive_data = aggregated_data[aggregated_data["year"] > HISTORICAL_DATA_CUTOFF_YEAR]

    figure = go.Figure()

    # Add historical data traces with solid lines
    if not historical_data.empty:
        figure.add_trace(go.Scatter(
            x=historical_data["year"], 
            y=historical_data[primary_indicator], 
            name=f"{primary_indicator} (Historical)",
            line=dict(color=theme.col_emission, width=2), 
            yaxis="y1"
        ))
        figure.add_trace(go.Scatter(
            x=historical_data["year"], 
            y=historical_data[secondary_indicator], 
            name=f"{secondary_indicator} (Historical)",
            line=dict(color=theme.col_revenue, width=2), 
            yaxis="y2"
        ))

    # Add predictive data traces with dashed lines
    if not predictive_data.empty:
        figure.add_trace(go.Scatter(
            x=predictive_data["year"], 
            y=predictive_data[primary_indicator], 
            name=f"{primary_indicator} (Predicted)",
            line=dict(color=theme.col_emission, width=2, dash="dot"), 
            yaxis="y1"
        ))
        figure.add_trace(go.Scatter(
            x=predictive_data["year"], 
            y=predictive_data[secondary_indicator], 
            name=f"{secondary_indicator} (Predicted)",
            line=dict(color=theme.col_revenue, width=2, dash="dot"), 
            yaxis="y2"
        ))

    # Enhance line visibility for improved readability
    _enhance_line_visibility(figure)
    
    # Configure dual-axis layout with indicator-specific styling
    figure.update_layout(
        title_text="",
        xaxis=dict(title="Year"),
        yaxis=dict(
            title=dict(text=primary_indicator, font=dict(color=theme.col_emission)),
            tickfont=dict(color=theme.col_emission),
        ),
        yaxis2=dict(
            title=dict(text=secondary_indicator, font=dict(color=theme.col_revenue)),
            tickfont=dict(color=theme.col_revenue),
            anchor="x",
            overlaying="y",
            side="right",
        ),
    )
    
    return style_figure(figure, theme)
