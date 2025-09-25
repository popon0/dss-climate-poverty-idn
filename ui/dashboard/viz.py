# dashboard/viz.py
"""
Advanced Visualization Utilities for Decision Support System Dashboard.

This module provides specialized visualization functions that extend the core charting
capabilities with domain-specific implementations for environmental and socioeconomic
data analysis. The utilities focus on creating publication-quality visualizations
optimized for policy decision-making and stakeholder communication.

Specialized Visualization Components:
    - create_emissions_choropleth(): Provincial emission distribution mapping
    - create_poverty_choropleth(): Socioeconomic indicator spatial visualization  
    - create_provincial_temporal_heatmap(): Multi-dimensional temporal analysis

Technical Features:
    - Integration with official Indonesian geographical boundaries (GeoJSON)
    - Theme-responsive color scaling for optimal visibility
    - Compact layout optimization for dashboard integration
    - Advanced heatmap capabilities for spatiotemporal analysis

Design Philosophy:
    - Prioritizes clarity and interpretability for policy stakeholders
    - Implements consistent visual language across different chart types
    - Optimizes for responsive dashboard environments
    - Maintains scientific accuracy in data representation

Dependencies:
    - plotly: For interactive visualization generation
    - pandas: For data preprocessing and pivot operations
    - Custom theme and charting systems for consistency

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .theme import get_theme, compact
from .charts import _select_color_scale_for_indicator, INDONESIA_GEOJSON_URL


def create_emissions_choropleth(yearly_provincial_data: pd.DataFrame) -> go.Figure:
    """
    Generate specialized choropleth visualization for provincial greenhouse gas emissions analysis.
    
    This function creates publication-quality thematic maps specifically optimized for
    environmental data visualization, implementing best practices for emission data
    presentation and stakeholder communication in policy contexts.
    
    Args:
        yearly_provincial_data (pd.DataFrame): Single-year provincial dataset containing:
            - province_code: Official BPS provincial identification codes
            - "Emissions_Tons": Greenhouse gas emission values for choropleth coloring
            
    Returns:
        go.Figure: Interactive emissions choropleth with the following features:
            - Environment-optimized color scaling (orange-red gradients)
            - Enhanced height for detailed provincial examination
            - Integrated color bar for quantitative interpretation
            - Theme-responsive styling for dashboard integration
            
    Design Considerations:
        - Uses warming color schemes (orange-red) to intuitively represent environmental concern
        - Implements larger figure height (380px) for detailed provincial identification
        - Maintains color bar visibility for quantitative data interpretation
        - Applies compact layout optimization for dashboard integration
        
    Policy Applications:
        - Environmental impact assessment across provinces
        - Emission hotspot identification for targeted interventions
        - Regional comparison for policy prioritization
        - Stakeholder communication of environmental challenges
        
    Example:
        >>> emissions_2022 = df[df['year'] == 2022]
        >>> emissions_map = create_emissions_choropleth(emissions_2022)
        >>> st.plotly_chart(emissions_map, use_container_width=True)
    """
    current_theme = get_theme()
    
    emissions_map = px.choropleth(
        yearly_provincial_data,
        geojson=INDONESIA_GEOJSON_URL,
        featureidkey="properties.kode",
        locations="province_code",
        color="Emissions_Tons",
        color_continuous_scale=_select_color_scale_for_indicator("Emissions_Tons", current_theme),
    )
    
    emissions_map.update_geos(fitbounds="locations", visible=False)
    return compact(emissions_map, current_theme, h=380, hide_cbar=False)


def create_poverty_choropleth(yearly_provincial_data: pd.DataFrame) -> go.Figure:
    """
    Generate specialized choropleth visualization for provincial poverty rate analysis.
    
    This function creates sophisticated thematic maps specifically designed for
    socioeconomic indicator visualization, implementing research-grade presentation
    standards for poverty data communication in policy and academic contexts.
    
    Args:
        yearly_provincial_data (pd.DataFrame): Single-year provincial dataset containing:
            - province_code: Official BPS provincial identification codes  
            - "Poverty_Rate_Percent": Poverty rate percentages for choropleth coloring
            
    Returns:
        go.Figure: Interactive poverty choropleth with the following features:
            - Socioeconomic-optimized color scaling (blue gradients)
            - Enhanced visualization height for detailed provincial analysis
            - Quantitative color bar for precise data interpretation
            - Responsive theme integration for consistent dashboard styling
            
    Design Methodology:
        - Implements cool color schemes (blues) for socioeconomic neutrality
        - Uses graduated color intensity to represent poverty severity levels  
        - Maintains scientific objectivity in color choice and scaling
        - Optimizes for both light and dark theme environments
        
    Analytical Applications:
        - Regional poverty assessment and comparison
        - Social development planning and resource allocation
        - Policy impact evaluation across provinces
        - Academic research on socioeconomic disparities
        
    Technical Features:
        - Automatic theme detection and color adaptation
        - Geographic boundary fitting for optimal provincial visibility
        - Compact layout optimization while preserving analytical detail
        - Integration with Indonesian official administrative boundaries
        
    Example:
        >>> poverty_data_2022 = df[df['year'] == 2022]
        >>> poverty_map = create_poverty_choropleth(poverty_data_2022)
        >>> st.plotly_chart(poverty_map, use_container_width=True)
    """
    current_theme = get_theme()
    
    poverty_map = px.choropleth(
        yearly_provincial_data,
        geojson=INDONESIA_GEOJSON_URL,
        featureidkey="properties.kode",
        locations="province_code",
        color="Poverty_Rate_Percent",
        color_continuous_scale=_select_color_scale_for_indicator("Poverty_Rate_Percent", current_theme),
    )
    
    poverty_map.update_geos(fitbounds="locations", visible=False)
    return compact(poverty_map, current_theme, h=380, hide_cbar=False)


def create_provincial_temporal_heatmap(time_series_data: pd.DataFrame, selected_indicator: str) -> go.Figure:
    """
    Generate advanced spatiotemporal heatmap for multi-dimensional indicator analysis.
    
    This function creates sophisticated matrix visualizations that reveal temporal patterns
    across provincial dimensions, enabling comprehensive analysis of indicator evolution
    and regional development trajectories over extended time periods.
    
    Args:
        time_series_data (pd.DataFrame): Multi-year provincial dataset containing:
            - province: Provincial names for y-axis categorization
            - year: Temporal dimension for x-axis progression
            - selected_indicator: Target indicator values for heatmap intensity
        selected_indicator (str): Column name of the indicator to be visualized
        
    Returns:
        go.Figure: Interactive spatiotemporal heatmap featuring:
            - Province-by-year matrix layout for comprehensive temporal analysis
            - Indicator-specific color scaling for optimal data interpretation
            - Enhanced visualization height for detailed provincial identification
            - Quantitative color bar for precise value interpretation
            
    Analytical Methodology:
        - Implements pivot table transformation for matrix data structure
        - Applies summation aggregation for consistent data consolidation
        - Uses automatic aspect ratio optimization for temporal data
        - Integrates indicator-specific color scaling for enhanced interpretability
        
    Advanced Features:
        - Dynamic color scale selection based on indicator characteristics
        - Adaptive layout optimization for varying dataset dimensions
        - Theme-responsive styling for consistent dashboard integration
        - Interactive capabilities for detailed data exploration
        
    Research Applications:
        - Longitudinal analysis of regional development patterns
        - Policy impact assessment across temporal and spatial dimensions
        - Identification of regional convergence or divergence trends
        - Strategic planning based on historical trajectory analysis
        
    Technical Implementation:
        - Utilizes pandas pivot_table for efficient data restructuring
        - Implements plotly imshow for high-performance heatmap rendering
        - Applies compact layout optimization while preserving analytical detail
        - Maintains scientific color mapping standards for accurate interpretation
        
    Example:
        >>> multi_year_data = df[(df['year'] >= 2015) & (df['year'] <= 2022)]
        >>> heatmap = create_provincial_temporal_heatmap(
        ...     multi_year_data, "Emissions_Tons"
        ... )
        >>> st.plotly_chart(heatmap, use_container_width=True)
    """
    current_theme = get_theme()
    
    # Transform data into matrix format for heatmap visualization
    provincial_temporal_matrix = time_series_data.pivot_table(
        index="province",
        columns="year",
        values=selected_indicator,
        aggfunc="sum",
    )
    
    # Generate heatmap with indicator-specific color scaling
    spatiotemporal_heatmap = px.imshow(
        provincial_temporal_matrix,
        color_continuous_scale=_select_color_scale_for_indicator(selected_indicator, current_theme),
        aspect="auto",
        labels=dict(color=selected_indicator),
    )
    
    return compact(spatiotemporal_heatmap, current_theme, h=430, hide_cbar=False)
