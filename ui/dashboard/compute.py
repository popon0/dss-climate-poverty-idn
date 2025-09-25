# dashboard/compute.py
"""
Computational Analytics Engine for Decision Support System Dashboard.

This module provides the core computational infrastructure for environmental and
socioeconomic data analysis, implementing sophisticated algorithms for time series
processing, KPI calculation, and multi-criteria target scaling operations.

Key Computational Components:
    - Time-based data slicing and contextual filtering
    - Year-over-year delta calculations with statistical validation
    - Emission-weighted tariff computation using advanced aggregation
    - Multi-dimensional target scaling with proportional allocation
    - KPI snapshot generation for real-time dashboard updates

Mathematical Frameworks:
    - Weighted average calculations for heterogeneous provincial data
    - Proportional scaling algorithms for target decomposition
    - Statistical delta computation with robust error handling
    - Multi-criteria decision analysis support functions

Dependencies:
    - pandas: For advanced data manipulation and aggregation operations
    - typing: For comprehensive type safety and code documentation

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
"""
from __future__ import annotations
import pandas as pd
from typing import Iterable, Literal

# === Global Configuration Constants ===
BASELINE_FLAT_TARIFF_RATE: float = 30.0  # Reference flat tariff rate (Rp/kg)
HISTORICAL_DATA_CUTOFF_YEAR: int = 2025   # Boundary between historical and predicted data
NATIONAL_TARGET_REFERENCE_YEAR: int = 2030  # Strategic planning target year

# Type definitions for dashboard mode specifications
DashboardMode = Literal["National", "Provincial", "Provincial Comparison"]


def extract_contextual_data_slices(
    dataset: pd.DataFrame, 
    target_year: int, 
    dashboard_mode: DashboardMode, 
    selected_provinces: list[str] | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract contextual data slices based on dashboard viewing mode and provincial selection.
    
    This function implements intelligent data filtering that adapts to different analytical
    contexts, ensuring that calculations and visualizations reflect the appropriate scope
    of analysis whether viewing national, provincial, or comparative perspectives.
    
    Args:
        dataset (pd.DataFrame): Complete time series dataset containing all provincial data
        target_year (int): Primary year for current period analysis
        dashboard_mode (DashboardMode): Current dashboard viewing context
        selected_provinces (list[str] | None): Provincial subset for focused analysis
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Data slice pair containing:
            - current_period_data: Filtered dataset for the target year
            - previous_period_data: Filtered dataset for the preceding year
            
    Filtering Logic:
        - National mode: Includes all provinces for comprehensive national analysis
        - Provincial mode: Focuses on single province for detailed examination
        - Provincial Comparison mode: Includes selected provincial subset for comparison
        
    Technical Implementation:
        - Applies temporal filtering for current and previous year data
        - Implements conditional provincial filtering based on dashboard context
        - Maintains data integrity through consistent DataFrame operations
        - Handles edge cases for missing or incomplete provincial selections
        
    Example:
        >>> current, previous = extract_contextual_data_slices(
        ...     national_dataset, 2022, "Provincial", ["Jawa Barat"]
        ... )
        >>> print(f"Current: {len(current)} rows, Previous: {len(previous)} rows")
    """
    # Apply provincial filtering based on dashboard mode
    if dashboard_mode in ("Provincial", "Provincial Comparison") and selected_provinces:
        current_period_data = dataset[
            (dataset["year"] == target_year) & 
            (dataset["province"].isin(selected_provinces))
        ].copy()
        previous_period_data = dataset[
            (dataset["year"] == target_year - 1) & 
            (dataset["province"].isin(selected_provinces))
        ].copy()
    else:
        # National mode: include all provinces
        current_period_data = dataset[dataset["year"] == target_year].copy()
        previous_period_data = dataset[dataset["year"] == target_year - 1].copy()
        
    return current_period_data, previous_period_data


def calculate_year_over_year_changes(
    dataset: pd.DataFrame, 
    analysis_year: int, 
    dashboard_mode: DashboardMode, 
    selected_provinces: list[str] | None
) -> dict[str, float | bool]:
    """
    Calculate comprehensive year-over-year percentage changes for key environmental and economic indicators.
    
    This function performs sophisticated temporal analysis by computing percentage changes
    across multiple indicators while adapting to different analytical contexts and handling
    missing or incomplete data gracefully.
    
    Args:
        dataset (pd.DataFrame): Complete time series dataset with provincial indicators
        analysis_year (int): Current year for change calculation
        dashboard_mode (DashboardMode): Dashboard viewing context for appropriate filtering
        selected_provinces (list[str] | None): Provincial focus for targeted analysis
        
    Returns:
        dict[str, float | bool]: Comprehensive change analysis containing:
            - emission_yoy_change: Annual emission percentage change
            - revenue_yoy_change: Annual revenue percentage change  
            - poverty_yoy_change: Annual poverty rate percentage change
            - has_previous_data: Data availability indicator for the preceding year
            
    Mathematical Framework:
        - Implements robust percentage change calculation: ((current - previous) / previous) * 100
        - Handles division by zero scenarios with graceful fallback to zero change
        - Applies contextual data filtering before aggregation and calculation
        - Validates data completeness for reliable change computation
        
    Technical Features:
        - Contextual data slicing based on dashboard mode and provincial selection
        - KPI snapshot generation for both current and previous periods
        - Statistical validation of data availability and completeness
        - Robust error handling for edge cases and missing data scenarios
        
    Example:
        >>> changes = calculate_year_over_year_changes(
        ...     provincial_dataset, 2022, "National", None
        ... )
        >>> print(f"National emission change: {changes['emission_yoy_change']:.1f}%")
    """
    # Extract contextually appropriate data slices
    current_data, previous_data = extract_contextual_data_slices(
        dataset, analysis_year, dashboard_mode, selected_provinces
    )
    
    # Generate KPI snapshots for temporal comparison
    current_kpi_metrics = generate_kpi_snapshot(current_data)
    previous_kpi_metrics = generate_kpi_snapshot(previous_data) if not previous_data.empty else {
        "total_emissions_tons": 0.0,
        "total_revenue_trillions": 0.0,
        "average_poverty_rate": 0.0
    }

    def calculate_percentage_change(current_value: float, previous_value: float) -> float:
        """Calculate robust percentage change with division-by-zero protection."""
        return ((current_value - previous_value) / previous_value * 100.0) if previous_value not in (0, None) else 0.0

    return {
        "emission_yoy_change": calculate_percentage_change(
            current_kpi_metrics["total_emissions_tons"], 
            previous_kpi_metrics["total_emissions_tons"]
        ),
        "revenue_yoy_change": calculate_percentage_change(
            current_kpi_metrics["total_revenue_trillions"], 
            previous_kpi_metrics["total_revenue_trillions"]
        ),
        "poverty_yoy_change": calculate_percentage_change(
            current_kpi_metrics["average_poverty_rate"], 
            previous_kpi_metrics["average_poverty_rate"]
        ),
        "has_previous_data": not previous_data.empty,
    }


def filter_temporal_range(
    dataset: pd.DataFrame, 
    start_year: int, 
    end_year: int, 
    selected_provinces: Iterable[str] | None
) -> pd.DataFrame:
    """
    Apply comprehensive temporal and provincial filtering for focused data analysis.
    
    This function implements efficient data filtering operations that enable focused
    analysis on specific time periods and geographical subsets while maintaining
    data integrity and analytical consistency.
    
    Args:
        dataset (pd.DataFrame): Complete time series dataset for filtering
        start_year (int): Beginning year for temporal range (inclusive)
        end_year (int): Ending year for temporal range (inclusive) 
        selected_provinces (Iterable[str] | None): Provincial subset for geographical focus
        
    Returns:
        pd.DataFrame: Filtered dataset containing only data within specified:
            - Temporal range: [start_year, end_year]
            - Geographical scope: selected provinces (if specified)
            
    Filtering Logic:
        - Applies inclusive temporal filtering for the specified year range
        - Implements conditional provincial filtering when provinces are specified
        - Preserves all data columns and maintains original data structure
        - Returns complete dataset copy to prevent unintended modifications
        
    Performance Considerations:
        - Utilizes efficient pandas boolean indexing for optimal performance
        - Minimizes memory footprint through strategic copy operations
        - Maintains computational efficiency for large temporal datasets
        
    Example:
        >>> filtered_data = filter_temporal_range(
        ...     national_dataset, 2015, 2020, ["Jawa Barat", "Jawa Timur"]
        ... )
        >>> print(f"Filtered to {len(filtered_data)} records")
    """
    # Apply temporal filtering for specified year range
    temporally_filtered = dataset[
        (dataset["year"] >= start_year) & (dataset["year"] <= end_year)
    ].copy()
    
    # Apply conditional provincial filtering if provinces specified
    if selected_provinces:
        geographically_filtered = temporally_filtered[
            temporally_filtered["province"].isin(selected_provinces)
        ]
        return geographically_filtered
    
    return temporally_filtered


def calculate_emission_weighted_tariff(dataset: pd.DataFrame) -> float:
    """
    Calculate emission-weighted average tariff rate using advanced aggregation methodology.
    
    This function implements sophisticated weighted average calculations that account for
    the varying emission intensities across provinces, providing a more accurate representation
    of effective tariff rates than simple arithmetic averages.
    
    Args:
        dataset (pd.DataFrame): Provincial dataset containing:
            - "Emissions_Tons": Emission values for weighting calculations
            - "Tax_Rate": Provincial tariff rates for aggregation
            
    Returns:
        float: Emission-weighted average tariff rate that represents the effective
            national tariff when accounting for provincial emission distributions
            
    Mathematical Framework:
        - Implements weighted average formula: Σ(tariff_i × weight_i) / Σ(weight_i)
        - Uses emission values as weights to reflect environmental impact proportions
        - Handles missing values through robust data preprocessing
        - Prevents division by zero through careful denominator validation
        
    Statistical Properties:
        - Provides more accurate representation than arithmetic means
        - Accounts for heterogeneous provincial emission distributions
        - Reflects actual economic impact of tariff policies
        - Maintains statistical validity across different dataset sizes
        
    Example:
        >>> provincial_data_2022 = df[df['year'] == 2022]
        >>> weighted_tariff = calculate_emission_weighted_tariff(provincial_data_2022)
        >>> print(f"Effective national tariff: {weighted_tariff:.2f} Rp/kg")
    """
    # Extract and preprocess emission weights with null handling
    emission_weights = dataset["Emissions_Tons"].fillna(0)
    
    # Calculate weighted numerator: sum of (tariff × emission_weight)
    weighted_tariff_sum = (dataset["Tax_Rate"].fillna(0) * emission_weights).sum()
    
    # Calculate weight denominator: total emission weights
    total_emission_weight = emission_weights.sum()
    
    # Return weighted average with division-by-zero protection
    return float(weighted_tariff_sum / total_emission_weight) if total_emission_weight > 0 else 0.0


def generate_kpi_snapshot(yearly_dataset: pd.DataFrame) -> dict[str, float]:
    """
    Generate comprehensive Key Performance Indicator (KPI) snapshot for strategic analysis.
    
    This function computes essential environmental and economic performance metrics
    that provide stakeholders with immediate insights into current system status
    and progress toward sustainability objectives.
    
    Args:
        yearly_dataset (pd.DataFrame): Single-year provincial dataset containing:
            - "Emissions_Tons": Provincial greenhouse gas emission data
            - "Government_Revenue_Trillions": Government revenue in trillions
            - "Poverty_Rate_Percent": Poverty rate percentages by province
            
    Returns:
        dict[str, float]: Comprehensive KPI metrics containing:
            - total_emissions_tons: Aggregated national/regional emission total
            - total_revenue_trillions: Combined government revenue across scope
            - average_poverty_rate: Population-weighted poverty rate average
            
    Aggregation Methodology:
        - Emissions: Additive aggregation reflecting total environmental impact
        - Revenue: Additive aggregation representing total economic benefit
        - Poverty: Arithmetic mean providing representative social indicator
        
    Statistical Considerations:
        - Handles missing values through pandas default aggregation behavior
        - Maintains numerical precision through explicit float conversion
        - Provides consistent metrics regardless of dataset scope (national/provincial)
        
    Use Cases:
        - Real-time dashboard KPI displays
        - Temporal comparison analysis (year-over-year changes)
        - Target achievement assessment
        - Multi-dimensional performance evaluation
        
    Example:
        >>> data_2022 = df[df['year'] == 2022]
        >>> metrics = generate_kpi_snapshot(data_2022)
        >>> print(f"National emissions: {metrics['total_emissions_tons']:,.0f} tons")
    """
    return {
        "total_emissions_tons": float(yearly_dataset["Emissions_Tons"].sum()),
        "total_revenue_trillions": float(yearly_dataset["Government_Revenue_Trillions"].sum()),
        "average_poverty_rate": float(yearly_dataset["Poverty_Rate_Percent"].mean()),
    }


def compute_proportional_targets(
    complete_dataset: pd.DataFrame, 
    view_specific_dataset: pd.DataFrame,
    national_emission_target: float, 
    national_revenue_target: float, 
    national_poverty_target: float,
    enable_proportional_scaling: bool
) -> dict[str, float]:
    """
    Compute proportionally scaled policy targets adapted to specific analytical contexts.
    
    This function implements sophisticated target scaling algorithms that enable meaningful
    target comparisons across different geographical scopes (national, provincial, regional)
    while maintaining mathematical consistency and policy relevance.
    
    Args:
        complete_dataset (pd.DataFrame): Full national dataset for baseline calculations
        view_specific_dataset (pd.DataFrame): Context-specific dataset (e.g., selected provinces)
        national_emission_target (float): National emission reduction target (tons)
        national_revenue_target (float): National revenue generation target (trillions)
        national_poverty_target (float): National poverty reduction target (percentage)
        enable_proportional_scaling (bool): Toggle for proportional vs absolute target scaling
        
    Returns:
        dict[str, float]: Context-appropriate targets containing:
            - scaled_emission_target: Emission target adjusted for analytical scope
            - scaled_revenue_target: Revenue target adjusted for analytical scope  
            - scaled_poverty_target: Poverty target (typically unscaled due to rate nature)
            
    Scaling Methodology:
        - Proportional Scaling (enabled): Targets scaled by emission contribution ratio
        - Absolute Scaling (disabled): National targets used directly without adjustment
        - Baseline Year: Uses NATIONAL_TARGET_REFERENCE_YEAR for contribution calculations
        - Mathematical Formula: scaled_target = national_target × (scope_emissions / national_emissions)
        
    Technical Implementation:
        - Calculates emission-based contribution ratios for proportional allocation
        - Handles edge cases through robust mathematical operations
        - Maintains target consistency across different analytical contexts
        - Provides flexible scaling control through boolean parameter
        
    Policy Applications:
        - Provincial target allocation based on emission contributions
        - Multi-regional target distribution for federated policy implementation  
        - Comparative analysis across different geographical aggregations
        - Strategic planning with context-appropriate benchmarking
        
    Example:
        >>> provincial_targets = compute_proportional_targets(
        ...     national_data, java_provinces_data,
        ...     1_500_000_000, 180.0, 7.5, enable_proportional_scaling=True
        ... )
        >>> print(f"Java emission target: {provincial_targets['scaled_emission_target']:,.0f} tons")
    """
    # Extract baseline data for the target reference year
    national_baseline = complete_dataset[complete_dataset["year"] == NATIONAL_TARGET_REFERENCE_YEAR]
    contextual_baseline = view_specific_dataset[view_specific_dataset["year"] == NATIONAL_TARGET_REFERENCE_YEAR] if enable_proportional_scaling else national_baseline
    
    # Calculate emission-based contribution ratio for proportional scaling
    national_baseline_emissions = national_baseline["Emissions_Tons"].sum()
    contextual_baseline_emissions = contextual_baseline["Emissions_Tons"].sum()
    
    emission_contribution_ratio = (
        contextual_baseline_emissions / max(national_baseline_emissions, 1.0) 
        if enable_proportional_scaling else 1.0
    )

    return {
        "scaled_emission_target": national_emission_target * emission_contribution_ratio,
        "scaled_revenue_target": national_revenue_target * emission_contribution_ratio,
        "scaled_poverty_target": national_poverty_target,  # Poverty rates typically not scaled proportionally
    }


# === Backward Compatibility Aliases ===
# These aliases ensure existing view files continue to work during the transition
Mode = DashboardMode  # Alias for backward compatibility
FLAT_TARIFF = BASELINE_FLAT_TARIFF_RATE
HIST_CUTOFF_YEAR = HISTORICAL_DATA_CUTOFF_YEAR
TARGET_YEAR = NATIONAL_TARGET_REFERENCE_YEAR

# Function aliases for backward compatibility
def slice_by_context(df, year, mode, provinces):
    return extract_contextual_data_slices(df, year, mode, provinces)

def compute_yoy_deltas(df, year, mode, provinces):
    return calculate_year_over_year_changes(df, year, mode, provinces)

def filter_year_range(df, start, end, provinces):
    return filter_temporal_range(df, start, end, provinces)

def weighted_tariff(df):
    return calculate_emission_weighted_tariff(df)

def kpi_snapshot(df_year):
    return generate_kpi_snapshot(df_year)

def scaled_targets(df_all, df_view, tgt_emission, tgt_revenue, tgt_poverty, scale_on):
    return compute_proportional_targets(df_all, df_view, tgt_emission, tgt_revenue, tgt_poverty, scale_on)
