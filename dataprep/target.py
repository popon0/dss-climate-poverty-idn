"""
target.py

Policy score computation utilities for the Decision Support System (DSS).

This module provides functions for calculating composite policy effectiveness
scores that serve as target variables for the ANFIS (Adaptive Neuro-Fuzzy
Inference System) model. The policy scores integrate multiple socioeconomic
and environmental indicators to assess overall policy performance.

Key Functions:
    - calculate_policy_effectiveness_score(): Computes weighted policy performance metric

Dependencies:
    - pandas: For structured data manipulation and DataFrame operations

Mathematical Framework:
    The policy score represents a balanced assessment of environmental protection
    (emissions reduction) and social welfare (poverty reduction) objectives,
    following multi-criteria decision analysis principles.

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
"""

from __future__ import annotations
import pandas as pd


def calculate_policy_effectiveness_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate composite policy effectiveness score as a weighted combination of
    environmental and socioeconomic performance indicators.
    
    This function computes a multi-criteria policy assessment score that balances
    environmental protection objectives (emission reduction) with social welfare
    goals (poverty alleviation). The score serves as a target variable for training
    the ANFIS model to predict optimal policy interventions.
    
    Mathematical Formula:
        policy_score = 0.6 × scaled_net_emissions + 0.4 × (1 - scaled_poverty_rate)
        
    Where:
        - scaled_net_emissions: Normalized emission intensity [0, 1] 
        - scaled_poverty_rate: Normalized poverty rate [0, 1]
        - Weight 0.6: Emphasizes environmental protection priority
        - Weight 0.4: Balances social welfare considerations
        - (1 - scaled_poverty_rate): Inverts poverty to reward reduction
    
    Args:
        dataframe (pd.DataFrame): Input dataset containing normalized indicators:
            - "scaled_net_emissions" (float): Normalized emission values [0, 1]
            - "scaled_poverty_rate" (float): Normalized poverty rates [0, 1]
            
    Returns:
        pd.DataFrame: Enhanced dataset with additional column:
            - "policy_effectiveness_score": Composite policy performance metric [0, 1]
            
    Note:
        Higher scores indicate better overall policy performance, combining
        both environmental sustainability and social equity objectives.
        The weighting scheme can be adjusted based on policy priorities.
        
    Example:
        >>> scored_df = calculate_policy_effectiveness_score(normalized_data)
        >>> print(f"Policy score range: [{scored_df['policy_effectiveness_score'].min():.3f}, "
        ...       f"{scored_df['policy_effectiveness_score'].max():.3f}]")
    """
    dataframe["policy_effectiveness_score"] = (
        0.6 * dataframe["scaled_net_emissions"] + 0.4 * (1 - dataframe["scaled_poverty_rate"])
    )
    return dataframe
