"""
preprocessor.py

Data preprocessing utilities for the Decision Support System (DSS).
            - "scaled_net_emissions": Normalized emission values  
            - "scaled_poverty_rate": Normalized poverty rate valueshis module provides essential data transformation functions for preparing
raw environmental and socioeconomic data for machine learning model training.
The preprocessing pipeline ensures consistent data scaling and sequential 
structuring required for time series forecasting models.

Key Functions:
    - normalize_features(): Applies min-max normalization to emission and poverty data
    - create_lstm_sequences(): Constructs sequential data structures for LSTM training
    
Dependencies:
    - numpy: For numerical computations and array operations
    - torch: For PyTorch tensor operations
    - pandas: For structured data manipulation

Author: Teuku Hafiez Ramadhan  
License: Apache License 2.0
"""

from __future__ import annotations

import numpy as np
import torch
import pandas as pd


def normalize_features(
    dataframe: pd.DataFrame,
    min_emissions: float,
    max_emissions: float,
    min_poverty_rate: float,
    max_poverty_rate: float,
) -> pd.DataFrame:
    """
    Apply min-max normalization to greenhouse gas emissions and poverty rate features.
    
    This function scales the raw environmental and socioeconomic indicators to a 
    standardized range [0, 1], which is essential for optimal machine learning 
    model performance and numerical stability during training.
    
    Args:
        dataframe (pd.DataFrame): Input dataset containing raw indicator columns:
                    Expected input structure:
            - "Emissions_Tons": Greenhouse gas emissions in tons
            - "Poverty_Rate_Percent": Poverty rate as percentage
        min_emissions (float): Historical minimum emission value for scaling reference
        max_emissions (float): Historical maximum emission value for scaling reference  
        min_poverty_rate (float): Historical minimum poverty rate for scaling reference
        max_poverty_rate (float): Historical maximum poverty rate for scaling reference
        
    Returns:
        pd.DataFrame: Enhanced dataframe with normalized feature columns:
            - scaled_net_emissions: Normalized emission values [0, 1]
            - scaled_poverty_rate: Normalized poverty rate values [0, 1]    Note:
        The original raw columns are preserved alongside the new scaled features
        to maintain data traceability and support inverse transformations.
        
    Example:
        >>> normalized_df = normalize_features(
        ...     raw_df, min_e=1000, max_e=50000, min_p=2.5, max_p=15.8
        ... )
        >>> print(normalized_df[['scaled_net_emissions', 'scaled_poverty_rate']].describe())
    """
    dataframe["scaled_net_emissions"] = (
        (dataframe["Emissions_Tons"] - min_emissions) / (max_emissions - min_emissions)
    )
    dataframe["scaled_poverty_rate"] = (
        (dataframe["Poverty_Rate_Percent"] - min_poverty_rate) / (max_poverty_rate - min_poverty_rate)
    )
    return dataframe


def create_lstm_sequences(
    sorted_dataframe: pd.DataFrame, sequence_length: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct sequential data structures optimized for LSTM time series training.
    
    This function transforms the normalized provincial time series data into 
    sliding window sequences suitable for supervised learning with LSTM networks.
    Each sequence captures temporal dependencies in emission and poverty patterns.
    
    Args:
        sorted_dataframe (pd.DataFrame): Time-ordered dataset sorted by [province_id, year]
            containing normalized feature columns:
            - "emissions_scaled": Normalized emission values
            - "poverty_scaled": Normalized poverty rate values  
        sequence_length (int, optional): Number of historical timesteps per input sequence.
            Defaults to 3, representing a 3-year lookback window.
            
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Training-ready tensor pair:
            - input_sequences (torch.Tensor): Shape (N, sequence_length, 2) 
              Input sequences for model training
            - target_values (torch.Tensor): Shape (N, 2)
              Target values for supervised learning
              
    Note:
        The function processes each province independently to maintain temporal
        coherence and prevent cross-provincial data leakage in the sequences.
        
    Example:
        >>> X_train, y_train = create_lstm_sequences(
        ...     normalized_provincial_data, sequence_length=3
        ... )
        >>> print(f"Input shape: {X_train.shape}, Target shape: {y_train.shape}")
        Input shape: torch.Size([420, 3, 2]), Target shape: torch.Size([420, 2])
    """
    input_sequences, target_values = [], []

    for province_id in sorted_dataframe["province_id"].unique():
        provincial_subset = sorted_dataframe[sorted_dataframe["province_id"] == province_id]
        feature_values = provincial_subset[["scaled_net_emissions", "scaled_poverty_rate"]].values

        # Create sliding window sequences for this province
        for timestep in range(len(feature_values) - sequence_length):
            sequence_input = feature_values[timestep : timestep + sequence_length]
            sequence_target = feature_values[timestep + sequence_length]
            
            input_sequences.append(sequence_input)
            target_values.append(sequence_target)

    # Convert to numpy arrays and then to PyTorch tensors
    input_array = np.array(input_sequences, dtype=np.float32)
    target_array = np.array(target_values, dtype=np.float32)

    return torch.from_numpy(input_array), torch.from_numpy(target_array)
