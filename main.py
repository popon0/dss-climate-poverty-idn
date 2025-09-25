# Copyright 2025 Teuku Hafiez Ramadhan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
main.py

Comprehensive Training Pipeline for Decision Support System (DSS) Machine Learning Models.

This module orchestrates the complete model training workflow for the DSS project,
implementing a dual-model architecture consisting of LSTM (Long Short-Term Memory)
networks for time series forecasting and ANFIS (Adaptive Neuro-Fuzzy Inference
System) for policy effectiveness prediction.

Training Pipeline Architecture:
    1. Data Loading & Preprocessing: Load and normalize raw environmental and 
       socioeconomic datasets from multiple sources
    2. Feature Engineering: Apply min-max normalization and create sequential 
       data structures optimized for time series learning
    3. LSTM Model Training: Train recurrent neural networks for emission and 
       poverty trajectory forecasting
    4. ANFIS Model Training: Train neuro-fuzzy systems for policy score prediction
    5. Model Evaluation: Comprehensive performance assessment using multiple metrics
    6. Artifact Persistence: Save trained models and performance summaries

Training Modes:
    - from-raw: Build complete dataset from raw CSV files with full preprocessing
    - from-processed: Use pre-computed normalized dataset for expedited training

Output Artifacts:
    - Trained LSTM model parameters (outputs/lstm_model.pt)
    - Trained ANFIS model parameters (outputs/anfis_model.pt)  
    - Comprehensive training metrics summary (outputs/training_summary.csv)

Dependencies:
    - PyTorch: Deep learning framework for model implementation and training
    - pandas: Structured data manipulation and analysis
    - numpy: Numerical computing operations
    - scikit-learn: Machine learning utilities and evaluation metrics

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
"""

from __future__ import annotations

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch

from config import (
    TRAINED_LSTM_MODEL_PATH,
    TRAINED_ANFIS_MODEL_PATH,
    MODEL_OUTPUT_DIR,
)

from dataprep.preprocessor import normalize_features, create_lstm_sequences
from dataprep.target import calculate_policy_effectiveness_score
from dataprep.province_reference import INDONESIAN_PROVINCE_CODES  # optional if needed
from config import load_dataset_files

from models.lstm_model import (
    LSTMForecast,
    train_lstm_model,
    evaluate_lstm_model,
)
from models.anfis_model import ANFIS, train_anfis_model


# === Essential Utility Functions ===
def ensure_output_directory_exists() -> None:
    """
    Ensure that the model output directory exists for saving trained artifacts.
    
    Creates the output directory structure if it does not already exist,
    preventing file I/O errors during model persistence operations.
    """
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def configure_reproducible_training_environment(random_seed: int = 42) -> None:
    """
    Configure deterministic training environment for reproducible model training results.
    
    Sets random seeds across all relevant libraries (Python, NumPy, PyTorch) to ensure
    consistent training outcomes across multiple runs, which is essential for scientific
    reproducibility and model comparison studies.
    
    Args:
        random_seed (int, optional): Seed value for random number generators. 
                                   Defaults to 42 for conventional consistency.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


# === Data Preparation Pipeline Functions ===
def load_and_prepare_raw_datasets() -> pd.DataFrame:
    """
    Load and comprehensively prepare dataset from raw CSV data sources.
    
    This function implements a complete data preprocessing pipeline that transforms
    raw environmental and socioeconomic data into a normalized, analysis-ready format
    suitable for machine learning model training.
    
    Data Processing Steps:
        1. Load raw emission and poverty datasets from CSV files
        2. Merge datasets on common keys [province_id, year]
        3. Standardize column names to consistent format
        4. Apply min-max normalization using historical boundaries
        5. Calculate composite policy effectiveness scores for ANFIS training
        
    Returns:
        pd.DataFrame: Fully preprocessed dataset with normalized features and
                     calculated policy scores ready for model training
                     
    Raises:
        FileNotFoundError: If required raw data files cannot be located
        ValueError: If data merging fails due to incompatible schemas
        
    Note:
        This function is computationally intensive and should be used when
        complete data preprocessing is required from scratch.
    """
    print("📥 Loading raw environmental and socioeconomic datasets...")
    _, emissions_df, poverty_df = load_dataset_files()

    # Merge datasets on province and year dimensions
    merged_dataset = pd.merge(
        emissions_df, poverty_df,
        on=["province_id", "year"],
        how="inner",
        suffixes=("_emissions", "_poverty")
    )

    # Standardize column nomenclature for consistency
    merged_dataset["province"] = merged_dataset["province_emissions"]
    merged_dataset = merged_dataset.rename(columns={
        "Net_Emissions_Tons": "Emissions_Tons",
        "Poverty_Rate_Percent": "Poverty_Rate_Percent"
    })
    merged_dataset = merged_dataset[["province_id", "province", "year", "Emissions_Tons", "Poverty_Rate_Percent"]]

    # Apply normalization using historical data boundaries
    min_emissions, max_emissions = merged_dataset["Emissions_Tons"].min(), merged_dataset["Emissions_Tons"].max()
    min_poverty, max_poverty = merged_dataset["Poverty_Rate_Percent"].min(), merged_dataset["Poverty_Rate_Percent"].max()
    merged_dataset = normalize_features(merged_dataset, min_emissions, max_emissions, min_poverty, max_poverty)

    # Calculate composite policy effectiveness scores
    merged_dataset = calculate_policy_effectiveness_score(merged_dataset)
    return merged_dataset


def load_and_prepare_processed_datasets() -> pd.DataFrame:
    """
    Load and prepare dataset from preprocessed CSV files for expedited training.
    
    This function provides a streamlined approach to data loading when normalized
    datasets are already available, significantly reducing preprocessing overhead
    while ensuring data integrity and completeness for model training.
    
    Data Validation Steps:
        1. Load preprocessed dataset with normalized features
        2. Validate presence of required columns for training
        3. Calculate policy effectiveness scores if missing
        4. Return training-ready dataset
        
    Returns:
        pd.DataFrame: Validated dataset with all required features for model training
        
    Raises:
        ValueError: If essential columns are missing from the processed dataset
        FileNotFoundError: If the processed data file cannot be located
        
    Note:
        This function assumes data preprocessing has been completed previously
        and focuses on validation and completeness checks.
    """
    print("📥 Loading preprocessed training dataset from scaled data files...")
    processed_df, _, _ = load_dataset_files()

    # Validate essential column presence for model training
    required_columns = {
        "province_id", "province", "year",
        "scaled_net_emissions", "scaled_poverty_rate"
    }
    missing_columns = required_columns - set(processed_df.columns)
    if missing_columns:
        raise ValueError(f"Missing essential columns: {missing_columns}. Required columns: {required_columns}")

    # Calculate policy effectiveness scores if not present
    if "policy_effectiveness_score" not in processed_df.columns:
        processed_df = calculate_policy_effectiveness_score(processed_df)

    return processed_df


# === Model Training and Evaluation Functions ===
def train_and_evaluate_lstm_model(training_dataset: pd.DataFrame):
    """
    Train and comprehensively evaluate LSTM model for time series forecasting.
    
    This function implements the complete LSTM training pipeline, including data
    sequencing, model instantiation, supervised learning, and performance evaluation
    using multiple regression metrics.
    
    Args:
        training_dataset (pd.DataFrame): Preprocessed dataset with normalized features
                                       sorted by province and year for sequence creation
                                       
    Returns:
        tuple: Trained LSTM model and comprehensive evaluation metrics dictionary
               containing MAE, MSE, RMSE, and R² scores for model assessment
               
    Training Process:
        1. Create temporal sequences suitable for LSTM architecture
        2. Initialize LSTM model with optimal hyperparameters  
        3. Execute supervised training with backpropagation
        4. Evaluate model performance on training sequences
        5. Return trained model with performance metrics
    """
    # Prepare sequential data structures for LSTM training
    dataset_sorted = training_dataset.sort_values(["province_id", "year"])
    input_sequences, target_sequences = create_lstm_sequences(dataset_sorted)

    # Initialize and train LSTM forecasting model
    lstm_forecaster = LSTMForecast()
    print("🔁 Training LSTM model for time series forecasting...")
    trained_lstm_model = train_lstm_model(lstm_forecaster, input_sequences, target_sequences)

    # Comprehensive model evaluation
    print("✅ Evaluating LSTM model performance:")
    evaluation_metrics = evaluate_lstm_model(trained_lstm_model, input_sequences, target_sequences)
    for metric_name, metric_value in evaluation_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    return trained_lstm_model, evaluation_metrics


def train_and_evaluate_anfis_model(training_dataset: pd.DataFrame):
    """
    Train and evaluate ANFIS model for policy effectiveness prediction.
    
    This function implements the complete ANFIS (Adaptive Neuro-Fuzzy Inference
    System) training pipeline for predicting policy effectiveness scores based on
    normalized environmental and socioeconomic indicators.
    
    Args:
        training_dataset (pd.DataFrame): Dataset containing normalized features and
                                       calculated policy effectiveness scores
                                       
    Returns:
        tuple: Trained ANFIS model, evaluation metrics dictionary, and enhanced dataset
               with any additional computed features
               
    Training Process:
        1. Prepare input features (scaled_net_emissions, scaled_poverty_rate)
        2. Extract policy effectiveness scores as training targets
        3. Initialize ANFIS with Gaussian membership functions
        4. Execute neuro-fuzzy training with gradient descent
        5. Evaluate model accuracy using regression metrics
    """
    # Prepare input features for ANFIS training
    input_features = torch.tensor(
        training_dataset[["scaled_net_emissions", "scaled_poverty_rate"]].values,
        dtype=torch.float32
    )
    
    # Ensure policy effectiveness scores are available
    if "policy_effectiveness_score" not in training_dataset.columns:
        training_dataset = calculate_policy_effectiveness_score(training_dataset)
    
    target_scores = torch.tensor(
        training_dataset["policy_effectiveness_score"].values.reshape(-1, 1),
        dtype=torch.float32
    )

    # Initialize and train ANFIS model
    anfis_model = ANFIS()
    print("🔁 Training ANFIS model for policy effectiveness prediction...")
    trained_anfis_model = train_anfis_model(anfis_model, input_features, target_scores)

    # Comprehensive model evaluation
    with torch.no_grad():
        predicted_scores = trained_anfis_model(input_features).numpy().squeeze()
        actual_scores = target_scores.numpy().squeeze()
        
        # Calculate evaluation metrics
        mean_squared_error = float(np.mean((predicted_scores - actual_scores) ** 2))
        mean_absolute_error = float(np.mean(np.abs(predicted_scores - actual_scores)))

    print("✅ Evaluating ANFIS model performance:")
    print(f"  Mean Squared Error: {mean_squared_error:.6f}")
    print(f"  Mean Absolute Error: {mean_absolute_error:.6f}")

    evaluation_metrics = {"mse": mean_squared_error, "mae": mean_absolute_error}
    return trained_anfis_model, evaluation_metrics, training_dataset


# === Main Training Pipeline Orchestration ===
def execute_training_pipeline(training_mode: str) -> None:
    """
    Execute the complete machine learning model training pipeline.
    
    This function orchestrates the entire training workflow, from data preparation
    through model training to artifact persistence, supporting both raw data
    processing and preprocessed data workflows.
    
    Args:
        training_mode (str): Training data source mode:
                           - "from-raw": Complete preprocessing from raw CSV files
                           - "from-processed": Use preprocessed normalized data
                           
    Raises:
        ValueError: If an unsupported training mode is specified
        
    Training Pipeline:
        1. Environment configuration for reproducibility
        2. Data loading and preparation based on specified mode
        3. LSTM model training and evaluation for time series forecasting
        4. ANFIS model training and evaluation for policy effectiveness prediction
        5. Model artifact persistence with comprehensive metadata
        6. Training summary generation with performance metrics
    """
    # Initialize training environment
    ensure_output_directory_exists()
    configure_reproducible_training_environment(42)

    # Data preparation based on training mode
    if training_mode == "from-raw":
        training_dataset = load_and_prepare_raw_datasets()
    elif training_mode == "from-processed":
        training_dataset = load_and_prepare_processed_datasets()
    else:
        raise ValueError("Unsupported training mode. Use 'from-raw' or 'from-processed'.")

    # Train LSTM model for time series forecasting
    lstm_model, lstm_metrics = train_and_evaluate_lstm_model(training_dataset)

    # Train ANFIS model for policy effectiveness prediction  
    anfis_model, anfis_metrics, _ = train_and_evaluate_anfis_model(training_dataset)

    # Persist trained model artifacts
    torch.save(lstm_model.state_dict(), TRAINED_LSTM_MODEL_PATH)
    torch.save(anfis_model.state_dict(), TRAINED_ANFIS_MODEL_PATH)
    print(f"💾 Trained models saved: {TRAINED_LSTM_MODEL_PATH}, {TRAINED_ANFIS_MODEL_PATH}")

    # Generate comprehensive training summary
    training_summary = {
        "training_mode": training_mode,
        **{f"lstm_{metric}": value for metric, value in lstm_metrics.items()},
        **{f"anfis_{metric}": value for metric, value in anfis_metrics.items()},
    }
    
    summary_dataframe = pd.DataFrame([training_summary])
    summary_path = os.path.join(MODEL_OUTPUT_DIR, "training_summary.csv")
    summary_dataframe.to_csv(summary_path, index=False)
    print(f"📑 Training metrics summary saved: {summary_path}")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="Decision Support System Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Mode Options:
  from-raw        Build complete dataset from raw CSV files with full preprocessing
  from-processed  Use preprocessed normalized dataset for expedited training

Example Usage:
  python main.py --mode from-processed
  python main.py --mode from-raw
        """
    )
    argument_parser.add_argument(
        "--mode", 
        type=str, 
        default="from-processed",
        choices=["from-processed", "from-raw"],
        help="Data loading and preprocessing mode (default: from-processed)"
    )
    
    parsed_arguments = argument_parser.parse_args()
    execute_training_pipeline(parsed_arguments.mode)
