# 📘 Comprehensive Code Architecture and Execution Flow Documentation

This document provides a detailed technical specification of the code architecture and execution flow for the **Decision Support System (DSS)** repository. It systematically describes the transformation pipeline that converts raw environmental and socioeconomic datasets into predictive analytical outputs, visualized through an interactive web-based dashboard interface.

## 1. High-Level System Architecture Pipeline

### 1.1 Data Input Layer
- **Raw Environmental Data**: Greenhouse gas (GHG) emissions dataset (`data/raw/`)
- **Raw Socioeconomic Data**: Provincial poverty rate time series (`data/raw/`)

### 1.2 Data Preprocessing Layer (`dataprep/`)
- **Feature Normalization**: Min-max scaling of emissions and poverty indicators for optimal model performance
- **Sequential Data Construction**: Creation of temporal sequences optimized for LSTM neural network training
- **Policy Score Computation**: Calculation of composite policy effectiveness metrics for ANFIS training targets

### 1.3 Machine Learning Model Training Layer (`main.py`)
- **LSTM Model**: Long Short-Term Memory networks for temporal forecasting of emission and poverty trajectories
- **ANFIS Model**: Adaptive Neuro-Fuzzy Inference System for predicting adaptive carbon tax policy effectiveness scores

### 1.4 Predictive Analysis Layer (`predict.py`)
- **Future Scenario Forecasting**: Multi-year projections (2021–2030) based on trained model ensembles
- **Historical-Future Integration**: Seamless combination of observed data (2010–2020) with predicted scenarios
- **Policy Impact Assessment**: Computation of adaptive taxation scenarios and projected government revenues

### 1.5 Interactive Visualization Layer (`ui/`)
- **Streamlit Dashboard**: Web-based interface for interactive exploration and analysis of results
- **Multi-Scale Analytics**: National, provincial, and comparative visualization capabilities
- **Decision Support Tools**: Key Performance Indicators (KPIs), geospatial choropleth maps, temporal trend analysis, correlation heatmaps, and optimization utilities

---

## 2. Detailed Module Architecture and Functionality

### 2.1 Configuration Management (`config.py`)
**Purpose**: Centralized configuration management for dataset paths, model artifacts, output directories, default parameters, and utility functions.

**Key Components**:
- Base directory path definitions with cross-platform compatibility
- Dataset file path specifications for raw, processed, and final data stages
- Trained model artifact storage locations
- Data loading utility functions with comprehensive error handling
- Normalization boundary calculation utilities for feature scaling

### 2.2 Data Preprocessing Subsystem (`dataprep/`)

#### 2.2.1 Feature Engineering (`preprocessor.py`)
- **`normalize_features()`**: Implements min-max normalization with preservation of original data integrity
- **`create_lstm_sequences()`**: Constructs sliding window temporal sequences with configurable lookback periods

#### 2.2.2 Target Variable Engineering (`target.py`)
- **`calculate_policy_effectiveness_score()`**: Multi-criteria policy assessment using weighted environmental and social indicators

#### 2.2.3 Geographic Reference System (`province_reference.py`)
- **Official BPS Code Mapping**: Standardized provincial identification using Indonesian Statistical Bureau codes
- **Bidirectional Lookup Functions**: Forward and reverse mapping between province names and administrative codes

### 2.3 Machine Learning Model Architecture (`models/`)

#### 2.3.1 LSTM Forecasting Model (`lstm_model.py`)
- **Neural Architecture**: Multi-layer recurrent networks with dropout regularization
- **Training Pipeline**: Supervised learning with Adam optimization and early stopping
- **Evaluation Framework**: Comprehensive metrics including MAE, MSE, RMSE, and R²

#### 2.3.2 ANFIS Policy Model (`anfis_model.py`)
- **Fuzzy Logic System**: Sugeno-type inference with Gaussian membership functions
- **Neuro-Fuzzy Training**: Gradient-based parameter optimization for rule-based reasoning
- **Policy Score Prediction**: Adaptive taxation effectiveness assessment

### 2.4 Training Orchestration (`main.py`)
**Purpose**: Comprehensive training pipeline orchestration with support for multiple data loading modes and reproducible training environments.

**Key Features**:
- **Dual-Mode Data Processing**: Raw CSV preprocessing vs. pre-computed dataset loading
- **Sequential Model Training**: LSTM temporal forecasting followed by ANFIS policy prediction
- **Performance Evaluation**: Multi-metric assessment with detailed logging and reporting
- **Artifact Persistence**: Trained model serialization with comprehensive metadata

### 2.5 Prediction and Scenario Analysis (`predict.py`)
**Purpose**: Implementation of complete prediction pipeline for historical computation and future scenario forecasting.

**Capabilities**:
- **Historical Score Computation**: Retrospective policy effectiveness assessment (2010–2020)
- **Future Trajectory Forecasting**: Multi-year predictions using trained LSTM ensembles
- **Adaptive Policy Simulation**: Dynamic taxation scenario modeling with revenue projections
- **Integrated Dataset Generation**: Seamless temporal data fusion for comprehensive analysis

### 2.6 Interactive Dashboard System (`ui/`)

#### 2.6.1 Application Entry Point (`app.py`)
- **Multi-Modal Interface**: National overview, provincial analysis, and comparative assessment modes
- **Interactive Controls**: Dynamic filtering, parameter adjustment, and scenario exploration
- **Real-Time Data Integration**: Automatic detection of predicted vs. static datasets

#### 2.6.2 Visualization Components (`dashboard/`)
- **Geospatial Mapping**: Interactive choropleth visualizations with provincial detail
- **Temporal Analytics**: Multi-series time trend analysis with forecasting indicators
- **Correlation Analysis**: Heatmap visualization of indicator relationships
- **Performance Optimization**: Caching strategies and responsive layout management

---

## 3. Execution Workflow and Command Interface

### 3.1 Model Training Pipeline
```bash
# Complete training from raw data sources
python main.py --mode from-raw

# Expedited training using preprocessed datasets  
python main.py --mode from-processed
```

### 3.2 Prediction and Scenario Generation
```bash
# Generate comprehensive forecasting scenarios
python predict.py --start_year 2021 --end_year 2030
```

### 3.3 Interactive Dashboard Deployment
```bash
# Launch web-based decision support interface
streamlit run ui/app.py
```

---

## 4. Technical Implementation Specifications

### 4.1 Data Normalization Framework
**Methodology**: Min-max scaling ensures all environmental and socioeconomic indicators are transformed to [0, 1] range, which is essential for:
- **LSTM Convergence**: Numerical stability during recurrent neural network training
- **ANFIS Membership Functions**: Optimal fuzzy set boundary definition and rule activation

### 4.2 LSTM Neural Network Architecture
**Configuration**:
- **Input Structure**: Sliding window approach with 3-year temporal sequences (`t-3, t-2, t-1`)
- **Output Prediction**: Simultaneous forecasting of emission and poverty indicators at time `t`
- **Network Topology**: 64 hidden units, 2 stacked LSTM layers, 0.2 dropout regularization
- **Training Optimization**: Adam optimizer with 0.005 learning rate, 2000 epoch training duration

### 4.3 ANFIS Neuro-Fuzzy System Architecture
**Specifications**:
- **Input Variables**: Normalized emission intensity and poverty rate indicators
- **Membership Functions**: 3 Gaussian functions per input variable (9 comprehensive inference rules)
- **Inference Mechanism**: Sugeno-1 system with linear output layer combinations for continuous policy scoring
- **Rule Base Coverage**: Comprehensive input space mapping with balanced decision boundaries for optimal policy assessment

### 4.4 Output Artifact Specifications
**Trained Model Artifacts**:
- `outputs/lstm_model.pt`: Serialized LSTM state dictionary with trained parameters
- `outputs/anfis_model.pt`: Serialized ANFIS state dictionary with optimized membership functions

**Analytical Datasets**:
- `outputs/historical_and_predicted_2010_2030.csv`: Comprehensive temporal dataset with historical observations and future projections

**Performance Documentation**:
- `outputs/training_summary.csv`: Detailed training metrics and model performance indicators

---

## 5. Sustainable Development Goals (SDGs) Integration Framework

This Decision Support System directly contributes to achieving multiple United Nations Sustainable Development Goals through evidence-based policy recommendation:

### 5.1 SDG 13: Climate Action
**Contribution**: Advanced emission trajectory forecasting enables proactive climate policy design and carbon footprint reduction strategies.

### 5.2 SDG 1: No Poverty
**Contribution**: Integrated poverty-environment modeling ensures social equity considerations are systematically embedded within environmental policy frameworks, preventing regressive taxation impacts.

### 5.3 SDG 8: Decent Work and Economic Growth
**Contribution**: Evidence-based policy optimization supports sustainable economic development while maintaining environmental stewardship and social welfare objectives.

---

## 6. Decision Support System (DSS) Theoretical Framework Implementation

This implementation follows classical DSS architecture principles, mapping theoretical components to practical software modules:

| **DSS Theoretical Component** | **Implementation Module** | **Functional Description** |
|-------------------------------|---------------------------|----------------------------|
| **Data Management Subsystem** | `dataprep/`, `data/`, `config.py` | Raw data ingestion, preprocessing pipelines, normalization procedures, and provincial reference management |
| **Model Management Subsystem** | `models/`, `main.py` | LSTM temporal forecasting, ANFIS policy prediction, training orchestration, and performance evaluation frameworks |
| **Knowledge Management Subsystem** | `dataprep/target.py`, `ui/dashboard/views/extras.py` | Policy effectiveness scoring, scenario optimization algorithms, and sensitivity analysis capabilities |
| **User Interface Subsystem** | `ui/app.py`, `ui/dashboard/views/` | Interactive web dashboard, multi-scale visualization, decision support tools, and stakeholder communication interface |

This comprehensive mapping ensures that theoretical DSS foundations are rigorously implemented within the practical software architecture, providing robust decision support capabilities for climate-poverty policy analysis.

