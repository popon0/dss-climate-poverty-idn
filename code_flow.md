# 📘 Code Flow Documentation

This document provides a detailed explanation of the code architecture and execution flow of the **Decision Support System (DSS)** repository.  
It describes how raw datasets are transformed into predictive outputs and visualized through the Streamlit dashboard.

---

## 1. High-Level Pipeline

1. **Data Input**  
   - Raw greenhouse gas (GHG) emissions and poverty rate data (`data/raw/`).

2. **Preprocessing** (`dataprep/`)  
   - Normalization of emissions and poverty values (min–max scaling).  
   - Creation of sequential data for the LSTM model.  
   - Calculation of policy scores for ANFIS training.

3. **Model Training** (`main.py`)  
   - **LSTM**: Forecasts emission and poverty trajectories.  
   - **ANFIS (Sugeno-1)**: Predicts adaptive tax scores based on scaled features.

4. **Prediction** (`predict.py`)  
   - Forecasts for 2021–2030.  
   - Combines predicted values with historical data (2010–2020).  
   - Computes adaptive tax scenarios and revenues.

5. **Visualization** (`ui/`)  
   - Streamlit dashboard for interactive exploration of results.  
   - Includes KPIs, choropleth maps, line charts, heatmaps, and optimizers.

---

## 2. Module Breakdown

- **config.py**  
  Central configuration file. Defines dataset paths, output directories, model paths, default years, and utility loaders.

- **dataprep/**  
  - `preprocessor.py`: Implements normalization and LSTM sequence generation.  
  - `target.py`: Computes policy scores (0.6 × emission + 0.4 × poverty).  
  - `province_reference.py`: Mapping between province names and official BPS codes.

- **models/**  
  - `lstm_model.py`: Definition, training, and evaluation functions for the LSTM model.  
  - `anfis_model.py`: ANFIS (Sugeno-1) implementation with Gaussian membership functions.

- **main.py**  
  Orchestrates data preparation, training, and evaluation of LSTM and ANFIS models.  
  Saves trained models and training summary metrics.

- **predict.py**  
  Loads trained models, forecasts future values, computes adaptive tax scores,  
  and generates a combined historical + predicted dataset.

- **ui/**  
  Streamlit application for visualization.  
  - `app.py`: main entrypoint for the dashboard.  
  - `dashboard/`: modular views for national, provincial, and comparative analyses.

---

## 3. Execution Flow

```bash
# Step 1: Train models from raw data (2010–2020)
python main.py --mode from-raw

# Step 2: Forecast 2021–2030 and generate combined dataset
python predict.py --start_year 2021 --end_year 2030

# Step 3: Launch the interactive dashboard
streamlit run ui/app.py
```

---

## 4. Technical Notes

- **Scaling**  
  Min–max normalization ensures emissions and poverty rates are within [0, 1].  
  This is essential for both LSTM convergence and fuzzy membership functions in ANFIS.

- **LSTM Model**  
  - Input: sliding window of 3 years (`t-3, t-2, t-1`).  
  - Output: emission and poverty prediction at year `t`.  
  - Hidden size = 64, 2 stacked layers, dropout = 0.2.  
  - Optimizer: Adam, learning rate = 0.005, trained for 2000 epochs.

- **ANFIS Model (Sugeno-1)**  
  - Inputs: scaled emission and poverty.  
  - Membership functions: 3 Gaussian per input.  
  - Rule base: 9 fuzzy rules.  
  - Linear output layer (Sugeno-1 type).

- **Outputs**  
  - Trained models:  
    - `outputs/lstm_model.pt`  
    - `outputs/anfis_model.pt`  
  - Combined dataset:  
    - `outputs/historical_and_predicted_2010_2030.csv`  
  - Training summary:  
    - `outputs/training_summary.csv`

---

## 5. Relevance to SDGs

This system directly contributes to **Sustainable Development Goals (SDGs):**  
- **SDG 13 (Climate Action):** By forecasting emissions and supporting adaptive policy design.  
- **SDG 1 (No Poverty):** By integrating poverty reduction targets into the predictive framework.  
- **SDG 17 (Partnerships for the Goals):** By providing an open-source, data-driven decision support tool.

---

## 6. DSS Subsystem Mapping

This project aligns with the classical DSS architecture, which includes four subsystems:

| DSS Subsystem            | Folder / Files             | Description |
|---------------------------|----------------------------|-------------|
| **Data Management**       | `dataprep/`, `data/`       | Handles raw data, preprocessing, normalization, province code references |
| **Model Management**      | `models/`                  | LSTM & ANFIS predictive models, training and evaluation scripts |
| **Knowledge Management**  | `dataprep/target.py`, `ui/dss/views/extras.py` | Policy scoring, scenario optimizer, sensitivity analysis |
| **User Interface**        | `ui/app.py`, `ui/dss/views/` | Streamlit dashboard for visualization and decision support |

This mapping ensures that the theoretical foundation of DSS is reflected
in the practical implementation of the project.

