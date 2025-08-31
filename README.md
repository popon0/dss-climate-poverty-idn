# Decision Support System for Carbon Tax Policy

This project implements a Decision Support System (DSS) for evaluating
carbon tax policies across Indonesian provinces, using hybrid models
(LSTM + ANFIS) and an interactive Streamlit dashboard.

## 🚀 Features
- Data preprocessing and normalization (2010–2020 historical data).
- LSTM model for forecasting greenhouse gas emissions and poverty rates (2021–2030).
- ANFIS (Sugeno-1) for adaptive taxation policy scoring.
- Integration of historical + predicted data for scenario analysis.
- Interactive visualization dashboard with Streamlit.

## 🏛 DSS Architecture

This DSS follows the **classical four-subsystem architecture**:

![DSS Architecture](dss_architecture.png)

- **Data Management Subsystem**: preprocessing, normalization, province references.  
- **Model Management Subsystem**: LSTM + ANFIS predictive models.  
- **Knowledge Management Subsystem**: policy scoring, scenario optimizer, scenario analysis.  
- **User Interface Subsystem**: Streamlit-based dashboard for decision makers.

## 📊 Models
- **LSTM**: Trained on historical sequences (2010–2020), using a 3-year sliding window.  
- **ANFIS (Sugeno-1)**: Uses Gaussian membership functions to score policies adaptively.

---

## 🏗 Project Structure
```
├── config.py               # Centralized configuration (paths, constants, loaders)
├── data/                   # Raw, processed, and final datasets
│   ├── raw/                # Original historical data (2010–2020)
│   ├── processed/          # Scaled & normalized data
│   └── final/              # Combined dataset for DSS
├── dataprep/               # Data preprocessing modules
│   ├── preprocessor.py     # Normalization, sequence generation
│   ├── target.py           # Policy score calculation
│   └── province_reference.py
├── models/                 # ML models
│   ├── lstm_model.py       # LSTM forecasting
│   └── anfis_model.py      # ANFIS Sugeno-1
├── outputs/                # Saved models, metrics, and predictions
├── ui/                     # Streamlit dashboard
│   ├── app.py              # Main dashboard entrypoint
│   └── dashboard/          # Views, charts, analytics
├── main.py                 # Training pipeline (LSTM & ANFIS)
├── predict.py              # Generate predictions (2010–2030)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/popon0/dss-climate-poverty-idn.git
cd dss-climate-poverty-idn
```

### 2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train Models (Optional)
Train both **LSTM** and **ANFIS** from raw or processed data.
```bash
python main.py --mode from-raw
# or
python main.py --mode from-com
```
- Models will be saved under `outputs/lstm_model.pt` and `outputs/anfis_model.pt`.  
- Training summary → `outputs/training_summary.csv`.

---

### 2. Generate Predictions
Produce combined **historical (2010–2020)** + **future (2021–2030)** dataset:
```bash
python predict.py --start_year 2021 --end_year 2030
```
- Output CSV saved to: `outputs/historical_and_predicted_2010_2030.csv`

---

### 3. Run Streamlit Dashboard
Launch the interactive DSS dashboard:
```bash
streamlit run ui/app.py
```

UI Features:
- 📊 **National View**: trends, heatmaps, movers  
- 🌍 **Provincial View**: contribution, ranking, insights  
- 🔄 **Comparison View**: multi-province comparison, correlation, radar charts  
- 💸 **Revenue Comparison**: ANFIS vs Flat 30 Rp/kg tax  

---

## 📂 Documentation

For a detailed breakdown of the code architecture, data flow, and technical
explanations of the models:

- [Code Flow Documentation](code_flow.md) — full technical overview of
  data preparation, LSTM & ANFIS model design, prediction workflow, and output
  generation.
- `config.py` – central config for dataset paths, model paths, and constants. 

This complements the main README by providing an in-depth, academic-style
explanation of how the Decision Support System is implemented.

## 🎯 SDGs Relevance
This work contributes to:  
- **SDG 13 (Climate Action)**  
- **SDG 1 (No Poverty)**  
- **SDG 8 (Decent Work and Economic Growth)**

## 📜 Citation
If you use this project in academic work, please cite using the `CITATION.cff` file.

## 📜 License
This project is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.  

---
