# Decision Support System for Carbon Tax Policy

This project implements a comprehensive Decision Support System (DSS) for evaluating
carbon tax policies across Indonesian provinces, utilizing hybrid machine learning models
(LSTM + ANFIS) and an interactive Streamlit dashboard for evidence-based policy analysis.

## 🚀 Key Features
- Comprehensive data preprocessing and normalization pipeline (2010–2020 historical data)
- LSTM neural networks for forecasting greenhouse gas emissions and poverty rates (2021–2030)
- ANFIS (Adaptive Neuro-Fuzzy Inference System) with Sugeno-1 inference for adaptive taxation policy scoring
- Seamless integration of historical and predicted data for comprehensive scenario analysis
- Interactive web-based visualization dashboard powered by Streamlit

## 🏛 DSS Architecture

This DSS follows the **classical four-subsystem architecture**:

![DSS Architecture](dss_architecture.png)

- **Data Management Subsystem**: Advanced preprocessing, normalization algorithms, and provincial reference management
- **Model Management Subsystem**: LSTM temporal forecasting and ANFIS neuro-fuzzy predictive models
- **Knowledge Management Subsystem**: Policy effectiveness scoring, scenario optimization, and analytical frameworks
- **User Interface Subsystem**: Interactive Streamlit-based dashboard for decision-makers and stakeholders

## 📊 Machine Learning Models
- **LSTM (Long Short-Term Memory)**: Trained on historical temporal sequences (2010–2020) using a 3-year sliding window approach for accurate time series forecasting
- **ANFIS (Adaptive Neuro-Fuzzy Inference System)**: Implements Sugeno-1 inference with Gaussian membership functions for adaptive policy effectiveness scoring

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
Train both **LSTM** and **ANFIS** models using raw or preprocessed datasets.
```bash
python main.py --mode from-raw
# or
python main.py --mode from-processed
```
- Trained models are saved as `outputs/lstm_model.pt` and `outputs/anfis_model.pt`
- Comprehensive training metrics are exported to `outputs/training_summary.csv`

---

### 2. Generate Predictions
Generate comprehensive datasets combining **historical observations (2010–2020)** with **future projections (2021–2030)**:
```bash
python predict.py --start_year 2021 --end_year 2030
```
- Combined output dataset is saved to: `outputs/historical_and_predicted_2010_2030.csv`

---

### 3. Launch Interactive Dashboard
Deploy the comprehensive DSS dashboard interface:
```bash
streamlit run ui/app.py
```

**Dashboard Capabilities**:
- 📊 **National Analytics**: Comprehensive trend analysis, geospatial heatmaps, and performance indicators
- 🌍 **Provincial Analysis**: Detailed contribution assessments, ranking systems, and localized insights
- 🔄 **Comparative Assessment**: Multi-province comparative analysis, correlation matrices, and radar chart visualizations
- 💸 **Revenue Analysis**: Advanced comparison between ANFIS-based adaptive taxation and flat-rate (30 Rp/kg) carbon tax policies  

---

## 📂 Technical Documentation

For comprehensive technical specifications, architectural details, and implementation documentation:

- [**Code Flow Documentation**](code_flow.md) — In-depth technical analysis of
  data preprocessing pipelines, LSTM & ANFIS model architectures, prediction workflows, and output
  generation methodologies
- [**Configuration Management**](config.py) — Centralized configuration system for dataset paths, model artifacts, and system constants

This documentation provides academic-level technical specifications that complement this overview with detailed implementation analysis of the Decision Support System architecture.

## 🎯 Sustainable Development Goals (SDGs) Alignment
This research contributes directly to multiple United Nations Sustainable Development Goals:
- **SDG 13 (Climate Action)**: Advanced emission trajectory forecasting and carbon tax policy optimization
- **SDG 1 (No Poverty)**: Integrated poverty-environment modeling ensuring social equity in environmental policies
- **SDG 8 (Decent Work and Economic Growth)**: Evidence-based economic policy frameworks supporting sustainable development

## 📜 Academic Citation
For academic usage, please cite this work using the standardized citation format provided in the `CITATION.cff` file.

## 📜 License
This project is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.  

---
