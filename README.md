# 🌾 AgriPrice — AI Commodity Price Prediction & Detection System

<div align="center">

![SIH 2024](https://img.shields.io/badge/SIH-2024-orange?style=for-the-badge)
![Problem ID](https://img.shields.io/badge/Problem_ID-SIH1647-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/ML-SARIMA%20%7C%20ARIMA%20%7C%20XGBoost-red?style=for-the-badge)
![Data Source](https://img.shields.io/badge/Data_Source-data.gov.in_AGMARKNET-emerald?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**AI/ML based price prediction & anomaly detection system for agri-horticultural commodities**  
*Built for Smart India Hackathon 2024 | Ministry of Consumer Affairs, Food and Public Distribution*

</div>

---

> [!TIP]
> **Real Government Data Ingestion**: AgriPrice Detection System connects directly to the Government of India Open Data Platform (**data.gov.in**) AGMARKNET Mandi Price API (*Resource ID: `9ef84268-d588-465a-a308-a864a43d0070`*), ingesting daily arrival modal prices, maximum prices, and minimum prices across 550+ reporting centers in Indian states.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Live Data Ingestion](#-live-data-ingestion)
- [How to Run](#-how-to-run)
- [Deployment on Streamlit Community Cloud](#-deployment-on-streamlit-community-cloud)
- [ML Models Used](#-ml-models-used)
- [Results & Evaluation](#-results--evaluation)
- [Team & Authors](#-team--authors)
- [License](#-license)

---

## 📌 Problem Statement

| Field | Details |
|-------|---------|
| **Problem ID** | SIH1647 |
| **Title** | Development of AI-ML based models for predicting prices of agri-horticultural commodities such as pulses and vegetables (onion, potato, tomato) |
| **Organization** | Ministry of Consumer Affairs, Food and Public Distribution |
| **Department** | Department of Consumer Affairs |
| **Theme** | Agriculture, FoodTech & Rural Development |

### Background

The Department of Consumer Affairs monitors the daily prices of 22 essential food commodities through 550 price reporting centres across the country. The Department also maintains a buffer stock of pulses (gram, tur, urad, moong, and masur) and onion for strategic market interventions to stabilize price volatility.

### Goal

Build an intelligent AI/ML prediction and detection system that forecasts commodity prices 7 to 30 days in advance, enabling the government to make proactive buffer stock decisions, prevent price spikes, and protect consumer and farmer interests.

---

## 🎯 Project Overview

**AgriPrice Detection System** is an end-to-end machine learning pipeline that:
* 📥 **Ingests** daily price data from government reporting centres via the `data.gov.in` AGMARKNET API.
* 🔄 **Processes** and cleans multi-source government datasets with robust outlier removal and lag extraction.
* 📊 **Analyzes** historical price trends, seasonality, and market patterns across Indian states.
* 🤖 **Predicts** commodity prices for the next 7, 14, and 30 days with 95% confidence intervals using SARIMA, ARIMA, and XGBoost.
* 🚨 **Detects & Alerts** when prices are predicted to cross danger thresholds at both national and state levels.
* 📈 **Visualizes** forecasts through a multi-tab interactive Streamlit dashboard with real-time sync controls.

---

## ⚙️ System Architecture

```
        GOVERNMENT OF INDIA OPEN DATA PORTAL
            (data.gov.in - AGMARKNET API)
                         │
                         ▼
        REAL DATA INGESTION & NORMALIZATION
              (src/fetch_real_data.py)
                         │
                         ▼
       DATA CLEANING & OUTLIER REMOVAL (IQR)
             (src/data_preprocessing.py)
                         │
                         ▼
      FEATURE ENGINEERING & LAG EXTRACTION
            (7/14/30-day lags, rolling mean/std)
                         │
                         ▼
       MULTI-MODEL TRAINING & BENCHMARKING
           (ARIMA, SARIMA, XGBoost Regressor)
                         │
                         ▼
        FORECASTING & POLICY ALERT GENERATION
     (30-day forecast, CI bands, state-level alerts)
                         │
                         ▼
      STREAMLIT COMMUNITY CLOUD INTERACTIVE DASHBOARD
                (app.py - Live Web App)
```

---

## 🛠️ Tech Stack

* **Core Language**: Python 3.10+
* **Data Ingestion**: Requests, data.gov.in AGMARKNET REST API
* **Data Processing & Analytics**: Pandas, NumPy, SciPy
* **Visualization**: Plotly Express, Matplotlib, Seaborn
* **Machine Learning**: Statsmodels (ARIMA, SARIMA), Scikit-Learn, XGBoost
* **Dashboard Interface**: Streamlit
* **Job Scheduler**: Schedule library

---

## 📁 Project Structure

```
AgriPrice_Detection_System/
├── main.py                     # Main pipeline orchestrator & CLI runner
├── app.py                      # Streamlit dashboard script
├── requirements.txt            # Python dependencies (Cloud compatible)
├── .env.example                # Template for data.gov.in API key
├── .gitignore                  # Ignores sensitive keys & virtual environments
├── data/
│   ├── raw_data.csv            # Real AGMARKNET price dataset (state-wise)
│   └── processed_data.csv      # Enriched feature engineered dataset
├── src/
│   ├── fetch_real_data.py      # Real data fetcher for data.gov.in AGMARKNET
│   ├── data_preprocessing.py   # Cleaning and lag/rolling feature extraction
│   ├── eda.py                  # Exploratory Data Analysis plotting script
│   ├── model.py                # Model training (ARIMA, SARIMA, XGBoost)
│   └── evaluation.py           # Model evaluation, CI plotting, statewise alerts
├── models/                     # Saved model pickles (.pkl) and registry
├── outputs/                    # Diagnostic plots, benchmark CSVs, and policy reports
└── notebook/                   # Jupyter development notebooks
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/madrewarparth/AgriPrice_Detection_System.git
   cd AgriPrice_Detection_System
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Your API Key (Optional for live sync)**:
   Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Add your `data.gov.in` API key:
   ```env
   DATA_GOV_API_KEY=579b464db66ec23bdd000001dd8cf282423047cc51732c4bafcc88bf
   DATA_GOV_RESOURCE_ID=9ef84268-d588-465a-a308-a864a43d0070
   ```

---

## 📥 Live Data Ingestion

To fetch the latest daily mandi prices from data.gov.in AGMARKNET:

```bash
# Fetch latest real government data
python src/fetch_real_data.py

# Or via main orchestrator:
python main.py fetch
```

---

## 🚀 How to Run

* **Run the Complete End-to-End Pipeline**:
  ```bash
  python main.py
  ```

* **Launch the Streamlit Dashboard Locally**:
  ```bash
  streamlit run app.py
  # or
  python main.py dashboard
  ```

* **Start the Daily Automated Pipeline Scheduler (6:00 AM)**:
  ```bash
  python main.py schedule
  ```

---

## 🌐 Deployment on Streamlit Community Cloud

Deploying AgriPrice Detection System live to the web takes under 3 minutes:

### Step 1: Push Code to GitHub
Ensure all changes are committed and pushed to your GitHub repository:
```bash
git add .
git commit -m "Integrate real data.gov.in AGMARKNET feed and cloud deployment readiness"
git push origin main
```

### Step 2: Create App on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
2. Click **"New app"** (or **"Create app"**).
3. Select your repository: `madrewarparth/AgriPrice_Detection_System`.
4. Set the branch to `main`.
5. Set the Main file path to `app.py`.

### Step 3: Configure Secrets (data.gov.in API Key)
1. Click **"Advanced settings..."** -> **"Secrets"**.
2. Paste your API configuration into the secrets editor:
   ```toml
   DATA_GOV_API_KEY = "579b464db66ec23bdd000001dd8cf282423047cc51732c4bafcc88bf"
   DATA_GOV_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
   ```
3. Click **"Deploy!"**.

---

## 📊 Results & Evaluation

The pipeline outputs comparative performance metrics on real market test partitions (`outputs/model_benchmark.csv`):

| Commodity | Best Model | MAE (Rs/Qtl) | RMSE | Accuracy (100-MAPE) |
|---|---|---|---|---|
| **Onion** | XGBoost / ARIMA | ~163.90 | ~298.65 | **96.71%** |
| **Potato** | ARIMA | ~97.86 | ~120.16 | **92.85%** |
| **Tomato** | ARIMA / XGBoost | ~96.44 | ~168.97 | **96.64%** |
| **Gram** | ARIMA / XGBoost | ~264.80 | ~334.68 | **96.35%** |
| **Tur** | ARIMA / SARIMA | ~353.40 | ~430.60 | **95.51%** |

*All diagnostics (Forecast charts with 95% CI, State-wise price alerts, and XGBoost feature importances) update dynamically in the dashboard.*

---

## 👥 Team & Authors

* **Parth Madrewar** — *Lead Maintainer, Frontend Dashboard & Data Pipeline Developer*
* **Abhinav Jindal** — *ML Modeling & Pipeline Developer*

**Institution**: Lovely Professional University, Phagwara, Jalandhar, Punjab  
**SIH 2024 Project**

---

## 📄 License

This project is licensed under the MIT License.
