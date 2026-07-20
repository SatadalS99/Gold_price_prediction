


# 🏆 Gold Price Prediction & Interactive Forecasting Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Docker](https://img.shields.io/badge/Dev_Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)

An end-to-end machine learning pipeline and interactive web application designed to forecast historical **SPDR Gold Trust (GLD)** ETF prices. By analyzing key macroeconomic indicators—such as broader stock market indices, crude oil prices, silver trends, and currency exchange rates—this project models complex market dynamics using tree-based machine learning algorithms.

---

## 📌 Project Overview

Precious metal valuation is heavily influenced by macroeconomic interplay and cross-market correlations. This repository provides:
1. **In-Depth Exploratory Data Analysis & Modeling:** A comprehensive Jupyter Notebook exploring feature relationships, data preprocessing, and model training using **Decision Tree** and **Random Forest Regressors**.
2. **Real-Time Interactive Inference:** A lightweight, sleek **Streamlit web dashboard** that allows users to adjust macroeconomic feature sliders in real time to visualize predicted gold prices against historical data.
3. **Reproducible Development Environment:** A pre-configured VS Code Dev Container setup for containerized, dependency-free collaboration.

---

```text

## 🗂️ Repository Architecture


├── .devcontainer/
│   └── devcontainer.json                         # Containerized environment configuration
├── app_folder/
│   ├── __pycache__/                              # Compiled Python bytecode
│   ├── gld_price_app.py                          # Streamlit web application script
│   ├── gld_price_data.csv                        # Bundled dataset for standalone web deployment
│   ├── requirements.txt                          # Project dependency lockfile
│   └── streamlite.py.code-workspace              # VS Code workspace settings
├── Gold_price_prediction_using_decison_trees.ipynb  # Core ML training & evaluation notebook
├── Gold_price_prediction_Documentation.pdf       # Technical project documentation & methodology
├── gld_price_data.csv                            # Root financial dataset
└── README.md                                     # Project documentation

```

---

## 📊 Dataset & Macroeconomic Features

The dataset (`gld_price_data.csv`) tracks daily historical market indicators over a 10-year window. Each feature acts as a critical economic proxy:

| Feature Variable | Data Type | Market / Economic Significance |
| --- | --- | --- |
| **`Date`** | *Time-Series* | Daily historical timestamp (`MM/DD/YYYY`). |
| **`SPX`** | *Numerical* | **S&P 500 Index** — Reflects broader U.S. equities and equity market sentiment. |
| **`GLD`** | *Numerical* | **SPDR Gold Shares ETF** — The continuous target variable ($y$) representing gold valuation. |
| **`USO`** | *Numerical* | **United States Oil Fund** — Tracks crude oil price movements and inflation expectations. |
| **`SLV`** | *Numerical* | **iShares Silver Trust** — Reflects industrial demand and precious metal correlation. |
| **`EUR/USD`** | *Numerical* | **Euro / U.S. Dollar Rate** — Captures currency fluctuations and dollar purchasing power. |

> **💡 Key Insight:** In tree-based regression modeling, **Silver (`SLV`)** and the **`EUR/USD` exchange rate** consistently exhibit the highest feature importance due to strong inter-market colinearity and currency hedging behaviors.

---

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Machine Learning Engine:** `scikit-learn` (Decision Tree Regressor, Random Forest Regressor)
* **Data Visualization:** `matplotlib`, `seaborn`
* **Web Dashboard Application:** `streamlit`
* **Development Environment:** VS Code, Docker Dev Containers

---

## 🚀 Getting Started

Follow these instructions to set up the project locally on your machine.

### 1. Prerequisites

Ensure you have **Python 3.10 or higher** and `pip` installed. If you prefer containerized development, ensure you have **Docker** and **VS Code** with the *Dev Containers* extension installed.

### 2. Clone the Repository

```bash
git clone [https://github.com/SatadalS99/Gold_price_prediction.git](https://github.com/SatadalS99/Gold_price_prediction.git)
cd Gold_price_prediction

```

### 3. Install Dependencies

Create a clean virtual environment (optional but recommended) and install the required libraries:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r app_folder/requirements.txt

```

---

## 💻 Running the Project

### Option A: Explore the Model Training Notebook

To dive into the mathematics, exploratory visualizations, and decision tree splitting criteria, launch Jupyter:

```bash
jupyter notebook Gold_price_prediction_using_decison_trees.ipynb

```

### Option B: Launch the Interactive Streamlit App

To run the live web dashboard and perform real-time gold price predictions:

```bash
cd app_folder
streamlit run gld_price_app.py

```

*The application will automatically launch in your default web browser at `http://localhost:8501`.*

---

## 📈 Model Evaluation & Methodology

The regression models are trained using an **80/20 train-test split**. Performance is rigorously evaluated against unseen test data using industry-standard continuous error metrics:

* **Coefficient of Determination ($R^2$ Score):** Measures the proportion of variance in gold prices predictable from the macroeconomic indicators.
* **Root Mean Squared Error ($\text{RMSE}$):**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$



Provides error magnitude in the exact dollar units of the target variable.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

MIT
