# Superstore Sales Forecasting

An end-to-end Enterprise Business Intelligence (BI) and Time Series Forecasting Suite built with **Python**, **Statsmodels (SARIMAX)**, **XGBoost**, **Random Forest**, **Plotly**, and **Streamlit**.

The application bridges classical Excel pivot analytics with modern machine learning algorithms, enabling multi-frequency sales demand forecasting (Daily, Weekly, Monthly) with 95% confidence intervals.

---

## Key Architecture & Capabilities

### 1. Executive BI & Pivot Analytics Layer
- **Excel Pivot Re-creation**: Interactive cross-tabulations and pivot tables (`Category` vs `Region`) with row and column grand totals (`margins=True`).
- **Revenue Contribution**: Interactive donut and pie charts visualizing market share across Customer Segments and Product Categories.
- **Niche Drilldowns**: Sub-Category revenue bar charts colored by parent category.

### 2. Time Series Diagnostics Layer
- **Stationarity Testing**: Automated **Augmented Dickey-Fuller (ADF)** hypothesis test ($p \le 0.05$) to verify if differencing is required.
- **4-Way Seasonal Decomposition**: Decouples sales into **Observed = Trend + Seasonality + Residuals**.
- **Autocorrelation Insights**: Computes **ACF** and **PACF** to identify autoregressive ($p$) and moving average ($q$) lag orders.

### 3. Machine Learning & Statistical Arena
- **Benchmark Leaderboard**: Evaluates models side-by-side on an out-of-sample holdout test set using **MAE**, **RMSE**, and **MAPE (%)**.
- **SARIMAX $(1,1,1) \times (1,1,1)_s$**: Captures periodic seasonal cycles.
- **XGBoost & Random Forest**: Supervised regression using 20+ engineered features:
  - Multi-frequency lag variables ($t-1, t-7, t-30$, etc.)
  - Moving momentum and volatility windows (7-day / 30-day rolling mean, std, min, max)
  - Calendar signals (`day_of_week`, `day_of_year`, `month`, `is_weekend`)
- **Explainable AI**: Relative feature importance bar chart uncovering the key drivers of enterprise demand.

### 4. Future Sales Forecast Simulator
- Out-of-sample forward projections (7 to 60 periods ahead).
- 95% Confidence Interval risk boundaries for supply chain inventory buffers.

---

## Model Benchmark Leaderboard (30-Day Holdout Test Set)

| Rank | Model | Model Type | MAE ($) | RMSE ($) | MAPE (%) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 🥇 | **SARIMAX** | Statistical Time Series | **1,177.64** | 1,733.94 | **40.58%** |
| 🥈 | **Random Forest** | Bagging Ensemble | 1,332.62 | **1,700.98** | 55.82% |
| 🥉 | **XGBoost Regressor** | Gradient Boosted Trees | 1,415.55 | 1,801.63 | 62.04% |

> **Key Takeaway**: SARIMAX excelled on MAE due to consistent weekly rhythms, while Random Forest achieved the lowest RMSE, penalizing large outlier misses. XGBoost feature importance revealed that **30-day moving average (`rolling_mean_30`)** and **annual day-of-year (`day_of_year`)** account for over **26% of predictive power**.

---

## Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/sales-forecasting.git
cd sales-forecasting
