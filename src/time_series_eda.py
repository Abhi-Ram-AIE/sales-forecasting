import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf

def check_stationarity(ts, maxlag=None):
    """
    Performs Augmented Dickey-Fuller (ADF) test to evaluate time series stationarity.
    
    Returns:
        dict: ADF Statistic, p-value, critical values, and stationarity conclusion.
    """
    # Remove missing values
    clean_ts = ts.dropna()
    
    result = adfuller(clean_ts, maxlag=maxlag)
    
    adf_stat = result[0]
    p_value = result[1]
    lags_used = result[2]
    n_obs = result[3]
    critical_values = result[4]
    
    is_stationary = p_value <= 0.05
    
    return {
        "adf_statistic": round(adf_stat, 4),
        "p_value": round(p_value, 4),
        "lags_used": lags_used,
        "n_obs": n_obs,
        "critical_values": {k: round(v, 4) for k, v in critical_values.items()},
        "is_stationary": is_stationary,
        "conclusion": "Series is STATIONARY (p <= 0.05)" if is_stationary else "Series is NON-STATIONARY (p > 0.05, differencing needed)"
    }

def decompose_time_series(ts, period=7, model="additive"):
    """
    Decomposes time series into Observed, Trend, Seasonal, and Residual components.
    
    Parameters:
        ts (pd.Series): Time series indexed by datetime.
        period (int): Seasonal period (7 for daily data with weekly seasonality, 52 for weekly, 12 for monthly).
        model (str): 'additive' or 'multiplicative'.
    """
    clean_ts = ts.dropna()
    decomposition = seasonal_decompose(clean_ts, period=period, model=model, extrapolate_trend="freq")
    
    return {
        "observed": decomposition.observed,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "resid": decomposition.resid
    }

def compute_acf_pacf(ts, nlags=30):
    """
    Computes Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) values.
    """
    clean_ts = ts.dropna()
    acf_vals = acf(clean_ts, nlags=nlags, fft=True)
    pacf_vals = pacf(clean_ts, nlags=nlags, method="ywm")
    
    return {
        "acf": acf_vals,
        "pacf": pacf_vals,
        "lags": list(range(nlags + 1))
    }

if __name__ == "__main__":
    from data_loader import load_raw_data, resample_sales_data
    
    df = load_raw_data()
    daily_sales = resample_sales_data(df, freq="D").set_index("Order Date")["Sales"]
    
    print("=== STATIONARITY TEST (ADF) ===")
    adf_res = check_stationarity(daily_sales)
    for k, v in adf_res.items():
        print(f"{k}: {v}")
        
    print("\n=== SEASONAL DECOMPOSITION ===")
    decomp = decompose_time_series(daily_sales, period=7)
    print("Trend head:\n", decomp["trend"].head())
    print("Seasonal head:\n", decomp["seasonal"].head())
    
    print("\n=== ACF & PACF LAG VALUES ===")
    corrs = compute_acf_pacf(daily_sales, nlags=7)
    print("ACF (Lags 0-7):", np.round(corrs["acf"], 3))
    print("PACF (Lags 0-7):", np.round(pacf_vals if 'pacf_vals' in locals() else corrs["pacf"], 3))
