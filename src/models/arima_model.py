import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(y_true, y_pred):
    """
    Computes standard Time Series evaluation metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error (%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Avoid division by zero in MAPE
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan
        
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2)
    }

def fit_sarimax_model(train_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """
    Fits SARIMAX statistical model on historical sales time series.
    
    Parameters:
        train_ts (pd.Series): Training time series indexed by datetime.
        order (tuple): Non-seasonal ARIMA order (p, d, q).
        seasonal_order (tuple): Seasonal SARIMA order (P, D, Q, s).
        
    Returns:
        Fitted SARIMAXResults instance.
    """
    model = SARIMAX(
        train_ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)
    return fitted_model

def forecast_sarimax(fitted_model, steps=7):
    """
    Generates out-of-sample forecast for N steps into the future.
    
    Returns:
        pd.DataFrame: Contains columns ['forecast', 'ci_lower', 'ci_upper'].
    """
    forecast_obj = fitted_model.get_forecast(steps=steps)
    forecast_df = forecast_obj.predicted_mean.to_frame(name="forecast")
    conf_int = forecast_obj.conf_int(alpha=0.05)
    
    forecast_df["ci_lower"] = conf_int.iloc[:, 0].clip(lower=0)  # Sales cannot be negative
    forecast_df["ci_upper"] = conf_int.iloc[:, 1]
    
    return forecast_df

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from data_loader import load_raw_data, resample_sales_data
    
    df = load_raw_data()
    daily_sales = resample_sales_data(df, freq="D").set_index("Order Date")["Sales"]
    
    # Train / Test split (Last 30 days reserved for testing)
    train = daily_sales.iloc[:-30]
    test = daily_sales.iloc[-30:]
    
    print("Fitting SARIMAX(1, 1, 1)x(1, 1, 1, 7)...")
    model = fit_sarimax_model(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    
    forecast_df = forecast_sarimax(model, steps=30)
    forecast_df.index = test.index
    
    metrics = evaluate_forecast(test, forecast_df["forecast"])
    print("\n=== SARIMAX MODEL EVALUATION (30-Day Test Set) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    print("\nFirst 5 Predicted Days vs Actual:")
    comparison = pd.DataFrame({"Actual": test.head(), "Forecast": forecast_df["forecast"].head()})
    print(comparison)
