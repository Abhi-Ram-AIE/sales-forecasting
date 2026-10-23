import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else np.nan
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2)
    }

def train_ml_forecaster(features_df, target_col="Sales", model_type="xgboost", test_size=30):
    """
    Trains a machine learning regressor on lag/rolling features.
    
    Parameters:
        features_df (pd.DataFrame): Dataframe returned by create_time_series_features.
        target_col (str): Target column name.
        model_type (str): 'xgboost' or 'random_forest'.
        test_size (int): Number of trailing days to hold out for evaluation.
        
    Returns:
        dict: Trained model, test set predictions, evaluation metrics, and feature importances.
    """
    df = features_df.copy().sort_values("Order Date").reset_index(drop=True)
    
    feature_cols = [c for c in df.columns if c not in ["Order Date", target_col]]
    
    X = df[feature_cols]
    y = df[target_col]

    if test_size >= len(df)-4:
        test_size = max(1, int(len(df)*0.2))
    
    # Train / Test temporal split
    X_train = X.iloc[:-test_size]
    y_train = y.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]
    test_dates = df["Order Date"].iloc[-test_size:]
    
    if model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
        
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    preds = np.clip(preds, a_min=0, a_max=None)  # Enforce non-negative predictions
    
    metrics = evaluate_forecast(y_test, preds)
    
    # Feature Importances
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    test_results = pd.DataFrame({
        "Order Date": test_dates,
        "Actual": y_test,
        "Forecast": preds
    }).set_index("Order Date")
    
    return {
        "model": model,
        "feature_cols": feature_cols,
        "test_results": test_results,
        "metrics": metrics,
        "feature_importances": importances
    }

def recursive_future_forecast(model, last_features_row, feature_cols, steps=7):
    """
    Generates N-step future recursive forecasts for machine learning models.
    """
    future_preds = []
    current_row = last_features_row.copy()
    
    for step in range(steps):
        X_curr = current_row[feature_cols].values.reshape(1, -1)
        pred = float(model.predict(X_curr)[0])
        pred = max(0, pred)
        future_preds.append(pred)
        
        # Shift lags for next step
        if "lag_1" in feature_cols:
            current_row["lag_1"] = pred
            
    return future_preds

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from data_loader import load_raw_data, resample_sales_data
    from feature_engineering import create_time_series_features
    
    df = load_raw_data()
    daily = resample_sales_data(df, freq="D")
    features_df = create_time_series_features(daily)
    
    xgb_res = train_ml_forecaster(features_df, model_type="xgboost", test_size=30)
    
    print("=== XGBOOST MODEL EVALUATION (30-Day Test Set) ===")
    for k, v in xgb_res["metrics"].items():
        print(f"{k}: {v}")
        
    print("\nTop 5 Important Features:")
    print(xgb_res["feature_importances"].head())
    
    print("\nFirst 5 Predicted Days vs Actual:")
    print(xgb_res["test_results"].head())
