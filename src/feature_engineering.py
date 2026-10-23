import numpy as np
import pandas as pd

def create_time_series_features(df_resampled, target_col="Sales", freq="D", lags=None, rolling_windows=None):
    """
    Transforms resampled sales time series into supervised ML dataset with lag, rolling, and calendar features.
    
    Parameters:
        df_resampled (pd.DataFrame): Dataframe with 'Order Date' and target column.
        target_col (str): Column name of sales values.
        lags (list): List of lag periods to create (e.g. 1, 7, 14 days ago).
        rolling_windows (list): Window sizes for rolling mean/std.
        
    Returns:
        pd.DataFrame: Feature matrix with targets and dropped NaNs.
    """
    df = df_resampled.copy()
    
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"])
        df = df.sort_values("Order Date").reset_index(drop=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    n= len(df)

    if lags is None:
        if freq in ("M", "ME"):
            lags = [1,2,3,6,12] if n >= 24 else [1,2,3]
        elif freq == "W":
            lags = [1,2,4,8,12] if n >= 52 else [1,2,4]
        else:
            lags = [1,2,3,7,14,30] if n >= 60 else [1,2,3,7]

    if rolling_windows is None:
        if freq in ("M", "ME"):
            rolling_windows = [3,6] if n >= 24 else [3]
        elif freq == "W":
            rolling_windows = [4,12] if n >= 52 else [4]
        else:
            rolling_windows = [7,30] if n >= 60 else [7]

        
    # 1. Datetime / Calendar Features
    df["day_of_week"] = df["Order Date"].dt.dayofweek
    df["day_of_month"] = df["Order Date"].dt.day
    df["month"] = df["Order Date"].dt.month
    df["quarter"] = df["Order Date"].dt.quarter
    df["day_of_year"] = df["Order Date"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # 2. Lag Features
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
        
    # 3. Rolling Window Statistics (computed on lag_1 to prevent data leakage!)
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = df["lag_1"].rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df["lag_1"].rolling(window=window).std()
        df[f"rolling_min_{window}"] = df["lag_1"].rolling(window=window).min()
        df[f"rolling_max_{window}"] = df["lag_1"].rolling(window=window).max()

    if freq != "D":
        df = df.drop(columns=["day_of_week", "is_weekend"])
        
    # Drop rows containing NaNs introduced by lagging/rolling
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from data_loader import load_raw_data, resample_sales_data
    
    raw = load_raw_data()
    daily = resample_sales_data(raw, freq="D")
    
    features_df = create_time_series_features(daily)
    print(f"Original shape: {daily.shape} -> Feature-Engineered shape: {features_df.shape}")
    print("\nFeature Columns Generated:")
    print(features_df.columns.tolist())
    print("\nSample Rows:")
    print(features_df[["Order Date", "Sales", "lag_1", "lag_7", "rolling_mean_7"]].head())
