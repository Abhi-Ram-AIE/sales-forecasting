import os
import pandas as pd
import numpy as np

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")

def normalize_frequency(freq):
    if freq in ("M", "ME"):
        return pd.offsets.MonthEnd()
    return freq

def load_raw_data(file_path=DEFAULT_DATA_PATH):
    """Loads raw Superstore sales dataset and enforces proper data types."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}. Run scripts/fetch_data.py first.")
        
    df = pd.read_csv(file_path)
    
    # Parse dates
    date_cols = ["Order Date", "Ship Date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            
    # Sort chronologically by Order Date
    df = df.sort_values("Order Date").reset_index(drop=True)
    
    # Ensure numeric types
    numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df

def resample_sales_data(df, freq="D", group_by=None):
    """
    Resamples order-level sales transactions into continuous time series aggregated by specified frequency.
    
    Parameters:
        df (pd.DataFrame): Cleaned sales dataframe.
        freq (str): Resampling frequency ('D' for Daily, 'W' for Weekly, 'M' for Monthly).
        group_by (str or list, optional): Column(s) to group by before resampling (e.g. 'Category', 'Region').
        
    Returns:
        pd.DataFrame or pd.Series: Aggregated continuous time series.
    """
    df_copy = df.copy()
    freq = normalize_frequency(freq)
    
    if group_by:
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Group by specified column(s) and resample Order Date
        grouped = (
            df_copy.set_index("Order Date")
            .groupby(group_by + [pd.Grouper(freq=freq)])["Sales"]
            .sum()
            .reset_index()
        )
        return grouped
    else:
        # Overall aggregate time series
        ts = (
            df_copy.set_index("Order Date")["Sales"]
            .resample(freq)
            .sum()
            .reset_index()
        )
        return ts

if __name__ == "__main__":
    df = load_raw_data()
    print(f"Data Loaded Successfully: {df.shape[0]:,} records, {df.shape[1]} columns.")
    print(f"Date Range: {df['Order Date'].min().strftime('%Y-%m-%d')} to {df['Order Date'].max().strftime('%Y-%m-%d')}")
    
    daily_sales = resample_sales_data(df, freq="D")
    print(f"Daily Resampled Series Shape: {daily_sales.shape}")
    print(daily_sales.head())
