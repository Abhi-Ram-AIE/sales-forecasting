import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_raw_data, resample_sales_data, normalize_frequency
from src.pivot_analytics import build_pivot_table, get_category_distribution
from src.time_series_eda import check_stationarity, decompose_time_series
from src.feature_engineering import create_time_series_features
from src.models.arima_model import fit_sarimax_model, forecast_sarimax, evaluate_forecast
from src.models.ml_model import train_ml_forecaster

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Superstore Sales BI & Forecasting Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING (Modern Theme) ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #42A5F5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING (CACHED) ---
@st.cache_data
def get_dataset():
    df = load_raw_data()
    return df

df_raw = get_dataset()

# --- SIDEBAR FILTERS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
st.sidebar.title("Filter")
st.sidebar.markdown("Filter Superstore Transactions:")

categories = ["All"] + list(df_raw["Category"].unique())
selected_cat = st.sidebar.selectbox("Product Category", categories)

regions = ["All"] + list(df_raw["Region"].unique())
selected_region = st.sidebar.selectbox("Region", regions)

segments = ["All"] + list(df_raw["Segment"].unique())
selected_segment = st.sidebar.selectbox("Customer Segment", segments)

# Filter Dataframe
filtered_df = df_raw.copy()
if selected_cat != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_cat]
if selected_region != "All":
    filtered_df = filtered_df[filtered_df["Region"] == selected_region]
if selected_segment != "All":
    filtered_df = filtered_df[filtered_df["Segment"] == selected_segment]

if filtered_df.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

# Resampling frequency
freq_option = st.sidebar.radio("Time Aggregation Frequency", ["Daily", "Weekly", "Monthly"], index=0)
freq_code = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}[freq_option]
freq_obj = normalize_frequency(freq_code)
SEASONAL_S = {"D": 7, "W": 4, "ME": 12}[freq_code]

def get_seasonal_order(n_obs, s):
    """Only use seasonality if there are at least 3 full cycles of data."""
    return (1, 1, 1, s) if n_obs >= 3 * s else (0, 0, 0, 0)

# --- MAIN HEADER ---
st.markdown('<div class="main-header">Superstore Sales BI & Forecasting Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Pivot Analytics • Time Series Diagnostics • Multi-Model Machine Learning Predictions</div>', unsafe_allow_html=True)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
total_sales = filtered_df["Sales"].sum()
total_orders = filtered_df["Order ID"].nunique()
avg_order_val = total_sales / total_orders if total_orders > 0 else 0
total_profit = filtered_df["Profit"].sum()

with col1:
    st.metric("Total Revenue", f"${total_sales:,.2f}")
with col2:
    st.metric("Total Orders", f"{total_orders:,}")
with col3:
    st.metric("Avg Order Value", f"${avg_order_val:,.2f}")
with col4:
    st.metric("Total Profit", f"${total_profit:,.2f}")

st.markdown("---")

# --- TABS NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Excel BI & Pivot Analytics",
    "Time Series Diagnostics",
    "Predictive Model Arena",
    "Future Sales Forecast"
])

# ==========================================
# TAB 1: EXCEL BI & PIVOT ANALYTICS
# ==========================================
with tab1:
    st.subheader("Excel-Style Pivot Tables & Distribution Visualizations")
    
    t1_col1, t1_col2 = st.columns([1, 1])
    
    with t1_col1:
        st.markdown("Sales Share by Category")
        cat_dist = get_category_distribution(filtered_df, col="Category")
        fig_pie = px.pie(
            cat_dist,
            names="Category",
            values="Sales",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Revenue Share by Product Category"
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, width="stretch")

    with t1_col2:
        st.markdown("Sales Share by Segment")
        seg_dist = get_category_distribution(filtered_df, col="Segment")
        fig_seg = px.pie(
            seg_dist,
            names="Segment",
            values="Sales",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
            title="Revenue Share by Customer Segment"
        )
        fig_seg.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_seg, width="stretch")

    st.markdown("Excel Pivot Matrix: Category vs Region Sales")
    pivot_df = build_pivot_table(filtered_df, index_col="Category", columns_col="Region", values_col="Sales")
    st.dataframe(pivot_df.style.format("${:,.2f}"), width="stretch")
    
    st.markdown("Pivot Graph: Sub-Category Revenue Breakdown")
    subcat_sales = filtered_df.groupby(["Sub-Category", "Category"])["Sales"].sum().reset_index()
    fig_bar = px.bar(
        subcat_sales,
        x="Sub-Category",
        y="Sales",
        color="Category",
        title="Sales by Sub-Category",
        text_auto="$.2s"
    )
    st.plotly_chart(fig_bar, width="stretch")

# ==========================================
# TAB 2: TIME SERIES DIAGNOSTICS
# ==========================================
with tab2:
    st.subheader("Time Series Decomposition & Stationarity Testing")
    
    ts_data = resample_sales_data(filtered_df, freq=freq_code).set_index("Order Date")["Sales"].asfreq(freq_obj)
    
    # 1. Line chart of time series
    fig_ts = px.line(
        ts_data.reset_index(),
        x="Order Date",
        y="Sales",
        title=f"Historical Aggregated Sales ({freq_option})",
        line_shape="linear"
    )
    st.plotly_chart(fig_ts, width="stretch")
    
    # 2. Stationarity Card
    adf_res = check_stationarity(ts_data)
    st.markdown("Augmented Dickey-Fuller (ADF) Stationarity Test")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("ADF Statistic", str(adf_res["adf_statistic"]))
    col_b.metric("p-Value", str(adf_res["p_value"]))
    col_c.metric("Verdict", "Stationary ✅" if adf_res["is_stationary"] else "Non-Stationary ⚠️")
    
    st.info(f"**Interpretation**: {adf_res['conclusion']}. Critical Values: 1%: {adf_res['critical_values']['1%']}, 5%: {adf_res['critical_values']['5%']}")
    
    # 3. Decomposition
    st.markdown("Seasonal Decomposition (Trend, Seasonality, Residuals)")
    period_val = 7 if freq_code == "D" else (52 if freq_code == "W" else 12)
    decomp = decompose_time_series(ts_data, period=min(period_val, len(ts_data)//2))
    
    decomp_df = pd.DataFrame({
        "Observed": decomp["observed"],
        "Trend": decomp["trend"],
        "Seasonal": decomp["seasonal"],
        "Residual": decomp["resid"]
    }).rename_axis("Order Date").reset_index()
    
    fig_decomp = px.line(
        decomp_df,
        x="Order Date",
        y=["Observed", "Trend", "Seasonal", "Residual"],
        title="Time Series Decomposition Components",
        facet_col="variable",
        facet_col_wrap=1,
        height=700
    )
    fig_decomp.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig_decomp, width="stretch")

# ==========================================
# TAB 3: PREDICTIVE MODEL ARENA
# ==========================================
with tab3:
    if freq_code == "D":
        desired_test_size = 30
    elif freq_code == "W":
        desired_test_size = 12
    else:
        desired_test_size = 6

    daily_resampled = resample_sales_data(filtered_df, freq=freq_code)
    features_df = create_time_series_features(daily_resampled, freq=freq_code)

    if len(features_df) < 8:
        st.warning(f"Insufficient records ({len(features_df)} rows) under current filter/frequency to benchmark ML models. Please select a broader filter or Daily frequency.")
    else:
        test_size = min(desired_test_size, max(2, int(int(len(features_df)*0.2))))

        st.subheader("Model Leaderboard & Out-Of-Sample Benchmark")
        st.markdown(f"We hold out the last **{test_size} periods** as a test set to benchmark statistical vs. machine learning models.")

        with st.spinner("Training SARIMAX, XGBoost and RandomForest models..."):
            # 1. SARIMAX
            train_ts = ts_data.iloc[:-test_size]
            test_ts = ts_data.iloc[-test_size:]
            
            sarimax_fit = fit_sarimax_model(train_ts, order=(1, 1, 1), seasonal_order=get_seasonal_order(len(train_ts), SEASONAL_S))
            sarimax_fc = forecast_sarimax(sarimax_fit, steps=test_size)
            sarimax_fc.index = test_ts.index
            metrics_sarimax = evaluate_forecast(test_ts, sarimax_fc["forecast"])
            
            # 2. XGBoost
            xgb_res = train_ml_forecaster(features_df, model_type="xgboost", test_size=test_size)
            metrics_xgb = xgb_res["metrics"]
            
            # 3. Random Forest
            rf_res = train_ml_forecaster(features_df, model_type="random_forest", test_size=test_size)
            metrics_rf = rf_res["metrics"]

        # Leaderboard Table
        leaderboard = pd.DataFrame([
            {"Model": "SARIMAX (Statistical)", **metrics_sarimax},
            {"Model": "XGBoost Regressor (ML)", **metrics_xgb},
            {"Model": "Random Forest (ML)", **metrics_rf}
        ]).sort_values("MAE").reset_index(drop=True)
        
        st.markdown("Model Performance Comparison")
        st.dataframe(leaderboard, width="stretch")
        
        # Plot Comparison
        comp_df = pd.DataFrame({
            "Actual Sales": test_ts,
            "SARIMAX": sarimax_fc["forecast"],
            "XGBoost": xgb_res["test_results"]["Forecast"],
            "Random Forest": rf_res["test_results"]["Forecast"]
        }).reset_index()
        
        fig_comp = px.line(
            comp_df,
            x="Order Date",
            y=["Actual Sales", "SARIMAX", "XGBoost", "Random Forest"],
            title="{test_size}-Period Holdout Test Set: Actual vs Predicted Sales",
            markers=True
        )
        st.plotly_chart(fig_comp, width="stretch")
        
        st.markdown("XGBoost Feature Importance Breakdown")
        imp_df = xgb_res["feature_importances"].head(10).rename_axis("Feature").reset_index(name="Importance")
        fig_imp = px.bar(
            imp_df,
            x="Feature",
            y="Importance",
            labels={"index": "Feature", "0": "Relative Importance"},
            title="Top 10 Predictors driving Sales in XGBoost",
            color="Importance",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_imp, width="stretch")

# ==========================================
# TAB 4: FUTURE SALES FORECAST
# ==========================================
with tab4:
    st.subheader("Interactive Future Sales Forecast Simulator")

    if freq_code == "D":
        min_h, max_h, def_h, step_h = 7,60,14,7
    elif freq_code == "W":
        min_h, max_h, def_h, step_h = 4, 26, 8, 2
    else:
        min_h, max_h, def_h, step_h = 2, 12, 6, 1
    
    forecast_horizon = st.slider("Select Forecast Horizon (Periods into Future)", min_value=min_h, max_value=max_h, value=def_h, step=step_h)
    
    if st.button("Generate Future Forecast"):
        with st.spinner(f"Generating {forecast_horizon}-period future forecast..."):
            full_sarimax = fit_sarimax_model(
                ts_data, order=(1, 1, 1),
                seasonal_order=get_seasonal_order(len(ts_data), SEASONAL_S)
            )
            future_df = forecast_sarimax(full_sarimax, steps=forecast_horizon)

            # last_date is always on the frequency boundary, so this works for D, W and month-end
            last_date = ts_data.index[-1]
            future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=freq_obj)[1:]
            future_df.index = future_dates
            future_df.index.name = "Order Date"

            st.success(f"Generated forecast for {forecast_horizon} periods "
                    f"({future_dates[0].strftime('%Y-%m-%d')} to {future_dates[-1].strftime('%Y-%m-%d')})!")

            hist_df = ts_data.tail(60).reset_index()
            fut_df = future_df.reset_index()
            
            fig_fut = go.Figure()
            
            # Historical line
            fig_fut.add_trace(go.Scatter(
                x=hist_df["Order Date"],
                y=hist_df["Sales"],
                mode="lines+markers",
                name="Historical Sales",
                line=dict(color="#1E88E5", width=2)
            ))
            
            # Forecast line
            fig_fut.add_trace(go.Scatter(
                x=fut_df["Order Date"],
                y=fut_df["forecast"],
                mode="lines+markers",
                name="Future Forecast",
                line=dict(color="#FF7043", width=3, dash="dash")
            ))
            
            # Confidence Interval Band
            fig_fut.add_trace(go.Scatter(
                x=list(fut_df["Order Date"]) + list(fut_df["Order Date"])[::-1],
                y=list(fut_df["ci_upper"]) + list(fut_df["ci_lower"])[::-1],
                fill="toself",
                fillcolor="rgba(255, 112, 67, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval"
            ))
            
            fig_fut.update_layout(
                title=f"Sales Forecast for Next {forecast_horizon} Periods with 95% Confidence Intervals",
                xaxis_title="Date",
                yaxis_title="Sales ($)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_fut, width="stretch")
            
            st.markdown("Forecast Data Table")
            st.dataframe(future_df.style.format("${:,.2f}"), width="stretch")
