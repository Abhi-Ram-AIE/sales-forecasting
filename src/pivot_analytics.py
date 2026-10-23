import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def build_pivot_table(df, index_col="Category", columns_col="Region", values_col="Sales", aggfunc="sum"):
    """
    Recreates Excel Pivot Table functionality in Pandas.
    
    Example: Category (Rows) vs Region (Columns) with Sum of Sales.
    """
    pivot = pd.pivot_table(
        df,
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc=aggfunc,
        margins=True,          # Adds Grand Total (like Excel!)
        margins_name="Total"
    )
    return pivot

def get_category_distribution(df, col="Category", values_col="Sales"):
    """
    Computes absolute sales and percentage share for Pie / Donut Charts.
    """
    dist = df.groupby(col)[values_col].sum().reset_index()
    total_sales = dist[values_col].sum()
    dist["Share (%)"] = (dist[values_col] / total_sales * 100).round(2)
    dist = dist.sort_values(values_col, ascending=False).reset_index(drop=True)
    return dist

def build_crosstab(df, row_col="Sub-Category", col_col="Segment", values_col="Sales"):
    """
    Recreates Excel Cross-Tabulation matrix (e.g. Sub-Category vs Customer Segment).
    """
    ct = pd.crosstab(
        index=df[row_col],
        columns=df[col_col],
        values=df[values_col],
        aggfunc="sum"
    ).fillna(0)
    return ct

def generate_pivot_analytics_summary(df):
    """
    Generates a full Excel BI summary report dictionary containing pivot tables and distributions.
    """
    category_region_pivot = build_pivot_table(df, index_col="Category", columns_col="Region", values_col="Sales")
    cat_dist = get_category_distribution(df, col="Category")
    subcat_segment_ct = build_crosstab(df, row_col="Sub-Category", col_col="Segment")
    
    return {
        "pivot_category_region": category_region_pivot,
        "category_distribution": cat_dist,
        "crosstab_subcat_segment": subcat_segment_ct
    }

if __name__ == "__main__":
    from data_loader import load_raw_data
    df = load_raw_data()
    summary = generate_pivot_analytics_summary(df)
    
    print("=== EXCEL PIVOT TABLE: Category vs Region (Sales Sum) ===")
    print(summary["pivot_category_region"])
    print("\n=== PIE CHART DATA: Category Distribution ===")
    print(summary["category_distribution"])
    print("\n=== CROSS TABULATION: Sub-Category vs Segment ===")
    print(summary["crosstab_subcat_segment"].head())
