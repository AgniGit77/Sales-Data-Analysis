"""
Data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
import os


def load_data(file_path: str = None) -> pd.DataFrame:
    """Load sales data from CSV or Excel file."""
    if file_path is None:
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sales_data.csv")

    if not os.path.exists(file_path):
        # Generate synthetic data if none exists
        from data.generate_data import generate_sales_data
        return generate_sales_data(file_path)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return clean_data(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataframe."""
    # Convert date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove complete duplicates
    df = df.drop_duplicates()

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

    # Remove rows with invalid dates
    if "Date" in df.columns:
        df = df.dropna(subset=["Date"])

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed features for analysis."""
    if "Date" not in df.columns:
        return df

    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["Quarter"] = df["Date"].dt.quarter
    df["QuarterLabel"] = df["Date"].dt.year.astype(str) + " Q" + df["Date"].dt.quarter.astype(str)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayName"] = df["Date"].dt.strftime("%A")
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    # Profit margin
    if "Revenue" in df.columns and "Profit" in df.columns:
        df["ProfitMargin"] = np.where(
            df["Revenue"] != 0,
            (df["Profit"] / df["Revenue"] * 100).round(2),
            0
        )

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Calculate summary statistics."""
    stats = {
        "total_revenue": df["Revenue"].sum() if "Revenue" in df.columns else 0,
        "total_profit": df["Profit"].sum() if "Profit" in df.columns else 0,
        "total_orders": len(df),
        "total_customers": df["CustomerID"].nunique() if "CustomerID" in df.columns else 0,
        "avg_order_value": df["Revenue"].mean() if "Revenue" in df.columns else 0,
        "avg_profit_margin": (df["Profit"].sum() / df["Revenue"].sum() * 100)
                              if "Revenue" in df.columns and df["Revenue"].sum() != 0 else 0,
    }

    # Growth calculation (current vs previous period)
    if "Date" in df.columns and "Revenue" in df.columns:
        df_sorted = df.sort_values("Date")
        midpoint = df_sorted["Date"].iloc[len(df_sorted) // 2]
        first_half = df_sorted[df_sorted["Date"] <= midpoint]["Revenue"].sum()
        second_half = df_sorted[df_sorted["Date"] > midpoint]["Revenue"].sum()
        stats["revenue_growth"] = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
    else:
        stats["revenue_growth"] = 0

    return stats
