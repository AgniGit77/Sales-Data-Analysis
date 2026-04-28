"""
Sales Forecasting using Linear Regression.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import streamlit as st

from components.charts import apply_layout


def prepare_forecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily revenue for forecasting."""
    daily = df.groupby("Date")["Revenue"].sum().reset_index()
    daily = daily.sort_values("Date")
    daily["DayIndex"] = (daily["Date"] - daily["Date"].min()).dt.days
    return daily


def train_forecast_model(daily: pd.DataFrame, degree: int = 3):
    """Train polynomial regression model."""
    X = daily["DayIndex"].values.reshape(-1, 1)
    y = daily["Revenue"].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return model, poly, r2, mae


def generate_forecast(model, poly, daily: pd.DataFrame, days_ahead: int = 90):
    """Generate future predictions."""
    last_day = daily["DayIndex"].max()
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1)
    future_X = poly.transform(future_days)
    predictions = model.predict(future_X)

    # Ensure no negative predictions
    predictions = np.maximum(predictions, 0)

    last_date = daily["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Revenue": predictions,
        "DayIndex": future_days.flatten(),
    })

    return forecast_df


def forecast_chart(daily: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Create forecast visualization with confidence intervals."""
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["Revenue"],
        name="Historical Sales",
        mode="markers",
        marker=dict(size=3, color="#6366f1", opacity=0.4),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    ))

    # Rolling average of historical
    daily_sorted = daily.sort_values("Date")
    ma = daily_sorted["Revenue"].rolling(14, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=daily_sorted["Date"], y=ma,
        name="14-Day Average",
        mode="lines",
        line=dict(color="#6366f1", width=2),
    ))

    # ── Forecast ──
    fig.add_trace(go.Scatter(
        x=forecast["Date"], y=forecast["Predicted_Revenue"],
        name="Forecast",
        mode="lines",
        line=dict(color="#f59e0b", width=3, dash="dot"),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Predicted: $%{y:,.0f}<extra></extra>",
    ))

    # Confidence band (±20%)
    upper = forecast["Predicted_Revenue"] * 1.2
    lower = forecast["Predicted_Revenue"] * 0.8

    fig.add_trace(go.Scatter(
        x=forecast["Date"], y=upper,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["Date"], y=lower,
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(245,158,11,0.15)",
        name="Confidence Band (±20%)",
        hoverinfo="skip",
    ))

    # Forecast start line — add_shape avoids Plotly's internal date arithmetic error
    forecast_start = daily["Date"].max().strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=forecast_start, x1=forecast_start,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="rgba(148,163,184,0.5)", dash="dash", width=1.5),
    )
    fig.add_annotation(
        x=forecast_start, y=0.98,
        xref="x", yref="paper",
        text="Forecast Start",
        showarrow=False,
        font=dict(color="#94a3b8", size=11),
        bgcolor="rgba(15,23,42,0.6)",
        borderpad=4,
    )

    return apply_layout(fig, "90-Day Sales Forecast", height=480)


def render_forecasting(df: pd.DataFrame):
    """Render the complete forecasting section."""
    st.markdown("""
    <div class="section-header">
        <h3>🔮 Sales Forecasting</h3>
        <p class="section-subtitle">Polynomial Regression model with 90-day predictions</p>
    </div>
    """, unsafe_allow_html=True)

    daily = prepare_forecast_data(df)

    if len(daily) < 30:
        st.warning("Need at least 30 days of data for forecasting.")
        return

    model, poly, r2, mae = train_forecast_model(daily)
    forecast = generate_forecast(model, poly, daily, days_ahead=90)

    # Model metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">R² Score</p>
            <p class="metric-value" style="color: {'#10b981' if r2 > 0.7 else '#f59e0b'};">{r2:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Mean Absolute Error</p>
            <p class="metric-value">${mae:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        forecast_total = forecast["Predicted_Revenue"].sum()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">90-Day Forecast Total</p>
            <p class="metric-value" style="color: #f59e0b;">${forecast_total:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Chart
    fig = forecast_chart(daily, forecast)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly breakdown
    st.markdown("#### 📅 Monthly Forecast Breakdown")
    forecast["Month"] = forecast["Date"].dt.to_period("M")
    monthly_forecast = forecast.groupby("Month")["Predicted_Revenue"].sum().reset_index()
    monthly_forecast.columns = ["Month", "Predicted Revenue"]
    monthly_forecast["Predicted Revenue"] = monthly_forecast["Predicted Revenue"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(monthly_forecast, use_container_width=True, hide_index=True)
