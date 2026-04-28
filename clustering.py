"""
Customer Segmentation using K-Means Clustering.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

from components.charts import apply_layout, CHART_COLORS


def prepare_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customer-level features for clustering."""
    customer = df.groupby("CustomerID").agg(
        TotalSpend=("Revenue", "sum"),
        OrderCount=("OrderID", "count"),
        AvgOrderValue=("Revenue", "mean"),
        TotalProfit=("Profit", "sum"),
        UniqueProducts=("Product", "nunique"),
        AvgDiscount=("Discount", "mean"),
    ).reset_index()

    # Recency: days since last purchase
    if "Date" in df.columns:
        last_purchase = df.groupby("CustomerID")["Date"].max().reset_index()
        last_purchase.columns = ["CustomerID", "LastPurchase"]
        max_date = df["Date"].max()
        last_purchase["Recency"] = (max_date - last_purchase["LastPurchase"]).dt.days
        customer = customer.merge(last_purchase[["CustomerID", "Recency"]], on="CustomerID")

    return customer


def run_kmeans(features: pd.DataFrame, n_clusters: int = 4):
    """Run K-Means clustering."""
    feature_cols = [c for c in features.columns if c != "CustomerID"]
    X = features[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    features = features.copy()
    features["Cluster"] = labels

    return features, kmeans, scaler


def elbow_chart(features: pd.DataFrame, max_k: int = 8) -> go.Figure:
    """Generate elbow method chart for optimal K."""
    feature_cols = [c for c in features.columns if c != "CustomerID"]
    X = features[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig = go.Figure(go.Scatter(
        x=list(K_range), y=inertias,
        mode="lines+markers",
        line=dict(color="#6366f1", width=3),
        marker=dict(size=10, color="#6366f1"),
        hovertemplate="K=%{x}<br>Inertia: %{y:,.0f}<extra></extra>",
    ))

    return apply_layout(fig, "🔍 Elbow Method — Optimal Clusters", height=350)


def cluster_scatter(clustered: pd.DataFrame) -> go.Figure:
    """Scatter plot of customer segments."""
    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"]

    fig = go.Figure()

    for i in sorted(clustered["Cluster"].unique()):
        subset = clustered[clustered["Cluster"] == i]
        fig.add_trace(go.Scatter(
            x=subset["TotalSpend"],
            y=subset["OrderCount"],
            mode="markers",
            name=f"Segment {i + 1}",
            marker=dict(
                size=8,
                color=colors[i % len(colors)],
                opacity=0.7,
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            hovertemplate=(
                f"<b>Segment {i + 1}</b><br>"
                "Total Spend: $%{x:,.0f}<br>"
                "Orders: %{y}<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis_title="Total Spend ($)",
        yaxis_title="Number of Orders",
    )

    return apply_layout(fig, "👥 Customer Segments", height=480)


def get_segment_profiles(clustered: pd.DataFrame) -> pd.DataFrame:
    """Generate segment profile summary."""
    profiles = clustered.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        Avg_Spend=("TotalSpend", "mean"),
        Avg_Orders=("OrderCount", "mean"),
        Avg_Order_Value=("AvgOrderValue", "mean"),
        Avg_Profit=("TotalProfit", "mean"),
    ).reset_index()

    # Name the segments
    segment_names = []
    for _, row in profiles.iterrows():
        if row["Avg_Spend"] > profiles["Avg_Spend"].quantile(0.75):
            name = "🏆 High-Value VIP"
        elif row["Avg_Orders"] > profiles["Avg_Orders"].quantile(0.75):
            name = "🔄 Frequent Buyers"
        elif row["Avg_Spend"] < profiles["Avg_Spend"].quantile(0.25):
            name = "💤 Low-Activity"
        else:
            name = "⚡ Regular Customers"
        segment_names.append(name)

    profiles["Segment_Name"] = segment_names
    profiles["Cluster"] = profiles["Cluster"] + 1

    return profiles


def render_clustering(df: pd.DataFrame):
    """Render the complete clustering section."""
    st.markdown("""
    <div class="section-header">
        <h3>👥 Customer Segmentation</h3>
        <p class="section-subtitle">K-Means clustering to identify customer groups</p>
    </div>
    """, unsafe_allow_html=True)

    features = prepare_customer_features(df)

    if len(features) < 10:
        st.warning("Need at least 10 customers for segmentation.")
        return

    n_clusters = st.slider("Number of Clusters", 2, 6, 4, key="n_clusters")

    clustered, kmeans, scaler = run_kmeans(features, n_clusters)

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_elbow = elbow_chart(features)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        fig_scatter = cluster_scatter(clustered)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Segment profiles
    st.markdown("#### 📋 Segment Profiles")
    profiles = get_segment_profiles(clustered)

    cols = st.columns(len(profiles))
    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"]

    for i, (_, profile) in enumerate(profiles.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="segment-card" style="border-top: 3px solid {colors[i % len(colors)]};">
                <h4>{profile['Segment_Name']}</h4>
                <p class="segment-stat">👥 {int(profile['Customers'])} customers</p>
                <p class="segment-stat">💰 Avg Spend: ${profile['Avg_Spend']:,.0f}</p>
                <p class="segment-stat">📦 Avg Orders: {profile['Avg_Orders']:.1f}</p>
                <p class="segment-stat">💵 Avg AOV: ${profile['Avg_Order_Value']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
