"""
Streamlit dashboard for the End-to-End Sales Forecasting & Demand Intelligence System.

This app is self-contained: it reads train.csv directly and recomputes everything it
needs at runtime (aggregation, forecasting, anomaly detection, clustering), so it can be
deployed on Streamlit Community Cloud without depending on the notebook's saved state.
Expensive computations are wrapped in st.cache_data / st.cache_resource so the app stays
responsive after the first load.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")


# ---------------------------------------------------------------------------
# Data loading and feature engineering (cached so this only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the raw Superstore CSV and add the time features used throughout the app."""
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df


@st.cache_data
def get_monthly_series(_df, column=None, value=None):
    """Aggregate sales to monthly totals, optionally filtered to one Category or Region value.
    The leading underscore on _df tells Streamlit's cache not to try hashing the whole dataframe
    every call, since it never changes within a session."""
    data = _df if column is None else _df[_df[column] == value]
    return data.set_index("Order Date").resample("MS")["Sales"].sum()


@st.cache_data
def get_weekly_series(_df):
    return _df.set_index("Order Date").resample("W")["Sales"].sum()


# ---------------------------------------------------------------------------
# Forecasting (Task 7 Page 2 needs this on demand for whatever segment is selected)
# ---------------------------------------------------------------------------
@st.cache_resource
def fit_sarima_forecast(series_key, _series, horizon):
    """Fit SARIMA on a monthly series and return a horizon-month forecast plus backtest metrics.

    series_key is a plain string used as the cache key (Streamlit can't hash a pandas Series
    directly), while _series is the actual data passed in without being hashed.

    Uses a fixed, moderate order rather than re-running auto_arima on every dashboard
    interaction, since auto_arima's search is too slow for a responsive UI. A sanity check
    falls back to a simpler non-seasonal ARIMA if the seasonal fit produces an unreasonable
    forecast, the same numerical stability issue and fix documented in Task 4 of the notebook.
    """
    series = _series.dropna()
    if len(series) < 12:
        return None, None, None

    # Backtest: hold out the last 3 months (or fewer if the series is short) to compute MAE/RMSE
    test_len = min(3, len(series) // 4)
    train, test = series.iloc[:-test_len], series.iloc[-test_len:]

    def try_fit(order, seasonal_order, train_data, steps):
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        fc = fit.get_forecast(steps=steps).predicted_mean
        return fc

    order, seasonal_order = (1, 1, 1), (1, 1, 1, 12) if len(train) >= 24 else (0, 0, 0, 0)

    try:
        backtest_fc = try_fit(order, seasonal_order, train, test_len)
        backtest_fc.index = test.index
        # Sanity check against divergence, same fallback approach used in the notebook
        if backtest_fc.max() > 5 * series.max() or backtest_fc.min() < -5 * series.max():
            raise ValueError("Seasonal fit diverged")
    except Exception:
        order, seasonal_order = (1, 1, 1), (0, 0, 0, 0)
        backtest_fc = try_fit(order, seasonal_order, train, test_len)
        backtest_fc.index = test.index

    mae = mean_absolute_error(test.values, backtest_fc.values)
    rmse = np.sqrt(mean_squared_error(test.values, backtest_fc.values))

    # Now fit on the FULL series and forecast the requested horizon into the true future
    future_fc = try_fit(order, seasonal_order, series, horizon)
    future_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    future_fc.index = future_dates

    return future_fc, mae, rmse


# ---------------------------------------------------------------------------
# Anomaly detection (Task 7 Page 3)
# ---------------------------------------------------------------------------
@st.cache_data
def detect_anomalies(_weekly_series):
    weekly_df = _weekly_series.reset_index()
    weekly_df.columns = ["Date", "Sales"]

    # Isolation Forest: global outlier detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly_df["IsoForest_Anomaly"] = iso.fit_predict(weekly_df[["Sales"]].values) == -1

    # Rolling Z-score: local outlier detection against the trailing 8-week window
    weekly_df["RollingMean"] = weekly_df["Sales"].rolling(8).mean()
    weekly_df["RollingStd"] = weekly_df["Sales"].rolling(8).std()
    weekly_df["ZScore"] = (weekly_df["Sales"] - weekly_df["RollingMean"]) / weekly_df["RollingStd"]
    weekly_df["ZScore_Anomaly"] = weekly_df["ZScore"].abs() > 2

    return weekly_df


# ---------------------------------------------------------------------------
# Clustering (Task 7 Page 4)
# ---------------------------------------------------------------------------
@st.cache_data
def cluster_subcategories(_df):
    df = _df.copy()
    df["YearMonth"] = df["Order Date"].dt.to_period("M")
    subcat_monthly = df.groupby(["Sub-Category", "YearMonth"])["Sales"].sum().reset_index()

    total_sales = df.groupby("Sub-Category")["Sales"].sum()
    yearly = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack()
    growth = yearly.pct_change(axis=1).mean(axis=1) * 100
    volatility = subcat_monthly.groupby("Sub-Category")["Sales"].std()
    avg_order = df.groupby("Sub-Category")["Sales"].mean()

    features = pd.DataFrame({
        "Total_Sales": total_sales, "Growth_Rate_%": growth,
        "Volatility": volatility, "Avg_Order_Value": avg_order
    }).reset_index()

    feature_cols = ["Total_Sales", "Growth_Rate_%", "Volatility", "Avg_Order_Value"]
    X_scaled = StandardScaler().fit_transform(features[feature_cols])

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    features["Cluster"] = km.fit_predict(X_scaled)

    centers = pd.DataFrame(km.cluster_centers_, columns=feature_cols)

    def label_cluster(row):
        if row["Growth_Rate_%"] > 1.5:
            return "Growing Demand"
        if row["Avg_Order_Value"] > 1.5 and row["Volatility"] > 1.0:
            return "High-Value, Volatile Demand"
        if row["Total_Sales"] > 0.5 and row["Volatility"] < 1.0:
            return "High Volume, Stable Demand"
        if row["Total_Sales"] < 0 and row["Growth_Rate_%"] < 0:
            return "Low Volume, Declining Demand"
        return "Low Volume, High Volatility"

    centers["Label"] = centers.apply(label_cluster, axis=1)
    features["Cluster_Label"] = features["Cluster"].map(centers["Label"].to_dict())

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    features["PC1"], features["PC2"] = coords[:, 0], coords[:, 1]

    return features


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
df = load_data()

st.sidebar.title("Sales Forecasting & Demand Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

# ---------------- PAGE 1: Sales Overview Dashboard ----------------
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    yearly_totals = df.groupby("Year")["Sales"].sum().reset_index()
    fig_year = px.bar(yearly_totals, x="Year", y="Sales", title="Total Sales by Year")
    st.plotly_chart(fig_year, width='stretch')

    monthly = get_monthly_series(df).reset_index()
    monthly.columns = ["Date", "Sales"]
    fig_trend = px.line(monthly, x="Date", y="Sales", title="Monthly Sales Trend", markers=True)
    st.plotly_chart(fig_trend, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.multiselect("Filter by Region", options=df["Region"].unique(),
                                        default=list(df["Region"].unique()))
    with col2:
        category_filter = st.multiselect("Filter by Category", options=df["Category"].unique(),
                                          default=list(df["Category"].unique()))

    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)]
    breakdown = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig_breakdown = px.bar(breakdown, x="Region", y="Sales", color="Category", barmode="group",
                            title="Sales by Region and Category (filtered)")
    st.plotly_chart(fig_breakdown, width='stretch')

# ---------------- PAGE 2: Forecast Explorer ----------------
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Forecasts use SARIMA, the best performing model identified in the notebook (Task 3).")

    col1, col2 = st.columns(2)
    with col1:
        dimension = st.selectbox("Select dimension", ["Overall", "Category", "Region"])
    with col2:
        if dimension == "Category":
            segment_value = st.selectbox("Select Category", df["Category"].unique())
        elif dimension == "Region":
            segment_value = st.selectbox("Select Region", df["Region"].unique())
        else:
            segment_value = None

    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    if dimension == "Overall":
        series = get_monthly_series(df)
        series_key = "overall"
    else:
        col_name = "Category" if dimension == "Category" else "Region"
        series = get_monthly_series(df, col_name, segment_value)
        series_key = f"{col_name}:{segment_value}"

    forecast, mae, rmse = fit_sarima_forecast(series_key, series, horizon)

    if forecast is None:
        st.warning("Not enough data to forecast this segment.")
    else:
        fig = go.Figure()
        recent = series.iloc[-12:]
        fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines+markers", name="Recent Actual"))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers",
                                  name="SARIMA Forecast", line=dict(dash="dash")))
        fig.update_layout(title=f"{horizon}-Month Forecast ({series_key})",
                           xaxis_title="Date", yaxis_title="Sales ($)")
        st.plotly_chart(fig, width='stretch')

        st.subheader("Forecast values")
        st.dataframe(forecast.rename("Forecasted Sales").round(2))

        col1, col2 = st.columns(2)
        col1.metric("Backtest MAE", f"${mae:,.2f}")
        col2.metric("Backtest RMSE", f"${rmse:,.2f}")
        st.caption("MAE and RMSE are computed by holding out the last few known months and "
                   "comparing the model's prediction against what actually happened.")

# ---------------- PAGE 3: Anomaly Report ----------------
elif page == "Anomaly Report":
    st.title("Anomaly Report")

    weekly_series = get_weekly_series(df)
    anomaly_df = detect_anomalies(weekly_series)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anomaly_df["Date"], y=anomaly_df["Sales"], mode="lines", name="Weekly Sales"))

    iso_points = anomaly_df[anomaly_df["IsoForest_Anomaly"]]
    fig.add_trace(go.Scatter(x=iso_points["Date"], y=iso_points["Sales"], mode="markers",
                              name="Isolation Forest Anomaly",
                              marker=dict(color="red", size=10, symbol="x")))

    z_points = anomaly_df[anomaly_df["ZScore_Anomaly"]]
    fig.add_trace(go.Scatter(x=z_points["Date"], y=z_points["Sales"], mode="markers",
                              name="Z-Score Anomaly",
                              marker=dict(color="orange", size=10, symbol="diamond")))

    fig.update_layout(title="Weekly Sales with Detected Anomalies", xaxis_title="Date", yaxis_title="Sales ($)")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Detected anomaly weeks")
    combined = anomaly_df[anomaly_df["IsoForest_Anomaly"] | anomaly_df["ZScore_Anomaly"]][
        ["Date", "Sales", "IsoForest_Anomaly", "ZScore_Anomaly"]
    ].sort_values("Date").reset_index(drop=True)
    st.dataframe(combined, width='stretch')

    both_count = (combined["IsoForest_Anomaly"] & combined["ZScore_Anomaly"]).sum()
    st.caption(f"{len(combined)} weeks flagged in total, {both_count} flagged by both methods "
               "(these are the highest-confidence anomalies).")

# ---------------- PAGE 4: Product Demand Segments ----------------
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")

    clusters = cluster_subcategories(df)

    fig = px.scatter(clusters, x="PC1", y="PC2", color="Cluster_Label", text="Sub-Category",
                      title="Product Sub-Category Clusters (PCA-reduced to 2D)", size_max=15)
    fig.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=1, color="black")))
    st.plotly_chart(fig, width='stretch')

    st.subheader("Sub-categories by cluster")
    for label in sorted(clusters["Cluster_Label"].unique()):
        members = clusters[clusters["Cluster_Label"] == label]
        with st.expander(f"{label} ({len(members)} sub-categories)"):
            st.dataframe(
                members[["Sub-Category", "Total_Sales", "Growth_Rate_%", "Volatility", "Avg_Order_Value"]]
                .round(2).reset_index(drop=True),
                width='stretch',
            )
