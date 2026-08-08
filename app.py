import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    preprocess_data,
    find_best_k,
    perform_clustering,
    predict_customer,
    cluster_summary,
    get_clustering_features
)

from visualization import (
    scatter_plot,
    scatter_3d,
    pie_chart,
    bar_chart,
    histogram,
    box_plot,
    correlation_heatmap,
    cluster_center_chart,
    radar_chart,
    top_customers,
    parallel_coordinates,
    cluster_summary_table
)

from recommendation import (
    business_recommendation,
    customer_search
)

from sample_data import load_sample_data

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith("csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def to_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()


DEFAULT_FEATURES = ["Annual_Spending", "Order_Count"]

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("⚙ Dashboard")

data_source = st.sidebar.radio(
    "Data Source",
    ["Sample Flipkart Dataset", "Upload Dataset"]
)

uploaded_file = None

if data_source == "Upload Dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["xlsx", "csv"]
    )
    if uploaded_file is not None:
        data_source = "uploaded"
else:
    data_source = "sample"

show_dataset = st.sidebar.checkbox("Show Dataset", True)

show_heatmap = st.sidebar.checkbox("Correlation Heatmap", True)

show_box = st.sidebar.checkbox("Box Plot", True)

show_hist = st.sidebar.checkbox("Histogram", True)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Customer Segmentation

Machine Learning

K-Means Clustering

Developed using Streamlit
"""
)

# -------------------------------------------------------
# Main Title
# -------------------------------------------------------

st.title("🛒 Customer Purchase Behaviour Segmentation")

st.markdown(
"""
Cluster customers into:

- 🟢 High Value
- 🟡 Medium Value
- 🔴 Low Value

using K-Means Clustering.
"""
)

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------

if data_source == "uploaded" and uploaded_file is not None:

    df = read_uploaded_file(uploaded_file)

elif data_source == "sample":

    df = load_sample_data()

else:

    st.info("Upload your customer dataset to begin.")

    st.stop()

# -------------------------------------------------------
# Dataset Summary
# -------------------------------------------------------

st.subheader("Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers", len(df))

col2.metric("Features", len(df.columns))

col3.metric("Missing Values", df.isnull().sum().sum())

col4.metric("Duplicate Rows", df.duplicated().sum())

if show_dataset:

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

st.markdown("---")

# -------------------------------------------------------
# Feature Selection
# -------------------------------------------------------

st.subheader("Select Features")

all_numeric = df.select_dtypes(include="number").columns.tolist()

numeric_columns = get_clustering_features(df)

excluded = [c for c in all_numeric if c not in numeric_columns]

if excluded:
    st.caption(
        f"Ignored non-clustering columns: {', '.join(excluded)} "
        "(constant or id-like)."
    )

if not numeric_columns:
    st.error("No numeric features available for clustering.")
    st.stop()

defaults = [c for c in DEFAULT_FEATURES if c in numeric_columns]

if len(defaults) < 2:
    defaults = numeric_columns[:min(6, len(numeric_columns))]

features = st.multiselect(
    "Choose Numerical Features",
    numeric_columns,
    default=defaults
)

st.caption(
    "All features are standardized before clustering, so every "
    "feature contributes equally - datasets with many features "
    "are segmented correctly regardless of scale."
)

if len(features) < 2:

    st.warning("Select at least two features.")

    st.stop()

# -------------------------------------------------------
# Find Best K (Elbow Method)
# -------------------------------------------------------

st.subheader("Find Best Number of Clusters")

if st.button("Run Elbow Method"):

    X, _ = preprocess_data(df, features)

    best_k, wcss, silhouette = find_best_k(X)

    st.success(f"Suggested K = {best_k}  |  Best Silhouette = {round(silhouette, 3)}")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(2, 11)),
            y=wcss,
            mode="lines+markers"
        )
    )

    fig.update_layout(
        title="Elbow Method",
        xaxis_title="Clusters",
        yaxis_title="WCSS"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------
# K Selection
# -------------------------------------------------------

k = st.slider(
    "Number of Clusters",
    2,
    10,
    3
)

# -------------------------------------------------------
# Run K-Means
# -------------------------------------------------------

if st.button("Run K-Means"):

    with st.spinner("Training K-Means model..."):

        result, centers, silhouette, model, scaler, name_mapping = (
            perform_clustering(df, features, k)
        )

    st.session_state["pipeline"] = {
        "result": result,
        "centers": centers,
        "silhouette": silhouette,
        "model": model,
        "scaler": scaler,
        "name_mapping": name_mapping,
        "features": features
    }

    st.success("Model Trained Successfully")

pipeline = st.session_state.get("pipeline")

if pipeline is None:

    st.info("Click **Run K-Means** to train the model and view the dashboard.")

    st.stop()

result = pipeline["result"]

centers = pipeline["centers"]

features_used = pipeline["features"]

# -------------------------------------------------------
# Model Metrics
# -------------------------------------------------------

st.metric("Silhouette Score", round(pipeline["silhouette"], 3))

high = len(result[result.Customer_Type == "High Value"])

medium = len(result[result.Customer_Type == "Medium Value"])

low = len(result[result.Customer_Type == "Low Value"])

c1, c2, c3 = st.columns(3)

c1.metric("🟢 High Value", high)

c2.metric("🟡 Medium Value", medium)

c3.metric("🔴 Low Value", low)

st.markdown("---")

# -------------------------------------------------------
# Charts
# -------------------------------------------------------

st.subheader("Cluster Centers")

st.dataframe(centers)

st.markdown("---")

scatter_plot(result, features_used)

scatter_3d(result, features_used)

pie_chart(result)

bar_chart(result)

top_customers(result, features_used)

if show_hist:

    histogram(result, features_used)

if show_box:

    box_plot(result, features_used)

if show_heatmap:

    correlation_heatmap(result)

cluster_center_chart(centers)

radar_chart(centers)

parallel_coordinates(result, features_used)

cluster_summary_table(result, features_used)

st.markdown("---")

# -------------------------------------------------------
# Customer Search + Recommendations
# -------------------------------------------------------

customer_search(result)

st.markdown("---")

business_recommendation(result)

st.markdown("---")

# -------------------------------------------------------
# Predict New Customer
# -------------------------------------------------------

st.header("🔮 Predict New Customer Segment")

predict_inputs = {}

for feature in features_used:

    predict_inputs[feature] = st.number_input(
        feature,
        min_value=0.0,
        step=100.0
    )

if st.button("Predict Segment"):

    values = [predict_inputs[f] for f in features_used]

    segment = predict_customer(
        pipeline["model"],
        pipeline["scaler"],
        values,
        pipeline["name_mapping"]
    )

    if segment == "High Value":

        st.success(f"Predicted Segment : {segment} 🟢")

    elif segment == "Medium Value":

        st.warning(f"Predicted Segment : {segment} 🟡")

    else:

        st.error(f"Predicted Segment : {segment} 🔴")

st.markdown("---")

# -------------------------------------------------------
# Downloads
# -------------------------------------------------------

st.subheader("Download Results")

csv = result.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Clustered Dataset (CSV)",
    csv,
    "CustomerSegmentation.csv",
    "text/csv"
)

st.download_button(
    "Download Clustered Dataset (Excel)",
    to_excel_bytes(result),
    "CustomerSegmentation.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

with st.expander("Cluster Summary Table"):

    st.dataframe(cluster_summary(result, features_used))
