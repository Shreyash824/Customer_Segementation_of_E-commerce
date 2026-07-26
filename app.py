import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

from model import (
    preprocess_data,
    find_best_k,
    perform_clustering,
    save_model
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
   customer_search,
   business_recommendation
)
# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)
# -----------------------------
# Session State
# -----------------------------

if "trained" not in st.session_state:
    st.session_state.trained = False
# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("⚙ Dashboard")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["xlsx", "csv"]
)

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
# Upload Dataset
# -------------------------------------------------------

if uploaded_file is not None:

    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    # -----------------------------
    # Dataset Summary
    # -----------------------------

    st.subheader("Dataset Summary")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Customers",
        len(df)
    )

    col2.metric(
        "Features",
        len(df.columns)
    )

    col3.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )

    col4.metric(
        "Duplicate Rows",
        df.duplicated().sum()
    )

    if show_dataset:

        st.subheader("Dataset Preview")

        st.dataframe(df.head(20),use_container_width=True)

    st.markdown("---")

    # ----------------------------------
    # Feature Selection
    # ----------------------------------

    st.subheader("Select Features")

    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    features = st.multiselect(
        "Choose Numerical Features",
        numeric_columns,
        default=[
            "Annual_Spending",
            "Order_Count"
        ]
    )

    if len(features) < 2:

        st.warning("Select at least two features.")

        st.stop()

    required = ["Annual_Spending", "Order_Count"]

    missing = [c for c in required if c not in df.columns]

    if missing:
       st.error(f"Missing required columns: {missing}")
       st.stop()


    # ----------------------------------
    # Find Best K
    # ----------------------------------

    st.subheader("Find Best Number of Clusters")

    if st.button("Run Elbow Method"):

        X = preprocess_data(df,features)

        best_k,wcss,silhouette=find_best_k(X)

        st.success(f"Suggested K = {best_k}")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(2,11)),
                y=wcss,
                mode='lines+markers'
            )
        )

        fig.update_layout(
            title="Elbow Method",
            xaxis_title="Clusters",
            yaxis_title="WCSS"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ----------------------------------
    # K Selection
    # ----------------------------------

    k = st.slider(
        "Number of Clusters",
        2,
        10,
        3
    )

    # ----------------------------------
    # Run Clustering
    # ----------------------------------

    # ----------------------------------
# Run Clustering
# ----------------------------------

if st.button("Run K-Means"):

    X = preprocess_data(df, features)

    result, centers, silhouette, model = perform_clustering(
        df,
        X,
        features,
        k
    )

    save_model(model)

    st.session_state.result = result
    st.session_state.centers = centers
    st.session_state.silhouette = silhouette
    st.session_state.model = model
    st.session_state.trained = True

if st.session_state.trained:

    result = st.session_state.result
    centers = st.session_state.centers
    silhouette = st.session_state.silhouette
    model = st.session_state.model

    st.success("Model Trained Successfully")

    st.success("Model Trained Successfully")

    # EVERYTHING BELOW GOES HERE

    # KPI Cards
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🟢 High Value",
        len(result[result["Customer_Type"] == "High Value"])
    )

    c2.metric(
        "🟡 Medium Value",
        len(result[result["Customer_Type"] == "Medium Value"])
    )

    c3.metric(
        "🔴 Low Value",
        len(result[result["Customer_Type"] == "Low Value"])
    )

    st.markdown("---")

    st.subheader("Cluster Centers")
    st.dataframe(centers)

    # Charts
    scatter_plot(result, features)

    if len(features) >= 3:
        scatter_3d(result, features)

    pie_chart(result)

    bar_chart(result)

    if show_hist:
        histogram(result, features)

    if show_box:
        box_plot(result, features)

    if show_heatmap:
        correlation_heatmap(result)

    cluster_center_chart(centers)

    radar_chart(centers)

    top_customers(result)

    parallel_coordinates(result, features)

    cluster_summary_table(result, features)

    st.markdown("---")

    customer_search(result)

    st.markdown("---")

    business_recommendation(result)

    st.markdown("---")

    # Predict New Customer

    st.subheader("Predict New Customer")

    annual = st.number_input(
        "Annual Spending",
        min_value=0.0,
        key="annual"
    )

    orders = st.number_input(
        "Order Count",
        min_value=0,
        key="orders"
    )

    if st.button("Predict Customer"):

        spending = centers.iloc[:, 0]

        if annual >= spending.max():
            st.success("🟢 High Value Customer")

        elif annual >= spending.mean():
            st.warning("🟡 Medium Value Customer")

        else:
            st.error("🔴 Low Value Customer")

    # Download CSV

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download CSV",
        csv,
        "CustomerSegmentation.csv",
        "text/csv"
    )

    # Download Excel

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result.to_excel(writer, index=False)

    st.download_button(
        "⬇ Download Excel",
        buffer.getvalue(),
        "CustomerSegmentation.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Upload your customer dataset to begin.")
