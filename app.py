import io

import pandas as pd
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
    pie_chart,
    bar_chart,
    histogram,
    correlation_heatmap,
    top_customers,
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
# Styling
# -------------------------------------------------------

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #7b1fa2, #9c27b0);
        border-radius: 16px;
        padding: 18px 16px;
        box-shadow: 0 4px 14px rgba(123, 31, 162, 0.25);
    }
    div[data-testid="stMetricLabel"] {
        color: #f3e5f5 !important;
        font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.7rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #4a148c;
        margin: 1.4rem 0 0.7rem 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #f0e6fa;
    }
    .segment-card {
        background: #ffffff;
        border: 1px solid #ecebf3;
        border-left: 5px solid #9c27b0;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(123, 31, 162, 0.06);
    }
    .seg-name { font-size: 0.95rem; font-weight: 600; color: #4a148c; }
    .seg-count { font-size: 1.7rem; font-weight: 700; color: #262730; }
    .seg-pct { font-size: 0.8rem; color: #6b7280; }
    .empty-card {
        text-align: center;
        background: #ffffff;
        border: 1px dashed #c9b8e8;
        border-radius: 16px;
        padding: 48px 24px;
        margin-top: 24px;
        color: #6b7280;
    }
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
    }
    .seg-banner {
        background: linear-gradient(135deg, #4a148c, #7b1fa2 60%, #ab47bc);
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 6px 20px rgba(74, 20, 140, 0.3);
        margin: 10px 0 18px 0;
    }
    .seg-banner-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
    }
    .seg-banner-sub {
        font-size: 0.95rem;
        color: #f3e5f5;
        margin-top: 6px;
    }
    .seg-banner-chip {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 3px 12px;
        margin-right: 8px;
        font-weight: 600;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def section_header(text):
    st.markdown(
        f'<p class="section-header">{text}</p>',
        unsafe_allow_html=True
    )


def segment_cards(result):
    total = len(result)

    counts = {
        "High Value": len(result[result.Customer_Type == "High Value"]),
        "Medium Value": len(result[result.Customer_Type == "Medium Value"]),
        "Low Value": len(result[result.Customer_Type == "Low Value"]),
    }

    colors = {
        "High Value": "#2e7d32",
        "Medium Value": "#f9a825",
        "Low Value": "#d32f2f",
    }

    backgrounds = {
        "High Value": "#e8f5e9",
        "Medium Value": "#fff8e1",
        "Low Value": "#ffebee",
    }

    emojis = {
        "High Value": "🟢",
        "Medium Value": "🟡",
        "Low Value": "🔴",
    }

    c1, c2, c3 = st.columns(3)

    for col, name in zip([c1, c2, c3], counts):

        count = counts[name]

        pct = (count / total * 100) if total else 0

        with col:

            st.markdown(
                f"""
                <div class="segment-card"
                     style="border-left-color:{colors[name]};
                            background:{backgrounds[name]};">
                    <div class="seg-name">{emojis[name]} {name}</div>
                    <div class="seg-count" style="color:{colors[name]};">
                        {count:,}
                    </div>
                    <div class="seg-pct">{pct:.1f}% of customers</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def to_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("📊 Dashboard")

data_source = st.sidebar.radio(
    "Data Source",
    ["Sample Dataset", "Upload Dataset"]
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

show_dataset = st.sidebar.checkbox("Show Dataset Preview", True)

with st.sidebar.expander("Chart Options"):
    show_hist = st.checkbox("Histogram", True)
    show_heatmap = st.checkbox("Correlation Heatmap", True)

# -------------------------------------------------------
# Main Title
# -------------------------------------------------------

st.title("🛒 Flipkart Customer Segmentation")

st.caption(
    "Segment customers into High, Medium and Low value groups "
    "using K-Means clustering."
)

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------

if data_source == "uploaded" and uploaded_file is not None:

    @st.cache_data(show_spinner=False)
    def read_uploaded_file(uploaded_file):
        if uploaded_file.name.lower().endswith("csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)

    df = read_uploaded_file(uploaded_file)

elif data_source == "sample":

    df = load_sample_data()

else:

    st.markdown(
        """
        <div class="empty-card">
            <div style="font-size:3rem;">📂</div>
            <div style="font-weight:600; color:#4a148c; margin-top:8px;">
                No dataset loaded
            </div>
            <div style="margin-top:6px;">
                Pick <b>Sample Dataset</b> in the sidebar, or upload
                your own CSV / Excel file.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()

# -------------------------------------------------------
# 1. Data Overview
# -------------------------------------------------------

section_header("1 · Data Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Customers", len(df))

c2.metric("Features", len(df.columns))

c3.metric("Missing Values", df.isnull().sum().sum())

c4.metric("Duplicate Rows", df.duplicated().sum())

if show_dataset:

    st.dataframe(df.head(20), use_container_width=True)

# -------------------------------------------------------
# 2. Model Training (automatic)
# -------------------------------------------------------

features = get_clustering_features(df)

excluded = [
    c for c in df.select_dtypes(include="number").columns
    if c not in features
]

if len(features) < 2:

    st.error(
        "At least two numeric features are required for clustering. "
        "Check the dataset."
    )

    st.stop()

fingerprint = (
    len(df),
    tuple(map(str, df.columns)),
    int(pd.util.hash_pandas_object(df, index=False).sum())
)

if (
    st.session_state.get("fp") != fingerprint
    or st.session_state.get("pipeline") is None
):

    try:

        with st.spinner("Training K-Means model..."):

            X, _ = preprocess_data(df, features)

            best_k, _, _ = find_best_k(X)

            result, _, silhouette, model, scaler, name_mapping = (
                perform_clustering(df, features, best_k)
            )

    except Exception as exc:

        st.error(
            f"Could not train the model: {exc}. "
            "Check the dataset and try again."
        )

        st.stop()

    st.session_state["pipeline"] = {
        "result": result,
        "silhouette": silhouette,
        "model": model,
        "scaler": scaler,
        "name_mapping": name_mapping,
        "features": features,
        "k": best_k
    }

    st.session_state["fp"] = fingerprint

pipeline = st.session_state["pipeline"]

result = pipeline["result"]

features_used = pipeline["features"]

# -------------------------------------------------------
# 2. Customer Segmentation
# -------------------------------------------------------

excluded_note = (
    f" &nbsp;·&nbsp; Skipped id-like/constant: {', '.join(excluded)}"
    if excluded else ""
)

st.markdown(
    f"""
    <div class="seg-banner">
        <div class="seg-banner-title">🎯 Customer Segmentation Results</div>
        <div class="seg-banner-sub">
            <span class="seg-banner-chip">{len(result):,} customers</span>
            <span class="seg-banner-chip">{len(features_used)} features</span>
            <span class="seg-banner-chip">K = {pipeline['k']} (auto)</span>
            <span class="seg-banner-chip">
                Silhouette = {round(pipeline['silhouette'], 3)}
            </span>
            <br>
            <span style="color:#e1bee7; font-size:0.85rem;">
                Features: {', '.join(features_used)}
            </span>
            {excluded_note}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

segment_cards(result)

st.markdown("---")

left_col, right_col = st.columns([1, 1.15])

with left_col:

    pie_chart(result)

with right_col:

    cluster_summary_table(result, features_used)

st.markdown("---")

st.subheader("👥 Segmented Customers")

preview_cols = [
    c for c in ["Customer_ID", *features_used, "Customer_Type"]
    if c in result.columns
]

st.dataframe(result[preview_cols].head(10), use_container_width=True)

st.markdown("---")

scatter_plot(result, features_used)

bar_chart(result)

top_customers(result, features_used)

if show_hist:

    histogram(result, features_used)

if show_heatmap:

    correlation_heatmap(result)

st.markdown("---")

# -------------------------------------------------------
# 3. Search & Insights
# -------------------------------------------------------

section_header("3 · Search & Insights")

customer_search(result)

business_recommendation(result)

st.markdown("---")

# -------------------------------------------------------
# 4. Predict New Customer
# -------------------------------------------------------

section_header("4 · Predict New Customer")

predict_inputs = {}

cols = st.columns(min(len(features_used), 3))

for i, feature in enumerate(features_used):

    with cols[i % len(cols)]:

        predict_inputs[feature] = st.number_input(
            feature,
            min_value=0.0,
            step=100.0
        )

if st.button("Predict Segment", type="primary", use_container_width=True):

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
# 5. Export
# -------------------------------------------------------

section_header("5 · Export")

c1, c2 = st.columns(2)

with c1:

    st.download_button(
        "Download CSV",
        result.to_csv(index=False).encode("utf-8"),
        "CustomerSegmentation.csv",
        "text/csv",
        use_container_width=True
    )

with c2:

    st.download_button(
        "Download Excel",
        to_excel_bytes(result),
        "CustomerSegmentation.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with st.expander("Cluster Summary Table"):

    st.dataframe(cluster_summary(result, features_used))
