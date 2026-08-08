import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ----------------------------------------------------
# Scatter Plot
# ----------------------------------------------------

def scatter_plot(df, features):

    st.subheader("📈 Customer Segmentation")

    fig = px.scatter(
        df,
        x=features[0],
        y=features[1],
        color="Customer_Type",
        hover_data=["Customer_ID"],
        size=features[1],
        template="plotly_white"
    )

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# 3D Scatter Plot
# ----------------------------------------------------

def scatter_3d(df, features):

    if len(features) < 3:
        return

    st.subheader("🌍 3D Customer Segmentation")

    fig = px.scatter_3d(
        df,
        x=features[0],
        y=features[1],
        z=features[2],
        color="Customer_Type",
        hover_data=["Customer_ID"]
    )

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Pie Chart
# ----------------------------------------------------

def pie_chart(df):

    st.subheader("🥧 Customer Distribution")

    count = df["Customer_Type"].value_counts().reset_index()

    count.columns = ["Customer Type", "Count"]

    fig = px.pie(
        count,
        names="Customer Type",
        values="Count",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Bar Chart
# ----------------------------------------------------

def bar_chart(df):

    st.subheader("📊 Customer Count")

    count = df["Customer_Type"].value_counts().reset_index()

    count.columns = ["Customer Type", "Count"]

    fig = px.bar(
        count,
        x="Customer Type",
        y="Count",
        color="Customer Type",
        text="Count"
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Histogram
# ----------------------------------------------------

def histogram(df, features):

    st.subheader("📉 Feature Distribution")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x=features[0],
            color="Customer_Type",
            nbins=30
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.histogram(
            df,
            x=features[1],
            color="Customer_Type",
            nbins=30
        )

        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Box Plot
# ----------------------------------------------------

def box_plot(df, features):

    st.subheader("📦 Outlier Detection")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            df,
            x="Customer_Type",
            y=features[0],
            color="Customer_Type"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.box(
            df,
            x="Customer_Type",
            y=features[1],
            color="Customer_Type"
        )

        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Correlation Heatmap
# ----------------------------------------------------

def correlation_heatmap(df):

    st.subheader("🔥 Correlation Heatmap")

    numeric = df.select_dtypes(include="number")

    corr = numeric.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Cluster Centers
# ----------------------------------------------------

def cluster_center_chart(centers):

    st.subheader("🎯 Cluster Centers")

    fig = go.Figure()

    for column in centers.columns:

        fig.add_trace(
            go.Bar(
                x=centers.index.astype(str),
                y=centers[column],
                name=column
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Cluster",
        yaxis_title="Value"
    )

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Radar Chart
# ----------------------------------------------------

def radar_chart(centers):

    st.subheader("🕸 Cluster Profile")

    categories = centers.columns.tolist()

    fig = go.Figure()

    for i in range(len(centers)):

        values = centers.iloc[i].tolist()

        values += values[:1]

        fig.add_trace(

            go.Scatterpolar(

                r=values,

                theta=categories + [categories[0]],

                fill='toself',

                name=f'Cluster {i}'

            )

        )

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Customer Spending Ranking
# ----------------------------------------------------

def top_customers(df, features):

    st.subheader("🏆 Top 20 Customers")

    if "Customer_ID" not in df.columns:
        st.info("Customer_ID column not found - skipping ranking chart.")
        return

    metric = features[0]

    top = df.sort_values(

        metric,

        ascending=False

    ).head(20)

    fig = px.bar(

        top,

        x="Customer_ID",

        y=metric,

        color="Customer_Type"

    )

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Parallel Coordinates
# ----------------------------------------------------

def parallel_coordinates(df, features):

    st.subheader("📌 Feature Comparison")

    fig = px.parallel_coordinates(

        df,

        dimensions=features,

        color=df["Cluster"]

    )

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Cluster Summary Table
# ----------------------------------------------------

def cluster_summary_table(df, features):

    st.subheader("📋 Cluster Summary")

    summary = (

        df

        .groupby("Customer_Type")[features]

        .agg(["mean", "median", "max", "min"])

        .round(2)

    )

    st.dataframe(summary)


# ----------------------------------------------------
# Download Charts
# ----------------------------------------------------

def dashboard_header():

    st.markdown("""

# 📊 Customer Segmentation Dashboard

Machine Learning + Streamlit + Plotly

---""")