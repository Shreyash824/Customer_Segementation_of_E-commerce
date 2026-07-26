import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ----------------------------------------------------------
# Data Preprocessing
# ----------------------------------------------------------

def preprocess_data(df, features):
    """
    Cleans and scales the selected features.

    Parameters
    ----------
    df : DataFrame
    features : list

    Returns
    -------
    scaled_data : ndarray
    """

    data = df[features].copy()

    # Fill missing values
    data = data.fillna(data.mean(numeric_only=True))

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)

    return scaled_data


# ----------------------------------------------------------
# Find Best K using Elbow + Silhouette
# ----------------------------------------------------------

def find_best_k(X):
    """
    Returns:
    best_k
    wcss list
    silhouette score of best model
    """

    wcss = []

    silhouette_scores = []

    for k in range(2, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        wcss.append(model.inertia_)

        score = silhouette_score(X, labels)

        silhouette_scores.append(score)

    best_index = np.argmax(silhouette_scores)

    best_k = best_index + 2

    best_score = silhouette_scores[best_index]

    return best_k, wcss, best_score


# ----------------------------------------------------------
# Cluster Naming
# ----------------------------------------------------------

def assign_cluster_names(df, centers, features):
    """
    Automatically assign:
    High Value
    Medium Value
    Low Value

    based on average feature values.
    """

    score = centers.mean(axis=1)

    sorted_cluster = np.argsort(score)

    mapping = {}

    if len(sorted_cluster) == 3:

        mapping[sorted_cluster[0]] = "Low Value"

        mapping[sorted_cluster[1]] = "Medium Value"

        mapping[sorted_cluster[2]] = "High Value"

    else:

        for i, cluster in enumerate(sorted_cluster):
            mapping[cluster] = f"Cluster {i+1}"

    df["Customer_Type"] = df["Cluster"].map(mapping)

    return df


# ----------------------------------------------------------
# Perform Clustering
# ----------------------------------------------------------

def perform_clustering(df, X, features, k):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    df = df.copy()

    df["Cluster"] = labels

    silhouette = silhouette_score(X, labels)

    scaler = StandardScaler()

    scaler.fit(df[features])

    centers = scaler.inverse_transform(
        model.cluster_centers_
    )

    center_df = pd.DataFrame(
        centers,
        columns=features
    )

    df = assign_cluster_names(
        df,
        centers,
        features
    )

    return (
        df,
        center_df,
        silhouette,
        model
    )


# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

def save_model(model,
               filename="customer_segmentation.pkl"):

    joblib.dump(
        model,
        filename
    )


# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

def load_model(
        filename="customer_segmentation.pkl"):

    return joblib.load(filename)


# ----------------------------------------------------------
# Predict New Customer
# ----------------------------------------------------------

def predict_customer(
        model,
        scaler,
        values):

    values = np.array(values).reshape(1, -1)

    scaled = scaler.transform(values)

    prediction = model.predict(scaled)

    return prediction[0]


# ----------------------------------------------------------
# Cluster Summary
# ----------------------------------------------------------

def cluster_summary(df, features):

    summary = (
        df
        .groupby("Customer_Type")[features]
        .mean()
        .round(2)
    )

    summary["Customers"] = (
        df.groupby("Customer_Type").size()
    )

    return summary.reset_index()


# ----------------------------------------------------------
# Dataset Statistics
# ----------------------------------------------------------

def dataset_statistics(df):

    stats = {

        "Total Customers":
        len(df),

        "Missing Values":
        int(df.isnull().sum().sum()),

        "Duplicate Rows":
        int(df.duplicated().sum()),

        "Total Columns":
        len(df.columns)

    }

    return stats
