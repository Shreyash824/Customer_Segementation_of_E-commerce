import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ----------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------

def get_clustering_features(df):
    """
    Returns numeric columns that are safe/useful for clustering.

    Drop rules:
    - non-numeric columns
    - constant columns (zero variance)
    - id-like columns (name based: id/key/index/code, or a fully
      unique integer column e.g. numeric Customer_ID)

    Continuous monetary/rating features keep all their rows and
    are never dropped, so datasets with many features are used
    to their full extent.
    """

    total_rows = len(df)

    selected = []

    for col in df.select_dtypes(include="number").columns:

        s = df[col].dropna()

        if s.empty or s.nunique() <= 1:
            continue

        name_lower = str(col).lower()

        if any(tok in name_lower for tok in ("id", "key", "index", "code")):
            continue

        if (
            pd.api.types.is_integer_dtype(s)
            and s.nunique() == total_rows
        ):
            continue

        selected.append(col)

    return selected


# ----------------------------------------------------------
# Data Preprocessing
# ----------------------------------------------------------

def _clean_features(df, features):
    """
    Returns a cleaned copy of the selected numeric features.
    Handles missing values and infinite values.
    """
    data = df[features].copy()

    data = data.fillna(data.mean(numeric_only=True))

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.fillna(0.0)

    return data


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
    scaler : StandardScaler
    """

    data = _clean_features(df, features)

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data.to_numpy())

    return scaled_data, scaler


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

    best_index = int(np.argmax(silhouette_scores))

    best_k = best_index + 2

    best_score = silhouette_scores[best_index]

    return best_k, wcss, best_score


# ----------------------------------------------------------
# Cluster Naming
# ----------------------------------------------------------

def assign_cluster_names(df, scaled_centers, features):
    """
    Automatically assign:
    High Value
    Medium Value
    Low Value

    based on the average of the SCALED (standardized) cluster
    centers. Because every feature is standardized, all
    features contribute equally - segment labels stay correct
    even when features have very different scales or there
    are many features.

    Returns the DataFrame and the cluster -> name mapping.
    """

    score = np.asarray(scaled_centers).mean(axis=1)

    sorted_cluster = np.argsort(score)

    mapping = {}

    if len(sorted_cluster) == 3:

        mapping[int(sorted_cluster[0])] = "Low Value"

        mapping[int(sorted_cluster[1])] = "Medium Value"

        mapping[int(sorted_cluster[2])] = "High Value"

    else:

        for i, cluster in enumerate(sorted_cluster):
            mapping[int(cluster)] = f"Cluster {i+1}"

    df = df.copy()

    df["Customer_Type"] = df["Cluster"].map(mapping)

    return df, mapping


# ----------------------------------------------------------
# Perform Clustering
# ----------------------------------------------------------

def perform_clustering(df, features, k):
    """
    Preprocesses, clusters, names segments and returns:

    result_df, center_df, silhouette, model, scaler, name_mapping
    """

    data = _clean_features(df, features)

    scaler = StandardScaler()

    X = scaler.fit_transform(data.to_numpy())

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    df = df.copy()

    df["Cluster"] = labels

    silhouette = silhouette_score(X, labels)

    centers = scaler.inverse_transform(
        model.cluster_centers_
    )

    center_df = pd.DataFrame(
        centers,
        columns=features
    )

    df, name_mapping = assign_cluster_names(
        df,
        model.cluster_centers_,
        features
    )

    return (
        df,
        center_df,
        silhouette,
        model,
        scaler,
        name_mapping
    )


# ----------------------------------------------------------
# Save / Load Model
# ----------------------------------------------------------

def save_model(model,
               filename="customer_segmentation.pkl"):

    joblib.dump(
        model,
        filename
    )


def load_model(
        filename="customer_segmentation.pkl"):

    return joblib.load(filename)


# ----------------------------------------------------------
# Predict New Customer
# ----------------------------------------------------------

def predict_customer(
        model,
        scaler,
        values,
        name_mapping=None):

    values = np.array(values, dtype=float).reshape(1, -1)

    scaled = scaler.transform(values)

    cluster = int(model.predict(scaled)[0])

    if name_mapping is not None:
        return name_mapping.get(cluster, f"Cluster {cluster + 1}")

    return cluster


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