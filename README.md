# 🛒 Customer Segmentation Dashboard (Flipkart)

Customer purchase behaviour segmentation using K-Means Clustering, built with
Streamlit + Plotly + Scikit-Learn.

Cluster customers into:

- 🟢 High Value
- 🟡 Medium Value
- 🔴 Low Value

## Features

- Upload your own CSV / Excel dataset, or use the built-in sample Flipkart dataset
- Fully automatic — features and the number of clusters (K) are chosen by the
  model itself using the silhouette score
- Segments are auto-named High / Medium / Low value based on standardized
  cluster centers, so datasets with any number of features are segmented
  correctly regardless of feature scale
- 12 interactive Plotly charts (2D/3D scatter, pie, bar, histograms, box plots,
  correlation heatmap, radar, parallel coordinates, cluster centers, top customers)
- Predict the segment of a new customer from the trained model
- Cluster summaries + business recommendations + CSV/Excel export

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push this folder to a GitHub repository.
2. Go to <https://share.streamlit.io> → **Create app**.
3. Select the repository, set **Main file** to `app.py`, and click **Deploy**.
4. Done — the sample dataset works out of the box, no upload needed.

## Project structure

```
app.py              Streamlit dashboard (entry point)
model.py            Preprocessing, K-Means, elbow, prediction
visualization.py    Plotly charts
recommendation.py   Customer search + business insights
sample_data.py      Synthetic Flipkart-style dataset
.streamlit/config.toml   Streamlit theme/config
requirements.txt    Dependencies
```

## Technologies

Python • Streamlit • Scikit-Learn • Plotly • Pandas • NumPy