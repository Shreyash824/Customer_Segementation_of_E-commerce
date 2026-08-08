import numpy as np
import pandas as pd


# ----------------------------------------------------------
# Synthetic Flipkart Customer Dataset
# ----------------------------------------------------------

SEGMENTS = [
    {"name": "Low", "n": 600,
     "spending": (2500, 800), "orders": (3, 1.2),
     "tenure": (8, 5), "age": (24, 6)},
    {"name": "Medium", "n": 600,
     "spending": (8000, 1800), "orders": (10, 3.5),
     "tenure": (28, 14), "age": (33, 8)},
    {"name": "High", "n": 300,
     "spending": (20000, 5000), "orders": (24, 6.0),
     "tenure": (50, 15), "age": (42, 9)},
]

REGIONS = ["North", "South", "East", "West", "Central"]

GENDERS = ["Male", "Female", "Other"]


def load_sample_data(seed=42):
    """
    Builds a realistic Flipkart-style customer dataset so the
    dashboard is fully usable without an uploaded file.

    Returns a pandas DataFrame with the following columns:
    Customer_ID, Age, Gender, Region, Annual_Spending,
    Order_Count, Avg_Order_Value, Tenure_Months.
    """

    rng = np.random.default_rng(seed)

    records = []

    total = 0

    for seg in SEGMENTS:

        seg_n = seg["n"]

        spending = np.clip(
            rng.normal(seg["spending"][0], seg["spending"][1], seg_n),
            500.0, None
        )

        orders = np.clip(
            rng.normal(seg["orders"][0], seg["orders"][1], seg_n),
            1.0, None
        )

        tenure = np.clip(
            rng.normal(seg["tenure"][0], seg["tenure"][1], seg_n),
            1.0, 72.0
        )

        age = np.clip(
            rng.normal(seg["age"][0], seg["age"][1], seg_n),
            18.0, 70.0
        )

        for i in range(seg_n):

            annual = round(float(spending[i]), 2)

            order_count = int(round(float(orders[i])))

            avg_value = round(annual / max(order_count, 1), 2)

            records.append({
                "Customer_ID":
                    f"FLPCUST{total + i + 1:05d}",
                "Age":
                    int(age[i]),
                "Gender":
                    GENDERS[rng.integers(0, len(GENDERS))],
                "Region":
                    REGIONS[rng.integers(0, len(REGIONS))],
                "Annual_Spending":
                    annual,
                "Order_Count":
                    order_count,
                "Avg_Order_Value":
                    avg_value,
                "Tenure_Months":
                    int(round(tenure[i])),
            })

        total += seg_n

    return pd.DataFrame(records)


if __name__ == "__main__":

    sample = load_sample_data()

    print(sample.head(10))

    print(sample.describe())