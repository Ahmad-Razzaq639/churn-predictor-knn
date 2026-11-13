"""Shared preprocessing (same as the notebook), used by both train.py and app.py.
"""
import pandas as pd

BINARY_YES_NO = ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
                 "MultipleLines", "OnlineSecurity", "OnlineBackup",
                 "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
ONEHOT_COLS = ["InternetService", "Contract", "PaymentMethod"]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
PASSTHROUGH_COLS = ["gender", "SeniorCitizen"] + BINARY_YES_NO

RAW_COLUMNS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
               "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
               "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
               "MonthlyCharges", "TotalCharges"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Raw Telco columns -> binary-encoded columns (one-hot and scaling happen in the pipeline)."""
    d = df.copy()
    d = d[RAW_COLUMNS]
    d["gender"] = d["gender"].map({"Female": 0, "Male": 1})
    for c in BINARY_YES_NO:
        d[c] = (d[c].replace({"No phone service": "No", "No internet service": "No"})
                    .map({"No": 0, "Yes": 1}))
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    return d
