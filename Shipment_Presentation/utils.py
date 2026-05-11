import pandas as pd
import numpy as np
import joblib


def load_artifacts(model_path="models/shipment_model.pkl",
                   columns_path="models/model_columns.pkl"):
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    return model, model_columns


def preprocess_input(input_df: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    data = input_df.copy()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].fillna("Unknown")
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col].fillna(data[col].median() if not data[col].isna().all() else 0)

    data = pd.get_dummies(data, drop_first=True)
    data = data.reindex(columns=model_columns, fill_value=0)
    return data


def prediction_label(pred):
    return "On-Time Delivery" if int(pred) == 1 else "Delayed Delivery"


def confidence_from_proba(proba: float) -> int:
    return int(round(proba * 100))