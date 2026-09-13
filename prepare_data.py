"""
Cleans, engineers features on, and splits the appointments dataset.

Encoding/scaling is deliberately NOT done here — it's built as part of the
model pipeline in model_selection.py / train_model.py instead, so it gets
re-fit inside each cross-validation fold and never leaks test-set statistics
into training.

Run: python prepare_data.py
Output: data/train.csv, data/test.csv, data/feature_config.json
"""

import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("data", "appointments.csv")
TARGET = "no_show"

NUMERIC_FEATURES = [
    "age",
    "days_before_appointment",
    "previous_appointments",
    "previous_no_shows",
    "previous_no_show_rate",
    "reminder_sent",
    "new_patient",
]
CATEGORICAL_FEATURES = ["gender", "appointment_type", "weekday", "appointment_time"]


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic missing-value handling (synthetic data has none, but a real
    # dataset should never be assumed clean).
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Sanity: previous_no_shows can never exceed previous_appointments.
    df["previous_no_shows"] = df[["previous_no_shows", "previous_appointments"]].min(axis=1)

    # Feature engineering: a patient's personal no-show rate is a much
    # stronger signal than the raw counts on their own.
    has_history = df["previous_appointments"] > 0
    df["previous_no_show_rate"] = 0.0
    df.loc[has_history, "previous_no_show_rate"] = (
        df.loc[has_history, "previous_no_shows"] / df.loc[has_history, "previous_appointments"]
    )

    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = clean_and_engineer(df)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )

    os.makedirs("data", exist_ok=True)
    train_df.to_csv(os.path.join("data", "train.csv"), index=False)
    test_df.to_csv(os.path.join("data", "test.csv"), index=False)

    feature_config = {
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    with open(os.path.join("data", "feature_config.json"), "w") as f:
        json.dump(feature_config, f, indent=2)

    print(f"train: {train_df.shape}, test: {test_df.shape}")
    print(f"train no_show rate: {train_df[TARGET].mean():.3f}")
    print(f"test no_show rate:  {test_df[TARGET].mean():.3f}")


if __name__ == "__main__":
    main()
