"""
Predicts no-show risk for patients.

Two modes:
1. No arguments: scores a handful of built-in example patients so you can
   see the model's output without needing a file.
2. --input some.csv [--output preds.csv]: scores patients from a CSV with
   the same raw columns as data/appointments.csv (no_show not required).

Run: python predict_examples.py
     python predict_examples.py --input new_patients.csv --output preds.csv
"""

import argparse

import joblib
import pandas as pd

from prepare_data import clean_and_engineer
from preprocessing import load_feature_config

MODEL_PATH = "models/model.joblib"

EXAMPLE_PATIENTS = pd.DataFrame(
    [
        # Long-time patient, good attendance history, reminder sent, morning slot.
        dict(
            age=52, gender="F", appointment_type="follow_up", days_before_appointment=3,
            previous_appointments=12, previous_no_shows=0, weekday="Tuesday",
            appointment_time="morning", reminder_sent=1, new_patient=0,
        ),
        # Brand-new patient, no reminder, long lead time, evening slot -> high risk.
        dict(
            age=24, gender="M", appointment_type="dental", days_before_appointment=45,
            previous_appointments=0, previous_no_shows=0, weekday="Monday",
            appointment_time="evening", reminder_sent=0, new_patient=1,
        ),
        # Returning patient with a history of missing appointments, no reminder.
        dict(
            age=31, gender="M", appointment_type="general", days_before_appointment=21,
            previous_appointments=8, previous_no_shows=5, weekday="Friday",
            appointment_time="afternoon", reminder_sent=0, new_patient=0,
        ),
        # Regular patient, reminder sent, appointment is in a couple of days.
        dict(
            age=67, gender="F", appointment_type="specialist", days_before_appointment=2,
            previous_appointments=6, previous_no_shows=1, weekday="Wednesday",
            appointment_time="morning", reminder_sent=1, new_patient=0,
        ),
    ]
)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    feature_config = load_feature_config()
    feature_cols = feature_config["numeric_features"] + feature_config["categorical_features"]

    prepared = clean_and_engineer(df)
    pipeline = joblib.load(MODEL_PATH)

    proba = pipeline.predict_proba(prepared[feature_cols])[:, 1]
    pred = pipeline.predict(prepared[feature_cols])

    result = df.copy()
    result["no_show_probability"] = proba.round(3)
    result["predicted_no_show"] = pred
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict appointment no-show risk.")
    parser.add_argument("--input", help="CSV of patients to score (same raw columns as data/appointments.csv)")
    parser.add_argument("--output", help="Where to write predictions CSV (defaults to printing only)")
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input)
    else:
        df = EXAMPLE_PATIENTS

    result = predict(df)
    print(result.to_string(index=False))

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nWrote predictions to {args.output}")


if __name__ == "__main__":
    main()
