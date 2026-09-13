"""
Trains the model chosen by model_selection.py on the full training set and
saves it to disk.

Run: python train_model.py
Output: models/model.joblib
"""

import json
import os

import joblib
import pandas as pd

from models import CANDIDATES
from preprocessing import build_pipeline, load_feature_config

TRAIN_PATH = os.path.join("data", "train.csv")
SELECTED_MODEL_PATH = os.path.join("data", "selected_model.json")
MODEL_OUT_PATH = os.path.join("models", "model.joblib")


def main():
    feature_config = load_feature_config()
    target = feature_config["target"]
    feature_cols = feature_config["numeric_features"] + feature_config["categorical_features"]

    with open(SELECTED_MODEL_PATH) as f:
        selected_model_name = json.load(f)["selected_model"]

    train_df = pd.read_csv(TRAIN_PATH)
    X_train = train_df[feature_cols]
    y_train = train_df[target]

    model = CANDIDATES[selected_model_name]
    pipeline = build_pipeline(feature_config, model)
    pipeline.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT_PATH)

    train_accuracy = pipeline.score(X_train, y_train)
    print(f"Trained model: {selected_model_name}")
    print(f"Training-set accuracy: {train_accuracy:.3f}")
    print(f"Saved pipeline to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
