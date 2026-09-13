"""
Compares candidate models via cross-validation on the training set and
picks the one to move forward with.

Run: python model_selection.py
Output: reports/model_selection_results.csv, data/selected_model.json
"""

import json
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from models import CANDIDATES
from preprocessing import build_pipeline, load_feature_config

TRAIN_PATH = os.path.join("data", "train.csv")
REPORTS_DIR = "reports"

# Scored on more than accuracy: the target is imbalanced (~30% no-show), so a
# model that just predicts "attended" every time would score ~70% accuracy
# while being useless. ROC-AUC and F1 are what actually separate the models.
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main():
    feature_config = load_feature_config()
    target = feature_config["target"]
    feature_cols = feature_config["numeric_features"] + feature_config["categorical_features"]

    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df[feature_cols]
    y = train_df[target]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for name, model in CANDIDATES.items():
        pipeline = build_pipeline(feature_config, model)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            row[metric] = scores[f"test_{metric}"].mean()
        results.append(row)
        print(f"{name}: " + ", ".join(f"{m}={row[m]:.3f}" for m in SCORING))

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(REPORTS_DIR, "model_selection_results.csv"), index=False)

    best_model = results_df.iloc[0]["model"]
    print(f"\nSelected model: {best_model} (highest mean ROC-AUC across 5-fold CV)")

    with open(os.path.join("data", "selected_model.json"), "w") as f:
        json.dump({"selected_model": best_model}, f, indent=2)


if __name__ == "__main__":
    main()
