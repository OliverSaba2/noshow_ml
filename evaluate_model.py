"""
Evaluates the trained model on the held-out test set.

Run: python evaluate_model.py
Output: printed metrics + reports/confusion_matrix.png, reports/roc_curve.png,
        reports/feature_importance.csv, reports/feature_importance.png
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocessing import load_feature_config

TEST_PATH = os.path.join("data", "test.csv")
MODEL_PATH = os.path.join("models", "model.joblib")
REPORTS_DIR = "reports"


def main():
    feature_config = load_feature_config()
    target = feature_config["target"]
    feature_cols = feature_config["numeric_features"] + feature_config["categorical_features"]

    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("TEST SET METRICS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1:        {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["attended", "no_show"]))

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("CONFUSION MATRIX")
    print(cm)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["attended", "no_show"]).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))

    # ROC curve
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"))

    # feature importance (logistic regression coefficients, in log-odds)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        feature_names = preprocessor.get_feature_names_out()
        coefs = model.coef_[0]
        importance_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
        importance_df = importance_df.sort_values("abs_coefficient", ascending=False)
        importance_df.to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False)

        top = importance_df.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = np.where(top["coefficient"] > 0, "tab:red", "tab:blue")
        ax.barh(top["feature"], top["coefficient"], color=colors)
        ax.set_title("Top feature coefficients\n(red = raises no-show risk, blue = lowers it)")
        ax.set_xlabel("logistic regression coefficient (log-odds)")
        fig.tight_layout()
        fig.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"))

        print("\n" + "=" * 60)
        print("TOP FEATURES BY |COEFFICIENT|")
        print("=" * 60)
        print(importance_df.head(10).to_string(index=False))

    print(f"\nSaved plots and reports to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
