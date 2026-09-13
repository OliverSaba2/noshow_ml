"""
Shared preprocessing used by model_selection.py, train_model.py, and predict_examples.py.

Kept as a ColumnTransformer (not pre-applied to the data) so it can be dropped
into an sklearn Pipeline and re-fit inside each cross-validation fold.
"""

import json

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_feature_config(path: str = "data/feature_config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def build_preprocessor(feature_config: dict) -> ColumnTransformer:
    numeric_features = feature_config["numeric_features"]
    categorical_features = feature_config["categorical_features"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_pipeline(feature_config: dict, model) -> Pipeline:
    preprocessor = build_preprocessor(feature_config)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
