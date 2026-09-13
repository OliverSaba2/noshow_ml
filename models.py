"""
Candidate model definitions, shared by model_selection.py and train_model.py
so hyperparameters live in exactly one place.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "decision_tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}
