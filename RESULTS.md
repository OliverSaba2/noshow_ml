# Results

Numbers from this run of the pipeline (`data/appointments.csv`, `random_state=42`,
reproducible by re-running the scripts in the README). For the reasoning behind these
choices, see [EXPLANATION.md](EXPLANATION.md).

## Dataset

- 3000 appointments, no missing values
- No-show rate: **15.9%** (imbalanced, skewed toward attendance)
- Raw group differences are mostly small; the clearest one is reminders: 22.2% no-show
  without one vs. 14.2% with one

## Model comparison (5-fold cross-validation, training set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic regression** ⭐ | 0.694 | 0.273 | 0.547 | 0.364 | **0.691** |
| Random forest | 0.797 | 0.383 | 0.424 | 0.400 | 0.678 |
| Gradient boosting | 0.836 | 0.444 | 0.115 | 0.182 | 0.676 |
| Decision tree | 0.669 | 0.262 | 0.565 | 0.355 | 0.648 |

Selected: **logistic regression** — best ROC-AUC, and the most interpretable of the four.
(Random forest and gradient boosting post higher accuracy by mostly just predicting
"attended" - their low recall gives that away.)

## Test set performance

| Metric | Value |
|---|---|
| Accuracy | 0.703 |
| Precision | 0.289 |
| Recall | 0.583 |
| F1 | 0.386 |
| ROC-AUC | 0.701 |

Close to the cross-validation numbers above, so the model generalizes rather than having
gotten lucky during selection.

**Confusion matrix:**

| | Predicted attended | Predicted no-show |
|---|---|---|
| **Actually attended** (504) | 366 | 138 |
| **Actually no-show** (96) | 40 | 56 |

The model catches 56/96 (58%) of real no-shows, at the cost of 138 false alarms - the
deliberate trade-off given a missed no-show costs more than an extra reminder.

Plots: `reports/confusion_matrix.png`, `reports/roc_curve.png`

## What drives the prediction

Top logistic regression coefficients (positive = raises no-show risk):

| Feature | Effect |
|---|---|
| Personal history of no-shows | ↑ risk (by far the strongest) |
| Procedure appointment | ↓ risk |
| No reminder sent | ↑ risk |
| New Consultation appointment | ↑ risk |
| Friday | ↑ risk |
| Monday / Thursday | ↓ risk |
| New patient | ↑ risk |

Full table: `reports/feature_importance.csv` · chart: `reports/feature_importance.png`

## Example predictions

| Patient | Risk score | Flagged? |
|---|---|---|
| Regular patient, good history, reminder sent, morning slot | 0.23 | No |
| Brand-new patient, 45-day wait, no reminder, evening slot | 0.56 | Yes |
| Returning patient with a history of no-shows, no reminder | 0.98 | Yes |
| Regular patient, reminder sent, appointment in 2 days | 0.43 | No |

The model separates low- and high-risk patients cleanly, and a personal history of missed
appointments dominates the score more than any single-visit detail.
