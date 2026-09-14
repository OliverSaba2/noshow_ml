# Results

Numbers from this run of the pipeline (synthetic dataset, `random_state=42`, reproducible
by re-running the scripts in the README). For the reasoning behind these choices, see
[EXPLANATION.md](EXPLANATION.md).

## Dataset

- 3000 appointments, no missing values
- No-show rate: **30.4%** (moderately imbalanced)
- Strongest raw signals: no reminder sent (41% no-show vs. 26% with one), new patients
  (48% vs. 29%), longer lead time before the appointment

## Model comparison (5-fold cross-validation, training set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic regression** ⭐ | 0.648 | 0.444 | 0.641 | 0.525 | **0.716** |
| Random forest | 0.650 | 0.442 | 0.575 | 0.500 | 0.690 |
| Gradient boosting | 0.702 | 0.517 | 0.264 | 0.349 | 0.681 |
| Decision tree | 0.618 | 0.403 | 0.521 | 0.452 | 0.633 |

Selected: **logistic regression** — best ROC-AUC, and the most interpretable of the four.
(Gradient boosting has the highest accuracy but the worst recall — it mostly plays it safe
and predicts "attended," which is exactly the failure mode accuracy alone hides.)

## Test set performance (600 held-out appointments, never used before this point)

| Metric | Value |
|---|---|
| Accuracy | 0.685 |
| Precision | 0.488 |
| Recall | 0.689 |
| F1 | 0.571 |
| ROC-AUC | 0.726 |

These are close to the cross-validation numbers above, which means the model generalizes
rather than having gotten lucky during selection.

**Confusion matrix:**

| | Predicted attended | Predicted no-show |
|---|---|---|
| **Actually attended** (417) | 285 | 132 |
| **Actually no-show** (183) | 57 | 126 |

The model catches 126/183 (69%) of real no-shows. It over-flags some attendees as at-risk
(132 false alarms), which is the deliberate trade-off — see EXPLANATION.md for why missing
a no-show is costlier than a false alarm here.

Plots: `reports/confusion_matrix.png`, `reports/roc_curve.png`

## What drives the prediction

Top logistic regression coefficients (positive = raises no-show risk):

| Feature | Effect |
|---|---|
| Longer lead time before appointment | ↑ risk |
| Younger age | ↑ risk |
| No reminder sent | ↑ risk |
| Evening appointment | ↑ risk |
| Morning appointment | ↓ risk |
| New patient | ↑ risk |
| Personal no-show history | ↑ risk |

Full table: `reports/feature_importance.csv` · chart: `reports/feature_importance.png`

## Example predictions

| Patient | Risk score | Flagged? |
|---|---|---|
| Regular patient, reminder sent, morning slot | 0.12 | No |
| Regular patient, reminder sent, appointment in 2 days | 0.16 | No |
| Returning patient with a history of no-shows, no reminder | 0.91 | Yes |
| Brand-new patient, 45-day wait, no reminder, evening slot | 0.94 | Yes |

The model separates low- and high-risk patients cleanly, and the ranking matches what the
feature coefficients above would predict.
