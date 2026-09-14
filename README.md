# noshow_ml

Predicts whether a patient is likely to miss (no-show) an upcoming clinic appointment.

Built for the CliniKit AI Trainee Assessment, Part 2 — Machine Learning.

## Note on the data

The assessment describes a dataset but doesn't attach one, so `generate_data.py`
creates a synthetic dataset with the same columns and a realistic (not perfectly
separable) relationship between features and the `no_show` target. Swap in a
real dataset with the same column names and everything downstream keeps working.

## Columns

| Column | Meaning |
|---|---|
| age | Patient age |
| gender | M / F |
| appointment_type | general / specialist / dental / vaccination / follow_up |
| days_before_appointment | Lead time between booking and the appointment |
| previous_appointments | Count of the patient's past appointments |
| previous_no_shows | Count of the patient's past no-shows |
| weekday | Day of the appointment |
| appointment_time | morning / afternoon / evening |
| reminder_sent | Whether a reminder was sent (0/1) |
| new_patient | Whether this is the patient's first appointment (0/1) |
| no_show | Target: 1 = missed appointment, 0 = attended |

## Setup

```bash
pip install -r requirements.txt
```

## Progress

- [x] Data generation + basic exploration (`explore_data.py`)
- [x] Data preparation (`prepare_data.py`, `preprocessing.py`)
- [x] Model selection (`model_selection.py`)
- [x] Model training (`train_model.py`)
- [x] Evaluation (`evaluate_model.py`)
- [x] Example predictions (`predict_examples.py`)
- [ ] Full write-up of approach

## Step 1 — Data exploration

```bash
python explore_data.py    # prints EDA to console, saves plots to reports/
```

Findings from the synthetic data:
- Target is moderately imbalanced: ~30% no-show, ~70% attended.
- No missing values (synthetic data is clean; a real dataset should be checked again).
- Clear no-show signal from `reminder_sent` (41% no-show without a reminder vs 26% with one),
  `new_patient` (48% vs 29% for returning patients), and longer `days_before_appointment`.

## Step 2 — Data preparation

```bash
python prepare_data.py    # writes data/train.csv, data/test.csv, data/feature_config.json
```

What it does:
- Basic cleaning: fills any missing numeric/categorical values, clips `previous_no_shows`
  so it can never exceed `previous_appointments`.
- Feature engineering: adds `previous_no_show_rate` (a patient's personal history of
  missed appointments), which is a stronger, more directly comparable signal than the
  raw `previous_no_shows` / `previous_appointments` counts.
- Stratified 80/20 train/test split on `no_show`, so both splits keep the same class balance.
- Writes `data/feature_config.json` as the single source of truth for which columns are
  numeric vs. categorical, so every later script (model selection, training, prediction)
  reads the same definition instead of redefining it.

Encoding/scaling is intentionally **not** done in this step. `preprocessing.py` builds a
`ColumnTransformer` (`StandardScaler` for numeric features, `OneHotEncoder` for categorical
ones) that gets attached to the model itself as an sklearn `Pipeline` in the next step. This
matters: if we scaled/encoded once on the whole training set before cross-validation, each
CV fold would see statistics computed from data it's not supposed to know about yet (a subtle
form of data leakage). Fitting the transformer inside the pipeline avoids that.

## Step 3 — Model selection

```bash
python model_selection.py   # writes reports/model_selection_results.csv, data/selected_model.json
```

Compares four candidates with 5-fold stratified cross-validation on the training set only:
logistic regression, decision tree, random forest, gradient boosting. Each is wrapped in the
same `preprocessing.py` pipeline so scaling/encoding is refit per fold.

Why these metrics: `no_show` is imbalanced (~30% positive), so accuracy alone is misleading —
a model that always predicts "attended" scores ~70% accuracy while being useless. The model
is selected by mean **ROC-AUC** (ranking quality across all thresholds, useful since a clinic
can pick its own risk cutoff), with precision/recall/F1 also recorded so the trade-off is visible.

Result: **logistic regression** had the best ROC-AUC (~0.72) and F1 (~0.53). This isn't
surprising — the synthetic target was generated from a logistic combination of the features,
so a linear-in-log-odds model is a natural fit. It also has a practical edge for this use case:
its coefficients directly show which factors raise or lower no-show risk, which is easy to
surface to clinic staff.

Candidate hyperparameters live in `models.py`, shared by `model_selection.py` and
`train_model.py` so they can't drift out of sync between the comparison and the final fit.

## Step 4 — Model training

```bash
python train_model.py   # writes models/model.joblib
```

Reads `data/selected_model.json` to pick up the winner from Step 3, builds the same
preprocessing + model pipeline, and fits it on the **entire** training set (cross-validation
in Step 3 only ever trains on 4/5 of it per fold — the final model should use all of it). The
fitted pipeline (preprocessing + logistic regression together) is saved to `models/model.joblib`
so evaluation and prediction never have to re-fit or re-derive the preprocessing.

## Step 5 — Evaluation

```bash
python evaluate_model.py
```

Evaluated on the **test set only** (the 20% held out in Step 2, never touched during
selection or training). Results:

| Metric | Value |
|---|---|
| Accuracy | 0.685 |
| Precision (no_show) | 0.488 |
| Recall (no_show) | 0.689 |
| F1 (no_show) | 0.571 |
| ROC-AUC | 0.726 |

These line up closely with the 5-fold CV numbers from Step 3, which is the sign to look
for: it means the model generalizes and Step 3 wasn't overfit to a lucky CV split.

Why recall matters more than precision here: missing a real no-show (false negative) means
the clinic does nothing and loses the slot; flagging an attendee as at-risk (false positive)
just means an extra reminder or a slightly cautious overbook. That asymmetric cost is why
recall (0.69) was prioritized over precision (0.49) when comparing candidates — a model tuned
purely for accuracy would under-predict the minority `no_show` class instead.

`reports/confusion_matrix.png` and `reports/roc_curve.png` visualize this. `reports/feature_importance.png`
(and the underlying `.csv`) show the logistic regression coefficients — the biggest drivers of
predicted no-show risk are, in order: longer `days_before_appointment`, younger `age`, no
`reminder_sent`, and `appointment_time` (evening riskier than morning). This matches the EDA
in Step 1 and gives the clinic concrete, actionable levers (e.g. send reminders, prioritize
follow-up calls for long-lead-time bookings).

## Step 6 — Example predictions

```bash
python predict_examples.py                                   # scores 4 built-in example patients
python predict_examples.py --input new.csv --output preds.csv # scores your own patients, saves a CSV
```

`--input` accepts a CSV with the same raw columns as `data/appointments.csv` (no `no_show`
column needed). It reuses `clean_and_engineer` from `prepare_data.py`, so the exact same
feature engineering used in training runs on new data — nothing is duplicated or can drift
out of sync.

Example output:

| patient | no_show_probability | predicted_no_show |
|---|---|---|
| Returning patient, reminder sent, morning slot | 0.116 | 0 |
| New patient, no reminder, 45-day lead time, evening slot | 0.942 | 1 |
| Returning patient, history of no-shows, no reminder | 0.908 | 1 |
| Returning patient, reminder sent, appointment in 2 days | 0.164 | 0 |

The model separates these cleanly, and the ranking matches what the feature coefficients
predict it should.
