# noshow_ml

Predicts whether a patient is likely to miss (no-show) an upcoming clinic appointment.

Built for the CliniKit AI Trainee Assessment, Part 2 — Machine Learning.

## Approach at a glance

1. **No real dataset was attached to the assessment**, so `data/appointments.csv` is a
   synthetically generated stand-in with the exact columns described in the brief, and a
   realistic (not perfectly separable) relationship between the features and `no_show`.
   Drop in a real dataset with the same column names and every step below keeps working.
2. Explored the data, cleaned it, and engineered one extra feature (`previous_no_show_rate`).
3. Compared 4 candidate models with 5-fold cross-validation and picked by ROC-AUC.
4. **Logistic regression** won — it's also the most interpretable option, which matters
   for a clinic tool (see [Step 3](#step-3--model-selection)).
5. Trained it on the full training set, evaluated on a held-out test set
   (ROC-AUC 0.726, recall 0.689 — see [Step 5](#step-5--evaluation)), and used it to score
   example patients (see [Step 6](#step-6--example-predictions)).
6. See [Integrating this into a real product](#integrating-this-into-a-real-product) for
   how this would plug into clinic software.

Run everything end-to-end with:

```bash
pip install -r requirements.txt
python prepare_data.py
python model_selection.py
python train_model.py
python evaluate_model.py
python predict_examples.py
```

(`data/appointments.csv` is already committed, so `explore_data.py` can also be run any
time on its own.)

For a quick, visual look at the whole project instead of the CLI scripts, run:

```bash
streamlit run app.py
```

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

## Progress

- [x] Data generation + basic exploration (`explore_data.py`)
- [x] Data preparation (`prepare_data.py`, `preprocessing.py`)
- [x] Model selection (`model_selection.py`)
- [x] Model training (`train_model.py`)
- [x] Evaluation (`evaluate_model.py`)
- [x] Example predictions (`predict_examples.py`)
- [x] Full write-up of approach
- [x] Streamlit UI (`app.py`)

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

## Streamlit app

```bash
streamlit run app.py
```

A small UI over everything above, for browsing the project without touching the CLI:

- **Predict** — form to score a single patient, with a risk level (low/medium/high) and
  probability.
- **Model insights** — live test-set metrics, plus the confusion matrix, ROC curve, and
  feature-importance plots from Step 5.
- **Dataset** — headline stats and an interactive no-show-rate breakdown by any feature.

## Project structure

```
noshow_ml/
├── data/
│   ├── appointments.csv       # raw dataset (synthetic — see note above)
│   ├── train.csv, test.csv    # stratified split, written by prepare_data.py
│   ├── feature_config.json    # single source of truth for feature lists
│   └── selected_model.json    # which model won selection, written by model_selection.py
├── models/
│   └── model.joblib           # fitted preprocessing + model pipeline
├── reports/                   # plots and CSVs from exploration/selection/evaluation
├── explore_data.py            # Step 1
├── prepare_data.py            # Step 2
├── preprocessing.py           # shared ColumnTransformer, used by every step below
├── models.py                  # shared candidate model definitions
├── model_selection.py         # Step 3
├── train_model.py             # Step 4
├── evaluate_model.py          # Step 5
├── predict_examples.py        # Step 6
└── app.py                     # Streamlit UI
```

## Integrating this into a real product

The model itself is just `models/model.joblib` — an sklearn pipeline that takes a
DataFrame with the 10 raw columns and returns a no-show probability. Wiring it into a
real clinic system would look like:

- **Serve it behind an API**, not embedded in the app. Wrap `pipeline.predict_proba` in a
  small FastAPI/Flask service with one endpoint (e.g. `POST /predict` taking one or many
  appointment records, returning a probability per record — `predict_examples.py`'s
  `predict()` function is already shaped for this). This keeps the ML runtime decoupled
  from the scheduling app, deployable and scaled independently, and swappable without
  touching the app's code.
- **Score at booking time and again as the appointment approaches.** A no-show risk isn't
  static — `days_before_appointment` shrinks and `reminder_sent` flips as the date gets
  closer, both of which move the prediction. Re-score once when the appointment is booked
  and again shortly before it (e.g. nightly batch job or on reminder-send), and surface the
  risk level on the staff-facing schedule (e.g. a colored badge) rather than just a raw number.
- **Use the score to trigger action, not just display it**: e.g. auto-prioritize a second
  reminder or a confirmation call for high-risk bookings, or allow controlled overbooking
  in slots with a high predicted no-show rate — exactly the kind of decision the top
  coefficients in `reports/feature_importance.png` support (long lead time, no reminder yet,
  new patient, evening slot).
- **Retrain on a schedule, not once.** No-show behavior drifts (season, patient population,
  clinic policy changes). Keep `prepare_data.py` → `model_selection.py` → `train_model.py`
  → `evaluate_model.py` as a pipeline that can be re-run periodically (e.g. monthly) against
  fresh data, and gate deploying a new `model.joblib` on it beating the currently deployed
  model's test metrics — reusing `evaluate_model.py` as that gate.
- **Log predictions vs. outcomes** (predicted probability alongside what actually happened)
  so evaluation metrics can be recomputed on real production data over time, not just the
  original test split — that's what would catch drift early.
- **Treat the model as one input, not a gatekeeper.** False positives here just mean an
  extra reminder; false negatives mean a missed slot. Staff should be able to see *why* a
  patient was flagged (the feature values that drove it, since logistic regression makes
  this cheap to compute) and override the system, especially early on while trust is built.
