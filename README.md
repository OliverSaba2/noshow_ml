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
- [ ] Model selection
- [ ] Model training
- [ ] Evaluation
- [ ] Example predictions
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
