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

- [x] Data generation + basic exploration (`generate_data.py`, `explore_data.py`)
- [ ] Data preparation
- [ ] Model selection
- [ ] Model training
- [ ] Evaluation
- [ ] Example predictions
- [ ] Full write-up of approach

## Step 1 — Data exploration

```bash
python generate_data.py   # writes data/appointments.csv
python explore_data.py    # prints EDA to console, saves plots to reports/
```

Findings from the synthetic data:
- Target is moderately imbalanced: ~30% no-show, ~70% attended.
- No missing values (synthetic data is clean; a real dataset should be checked again).
- Clear no-show signal from `reminder_sent` (41% no-show without a reminder vs 26% with one),
  `new_patient` (48% vs 29% for returning patients), and longer `days_before_appointment`.
