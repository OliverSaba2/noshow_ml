# noshow_ml

Predicts whether a patient is likely to miss (no-show) an upcoming clinic appointment.

Built for the CliniKit AI Trainee Assessment, Part 2 — Machine Learning.

- **What/why/how it works:** see [EXPLANATION.md](EXPLANATION.md)
- **Numbers/metrics/findings:** see [RESULTS.md](RESULTS.md)

## Quickstart

```bash
pip install -r requirements.txt

python prepare_data.py      # clean + split the data
python model_selection.py   # compare candidate models
python train_model.py       # train the chosen model
python evaluate_model.py    # metrics on the test set
python predict_examples.py  # try it on a few example patients
```

Or skip the CLI and browse everything in a small UI:

```bash
streamlit run app.py
```

`data/appointments.csv` is a synthetic dataset (no real one was attached to the
assessment) — see [EXPLANATION.md](EXPLANATION.md) for why, and drop in a real
dataset with the same columns to use it instead.

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
