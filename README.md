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

## Columns

| Column | Meaning |
|---|---|
| appointment_id | Row identifier (not used as a model feature) |
| age | Patient age |
| gender | Female / Male |
| appointment_type | Follow-up / New Consultation / Procedure / Routine Check / Urgent Visit |
| days_before_appointment | Lead time between booking and the appointment |
| previous_appointments | Count of the patient's past appointments |
| previous_no_shows | Count of the patient's past no-shows |
| weekday | Day of the appointment |
| appointment_time | One of six 2-hour slots, e.g. `08:00-10:00` |
| reminder_sent | Whether a reminder was sent (0/1) |
| new_patient | Whether this is the patient's first appointment (0/1) |
| no_show | Target: 1 = missed appointment, 0 = attended |
