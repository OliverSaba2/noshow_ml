"""
Generates a synthetic patient-appointment dataset for the no-show prediction task.

No real clinic dataset was provided with the assessment, so this script creates
a realistic stand-in with the exact columns described in the task brief. The
no_show target is generated from a logistic combination of the features below,
so the resulting dataset has genuine (but not too obvious) learnable structure:

- Longer lead time (days_before_appointment) increases no-show risk.
- A higher personal history of no-shows increases risk.
- Reminders being sent lowers risk.
- New patients are somewhat more likely to miss their appointment.
- Younger adults skew slightly more likely to miss appointments.

Run: python generate_data.py
Output: data/appointments.csv
"""

import numpy as np
import pandas as pd
import os

RNG_SEED = 42
N_ROWS = 3000

GENDERS = ["M", "F"]
APPOINTMENT_TYPES = ["general", "specialist", "dental", "vaccination", "follow_up"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
APPOINTMENT_TIMES = ["morning", "afternoon", "evening"]


def generate_dataset(n_rows: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(1, 90, size=n_rows)
    gender = rng.choice(GENDERS, size=n_rows)
    appointment_type = rng.choice(
        APPOINTMENT_TYPES, size=n_rows, p=[0.35, 0.2, 0.15, 0.15, 0.15]
    )
    days_before_appointment = rng.integers(0, 61, size=n_rows)
    weekday = rng.choice(WEEKDAYS, size=n_rows)
    appointment_time = rng.choice(APPOINTMENT_TIMES, size=n_rows, p=[0.45, 0.4, 0.15])
    reminder_sent = rng.choice([0, 1], size=n_rows, p=[0.3, 0.7])

    previous_appointments = rng.poisson(lam=3, size=n_rows)
    new_patient = (previous_appointments == 0).astype(int)

    # previous_no_shows can't exceed previous_appointments
    no_show_rate_per_patient = rng.beta(2, 6, size=n_rows)  # skewed low
    previous_no_shows = np.floor(previous_appointments * no_show_rate_per_patient).astype(int)

    # --- build no_show probability from a logistic model of the features ---
    history_no_show_rate = np.where(
        previous_appointments > 0, previous_no_shows / np.maximum(previous_appointments, 1), 0.15
    )

    z = (
        -1.3
        + 0.03 * days_before_appointment
        + 2.4 * history_no_show_rate
        + 0.55 * new_patient
        - 0.9 * reminder_sent
        - 0.015 * (age - 35)  # younger -> slightly higher risk
        + np.where(appointment_time == "morning", -0.25, 0.0)
        + np.where(np.isin(weekday, ["Monday", "Friday"]), 0.15, 0.0)
    )
    prob_no_show = 1 / (1 + np.exp(-z))
    no_show = rng.binomial(1, prob_no_show)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "appointment_type": appointment_type,
            "days_before_appointment": days_before_appointment,
            "previous_appointments": previous_appointments,
            "previous_no_shows": previous_no_shows,
            "weekday": weekday,
            "appointment_time": appointment_time,
            "reminder_sent": reminder_sent,
            "new_patient": new_patient,
            "no_show": no_show,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "appointments.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"no_show rate: {df['no_show'].mean():.3f}")
