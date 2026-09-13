"""
Basic exploratory data analysis for the appointments dataset.

Run: python explore_data.py
Prints summary stats to the console and saves a couple of plots to reports/.
"""

import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "appointments.csv")
REPORTS_DIR = "reports"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("DTYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isna().sum())

    print("\n" + "=" * 60)
    print("NUMERIC SUMMARY")
    print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION (no_show)")
    print("=" * 60)
    print(df["no_show"].value_counts())
    print(df["no_show"].value_counts(normalize=True).round(3))

    print("\n" + "=" * 60)
    print("NO-SHOW RATE BY KEY CATEGORICAL FEATURES")
    print("=" * 60)
    for col in ["gender", "appointment_type", "weekday", "appointment_time", "reminder_sent", "new_patient"]:
        print(f"\n-- {col} --")
        print(df.groupby(col)["no_show"].mean().round(3).sort_values(ascending=False))

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # target balance
    fig, ax = plt.subplots()
    df["no_show"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xticklabels(["Attended (0)", "No-show (1)"], rotation=0)
    ax.set_title("Target distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "target_distribution.png"))

    # no-show rate vs. lead time (bucketed)
    bins = [0, 3, 7, 14, 30, 61]
    labels = ["0-3", "4-7", "8-14", "15-30", "31-60"]
    df["lead_time_bucket"] = pd.cut(df["days_before_appointment"], bins=bins, labels=labels, right=True, include_lowest=True)
    fig, ax = plt.subplots()
    df.groupby("lead_time_bucket", observed=True)["no_show"].mean().plot(kind="bar", ax=ax)
    ax.set_title("No-show rate by lead time (days before appointment)")
    ax.set_ylabel("no-show rate")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "no_show_rate_by_lead_time.png"))

    print(f"\nSaved plots to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
