# How this project works

A walkthrough of what was built and why, in plain terms. For the actual numbers, see
[RESULTS.md](RESULTS.md).

## The problem

Clinics lose money and time when patients book an appointment and just don't show up.
The goal here is a model that flags, ahead of time, which appointments are at risk.

## 1. Getting data

`data/appointments.csv` (3000 rows) has the columns seen in the Assessment, plus a row
identifier that's carried along but never used as a feature. The pipeline only depends on
the column names, so swapping in a different dataset with the same columns still works
without changing any code.

## 2. Looking at the data

Before building anything, `explore_data.py` checks the basics: how big is the dataset, is
anything missing, how imbalanced is the target (about 16% of appointments are no-shows) and which features already look related to
no-shows just by eyeballing group averages. Reminders stood out early (no-show rate roughly
1.5x higher without one); most other raw group differences were small, which is what
pushed the modeling step past logistic regression's raw coefficients and toward looking at
adjusted effects instead.

## 3. Cleaning and preparing

`prepare_data.py`: filling in anything missing,
fixing an impossible value (a patient can't have more no-shows than appointments), and
adding one new feature (a patient's personal no-show *rate*, alongside the raw counts -
on this dataset the raw count of past no-shows ended up mattering more to the model, but
the rate is cheap to compute and kept in case that changes on other data). It then splits
the data into a training set and a test set, keeping the same no-show ratio in both, so
the test set is a fair, untouched check later.

Turning categories like "weekday" into numbers, and scaling numeric columns, is deliberately
*not* done at this stage (it's bundled with the model itself in the next step, so it's
always refit correctly and never accidentally "peeks" at data it shouldn't).

## 4. Trying a few models

Rather than guessing, `model_selection.py` tries four different model types: logistic
regression, a decision tree, a random forest, and gradient boosting. Tests each one
fairly using cross-validation (repeatedly training on most of the data and checking against
a held-out slice, several times over). Logistic regression came out ahead, and it also has
a nice side benefit: it's easy to explain *why* it made a prediction, which matters for a
tool clinic staff would actually use and trust.

## 5. Training the final model

`train_model.py` takes the winning model type and trains it one more time, this time on
*all* of the training data (cross-validation only ever used most of it at a time), and
saves the result so it doesn't need to be retrained every time it's used.

## 6. Checking it actually works

`evaluate_model.py` runs the trained model against the test set: data it has never seen
in any form. Reports how well it does. It also shows which factors the model leans on
most heavily, which roughly matches what the early data exploration suggested.

## 7. Trying it on real examples

`predict_examples.py` runs a handful of made-up patients through the model — a reliable
regular with a reminder set, a brand-new patient booked weeks out with no reminder, and so
on — as a sanity check that the model's risk scores make sense, not just that its metrics
look good on paper.

## 8. A small UI

`app.py` (a Streamlit app) wraps all of this in a simple interface: a form to score one
patient, a tab showing how well the model performs, and a tab for poking around the
dataset. Not required, but it's a much easier way to actually *look* at the project than
reading through script output.

## If this were a real product

The trained model is small and self-contained, it would sit behind a simple API endpoint
that a scheduling system calls when an appointment is booked (and again as it gets closer,
since risk changes over time). The score itself isn't the end goal, it's what triggers an
action, like an extra reminder for high-risk bookings. And since no-show behavior can shift
over time, this whole pipeline (prepare → select → train → evaluate) is built to be re-run
periodically on fresh data, not just once.
