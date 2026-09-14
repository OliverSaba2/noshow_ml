# How this project works

A walkthrough of what was built and why, in plain terms. For the actual numbers, see
[RESULTS.md](RESULTS.md).

## The problem

Clinics lose money and time when patients book an appointment and just don't show up.
The goal here is a model that flags, ahead of time, which appointments are at risk — so
staff can send a reminder, make a confirmation call, or plan around it.

## 1. Getting data

The assessment describes a dataset (age, appointment type, reminders, no-show history,
etc.) but doesn't actually attach one. So this project generates a synthetic dataset with
those same columns instead (`data/appointments.csv`, 3000 rows). It's built so the target
depends on the features in a realistic way — reminders lower risk, a longer wait before
the appointment raises it, and so on — without being trivially easy to predict. Drop in a
real dataset with the same column names and everything else still works.

## 2. Looking at the data

Before building anything, `explore_data.py` checks the basics: how big is the dataset, is
anything missing, how imbalanced is the target (about 30% of appointments are no-shows),
and which features already look related to no-shows just by eyeballing group averages.
This is what pointed at reminders and lead time as promising signals early on.

## 3. Cleaning and preparing

`prepare_data.py` does the unglamorous but necessary part: filling in anything missing,
fixing an impossible value (a patient can't have more no-shows than appointments), and
adding one new feature — a patient's personal no-show *rate*, which turned out to be more
useful than the raw counts. It then splits the data into a training set and a test set,
keeping the same no-show ratio in both, so the test set is a fair, untouched check later.

Turning categories like "weekday" into numbers, and scaling numeric columns, is deliberately
*not* done at this stage — it's bundled with the model itself in the next step, so it's
always refit correctly and never accidentally "peeks" at data it shouldn't.

## 4. Trying a few models

Rather than guessing, `model_selection.py` tries four different model types — logistic
regression, a decision tree, a random forest, and gradient boosting — and tests each one
fairly using cross-validation (repeatedly training on most of the data and checking against
a held-out slice, several times over). Logistic regression came out ahead, and it also has
a nice side benefit: it's easy to explain *why* it made a prediction, which matters for a
tool clinic staff would actually use and trust.

## 5. Training the final model

`train_model.py` takes the winning model type and trains it one more time, this time on
*all* of the training data (cross-validation only ever used most of it at a time), and
saves the result so it doesn't need to be retrained every time it's used.

## 6. Checking it actually works

`evaluate_model.py` runs the trained model against the test set — data it has never seen
in any form — and reports how well it does. It also shows which factors the model leans on
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

The trained model is small and self-contained — it would sit behind a simple API endpoint
that a scheduling system calls when an appointment is booked (and again as it gets closer,
since risk changes over time). The score itself isn't the end goal — it's what triggers an
action, like an extra reminder for high-risk bookings. And since no-show behavior can shift
over time, this whole pipeline (prepare → select → train → evaluate) is built to be re-run
periodically on fresh data, not just once.
