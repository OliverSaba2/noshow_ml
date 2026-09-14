"""
Small Streamlit app for the no-show model: score a patient, see how the
model performs, and browse the dataset it was trained on.

Run: streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

from prepare_data import clean_and_engineer
from preprocessing import load_feature_config
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="No-Show Predictor", page_icon="🩺", layout="centered")


@st.cache_resource
def get_model():
    return joblib.load("models/model.joblib")


@st.cache_data
def get_feature_config():
    return load_feature_config()


@st.cache_data
def get_dataset():
    return pd.read_csv("data/appointments.csv")


@st.cache_data
def get_test_set():
    return pd.read_csv("data/test.csv")


model = get_model()
feature_config = get_feature_config()
feature_cols = feature_config["numeric_features"] + feature_config["categorical_features"]
df = get_dataset()

st.title("🩺 Appointment No-Show Predictor")
st.caption("CliniKit ML assessment — predicts whether a patient will miss an upcoming appointment.")

tab_predict, tab_insights, tab_data = st.tabs(["Predict", "Model insights", "Dataset"])

# ---------------------------------------------------------------- Predict --
with tab_predict:
    st.subheader("Score a patient")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 1, 90, 40)
        gender = st.selectbox("Gender", sorted(df["gender"].unique()))
        appointment_type = st.selectbox("Appointment type", sorted(df["appointment_type"].unique()))
        weekday = st.selectbox("Weekday", sorted(df["weekday"].unique()))
        appointment_time = st.selectbox("Appointment time", sorted(df["appointment_time"].unique()))
    with col2:
        days_before_appointment = st.slider("Days before appointment", 0, 60, 7)
        new_patient = st.checkbox("New patient", value=False)
        previous_appointments = st.number_input(
            "Previous appointments", min_value=0, max_value=50, value=3, disabled=new_patient
        )
        previous_no_shows = st.number_input(
            "Previous no-shows", min_value=0, max_value=int(previous_appointments), value=0,
            disabled=new_patient,
        )
        reminder_sent = st.checkbox("Reminder sent", value=True)

    if st.button("Predict no-show risk", type="primary"):
        row = pd.DataFrame(
            [
                dict(
                    age=age,
                    gender=gender,
                    appointment_type=appointment_type,
                    days_before_appointment=days_before_appointment,
                    previous_appointments=0 if new_patient else int(previous_appointments),
                    previous_no_shows=0 if new_patient else int(previous_no_shows),
                    weekday=weekday,
                    appointment_time=appointment_time,
                    reminder_sent=int(reminder_sent),
                    new_patient=int(new_patient),
                )
            ]
        )
        prepared = clean_and_engineer(row)
        proba = model.predict_proba(prepared[feature_cols])[0, 1]

        st.metric("Predicted no-show probability", f"{proba:.0%}")
        st.progress(min(proba, 1.0))
        if proba >= 0.6:
            st.error("High risk — consider a confirmation call or an extra reminder.")
        elif proba >= 0.3:
            st.warning("Medium risk.")
        else:
            st.success("Low risk.")

# --------------------------------------------------------- Model insights --
with tab_insights:
    st.subheader("Performance on the held-out test set")

    test_df = get_test_set()
    y_true = test_df[feature_config["target"]]
    y_proba = model.predict_proba(test_df[feature_cols])[:, 1]
    y_pred = model.predict(test_df[feature_cols])

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.2f}")

    st.caption(
        "Recall is weighted more heavily than precision when comparing models: a missed "
        "no-show (false negative) loses the clinic a slot, while a false alarm just means "
        "an extra reminder."
    )

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image("reports/confusion_matrix.png", width='stretch')
    with img_col2:
        st.image("reports/roc_curve.png", width='stretch')

    st.subheader("What drives the prediction")
    st.image("reports/feature_importance.png", width='stretch')

# ----------------------------------------------------------------- Dataset -
with tab_data:
    st.subheader("Dataset overview")
    st.caption(
        "No dataset was attached to the assessment, so this is a synthetically generated "
        "stand-in with a realistic (not perfectly separable) no-show pattern."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("No-show rate", f"{df['no_show'].mean():.0%}")
    c3.metric("Features", len(feature_cols))

    st.markdown("**No-show rate by feature**")
    factor = st.selectbox(
        "Break down by",
        ["reminder_sent", "new_patient", "appointment_type", "weekday", "appointment_time", "gender"],
    )
    st.bar_chart(df.groupby(factor)["no_show"].mean())

    with st.expander("Raw data sample"):
        st.dataframe(df.sample(20, random_state=1))
