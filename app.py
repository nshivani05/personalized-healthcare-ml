import streamlit as st
import pandas as pd
import numpy as np

import joblib


# Load model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Blood Donor Prediction App", layout="centered")

st.title("🩸 Personalized Blood Donor Recommendation")
st.write("Predict whether a person is likely to donate blood again based on donation history.")

# Input fields
recency = st.number_input("Months since last donation", min_value=0)
frequency = st.number_input("Total number of donations", min_value=0)
monetary = st.number_input("Total blood donated (in cc)", min_value=0)
time = st.number_input("Months since first donation", min_value=0)


if st.button("Predict"):
    input_data = np.array([[recency, frequency, monetary, time]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to Donate Again! (Confidence: {probability:.2f})")
    else:
        st.warning(f"❌ Not Likely to Donate Again. (Confidence: {1 - probability:.2f})")
