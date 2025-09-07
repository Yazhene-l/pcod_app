# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -----------------------
# Load Model and Scaler
# -----------------------
try:
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.warning("Model files not found. Please train the model first.")

# -----------------------
# App Layout
# -----------------------
st.set_page_config(page_title="PCOD Risk App", layout="wide")
st.title("🩺 PCOD Risk Detection Dashboard")

# -----------------------
# User Input Section
# -----------------------
st.header("Enter Your Details")
with st.form(key='user_form'):
    age = st.number_input("Age", min_value=12, max_value=60, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    cycle_length = st.number_input("Cycle Length (days)", min_value=20, max_value=60, value=30)
    sleep_hours = st.number_input("Sleep Hours per night", min_value=3, max_value=12, value=7)
    heart_rate = st.number_input("Average Heart Rate (bpm)", min_value=50, max_value=120, value=75)
    
    # Symptoms
    st.subheader("Symptoms (1 = Yes, 0 = No)")
    acne = st.selectbox("Acne", [0,1])
    hair_growth = st.selectbox("Excess Hair Growth", [0,1])
    weight_gain = st.selectbox("Weight Gain", [0,1])
    
    submit_button = st.form_submit_button(label="Predict Risk")

# -----------------------
# Prediction
# -----------------------
if submit_button:
    input_df = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "HeartRate": [heart_rate],
        "SleepHours": [sleep_hours],
        "CycleLength": [cycle_length],
        "Acne": [acne],
        "HairGrowth": [hair_growth],
        "WeightGain": [weight_gain]
    })

    try:
        # Scale numeric features
        numeric_features = ["Age", "BMI", "HeartRate", "SleepHours", "CycleLength"]
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"Your predicted PCOD Risk is: **{prediction}**")
    except:
        st.error("Error predicting. Make sure the model files exist and are correct.")

# -----------------------
# Simulated Dashboard Section
# -----------------------
st.header("📊 Dashboard (Simulated Data)")

# Example Heart Rate trend over 7 days
hr_data = pd.DataFrame({
    "Day": np.arange(1,8),
    "HeartRate": np.random.randint(65, 95, size=7)
})
fig_hr = px.line(hr_data, x="Day", y="HeartRate", title="Heart Rate Trend Over 7 Days")
st.plotly_chart(fig_hr, use_container_width=True)

# Symptoms Summary
symptoms = pd.DataFrame({
    "Symptom": ["Acne", "Excess Hair Growth", "Weight Gain"],
    "Presence": [acne, hair_growth, weight_gain]
})
fig_sym = px.bar(symptoms, x="Symptom", y="Presence", title="Symptoms Summary", range_y=[0,1])
st.plotly_chart(fig_sym, use_container_width=True)

# Sleep quality (simulated)
sleep_data = pd.DataFrame({
    "Day": np.arange(1,8),
    "SleepHours": np.random.randint(5,9, size=7)
})
fig_sleep = px.line(sleep_data, x="Day", y="SleepHours", title="Sleep Hours Trend")
st.plotly_chart(fig_sleep, use_container_width=True)

# -----------------------
# Tips / Suggestions
# -----------------------
if submit_button:
    st.header("💡 Lifestyle Suggestions")
    if prediction == "High":
        st.info("""
        - Consult a doctor for proper diagnosis  
        - Maintain a healthy diet and exercise regularly  
        - Track your cycle and symptoms daily  
        - Reduce stress and ensure proper sleep
        """)
    elif prediction == "Medium":
        st.info("""
        - Monitor your symptoms and cycle  
        - Maintain a balanced diet and exercise  
        - Use this app to track trends regularly
        """)
    else:
        st.success("Your risk is low. Keep maintaining a healthy lifestyle!")

