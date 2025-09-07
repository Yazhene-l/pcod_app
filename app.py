# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
    
    weight = st.number_input("Weight (kg)", min_value=30, max_value=150, value=60)
    height = st.number_input("Height (cm)", min_value=120, max_value=220, value=165)
    bmi = round(weight / ((height/100)**2), 1)
    st.info(f"Calculated BMI: {bmi}")
    
    cycle_length = st.number_input("Cycle Length (days)", min_value=20, max_value=60, value=30)
    menstrual_irregularity = st.slider("Menstrual Irregularity (0 = None, 5 = Very irregular)", 0, 5, 2)
    sleep_hours = st.number_input("Sleep Hours per night", min_value=3, max_value=12, value=7)
    stress_level = st.slider("Stress Level (0 = None, 5 = Very High)", 0, 5, 2)
    
    # Symptoms
    st.subheader("Symptoms (1 = Yes, 0 = No)")
    acne = st.selectbox("Acne", [0,1])
    hair_growth = st.selectbox("Excess Hair Growth", [0,1])
    weight_gain = st.selectbox("Weight Gain", [0,1])
    
    # Wearable HR input (simulated if no device)
    st.subheader("Average Heart Rate from Wearable (bpm)")
    wearable_data_available = st.checkbox("I have wearable HR data", value=False)
    
    if wearable_data_available:
        hr_menstrual = st.number_input("Menstrual Phase HR", min_value=50, max_value=120, value=70)
        hr_follicular = st.number_input("Follicular Phase HR", min_value=50, max_value=120, value=72)
        hr_ovulatory = st.number_input("Ovulatory Phase HR", min_value=50, max_value=120, value=75)
        hr_luteal = st.number_input("Luteal Phase HR", min_value=50, max_value=120, value=78)
    else:
        hr_menstrual = np.random.randint(65, 75)
        hr_follicular = np.random.randint(68, 78)
        hr_ovulatory = np.random.randint(70, 85)
        hr_luteal = np.random.randint(72, 88)
        st.info(f"Simulated HR (bpm) - Menstrual: {hr_menstrual}, Follicular: {hr_follicular}, Ovulatory: {hr_ovulatory}, Luteal: {hr_luteal}")

    submit_button = st.form_submit_button(label="Predict Risk")

# -----------------------
# Prediction
# -----------------------
if submit_button:
    input_df = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "CycleLength": [cycle_length],
        "MenstrualIrregularity": [menstrual_irregularity],
        "SleepHours": [sleep_hours],
        "StressLevel": [stress_level],
        "Acne": [acne],
        "HairGrowth": [hair_growth],
        "WeightGain": [weight_gain],
        "HR_Menstrual": [hr_menstrual],
        "HR_Follicular": [hr_follicular],
        "HR_Ovulatory": [hr_ovulatory],
        "HR_Luteal": [hr_luteal]
    })

    try:
        # Scale numeric features
        numeric_features = ["Age", "BMI", "CycleLength", "MenstrualIrregularity",
                            "SleepHours", "StressLevel", "HR_Menstrual", "HR_Follicular",
                            "HR_Ovulatory", "HR_Luteal"]
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

# HR Trend Across Phases
hr_data = pd.DataFrame({
    "Phase": ["Menstrual", "Follicular", "Ovulatory", "Luteal"],
    "HeartRate": [hr_menstrual, hr_follicular, hr_ovulatory, hr_luteal]
})
fig_hr = px.line(hr_data, x="Phase", y="HeartRate", title="Heart Rate Across Menstrual Phases", markers=True)
st.plotly_chart(fig_hr, use_container_width=True)

# Symptoms Summary
symptoms = pd.DataFrame({
    "Symptom": ["Acne", "Excess Hair Growth", "Weight Gain"],
    "Presence": [acne, hair_growth, weight_gain]
})
fig_sym = px.bar(symptoms, x="Symptom", y="Presence", title="Symptoms Summary", range_y=[0,1])
st.plotly_chart(fig_sym, use_container_width=True)

# Sleep trend (simulated)
sleep_data = pd.DataFrame({
    "Day": np.arange(1,8),
    "SleepHours": np.random.randint(5,9, size=7)
})
fig_sleep = px.line(sleep_data, x="Day", y="SleepHours", title="Sleep Hours Trend Over 7 Days", markers=True)
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
