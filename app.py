import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px

# Load model & scaler
try:
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.warning("Model files missing. Run model.py first.")

# Page config
st.set_page_config(page_title="PCOD Risk App", layout="wide")
st.title("🩺 PCOD Risk Detection Dashboard")

# -----------------------
# User Input Form
# -----------------------
with st.form("user_form"):
    st.header("Enter Your Details (Manual)")

    age = st.number_input("Age", min_value=12, max_value=60, value=25)
    weight = st.number_input("Weight (kg)", 30.0, 120.0, 60.0)
    height = st.number_input("Height (cm)", 130.0, 200.0, 160.0)
    cycle_length = st.number_input("Cycle Length (days)", 20, 60, 30)
    last_period = st.date_input("Last Period Start Date", datetime.today())
    menstrual_history = st.number_input("Average cycle length last 3-6 cycles", 20, 60, 30)
    fertility_issues = st.selectbox("Fertility Issues (if married)", [0,1])

    st.subheader("Symptoms (1 = Yes, 0 = No)")
    acne = st.selectbox("Acne", [0,1])
    hair_growth = st.selectbox("Excess Hair Growth", [0,1])
    weight_gain = st.selectbox("Weight Gain", [0,1])
    lifestyle_sedentary = st.selectbox("Sedentary Lifestyle", [0,1])
    family_history = st.selectbox("Family History of Hormone Issues", [0,1])

    st.subheader("Wearable Device Inputs (Automatic)")
    heart_rate = st.number_input("Average Heart Rate (bpm)", 50, 120, 75)
    body_temp = st.number_input("Body Temperature (°C)", 35.0, 38.0, 36.6)
    sleep_hours = st.number_input("Sleep Hours per night", 3, 12, 7)
    stress_hrv = st.number_input("Stress (HRV-based)", 20, 100, 50)

    submit_button = st.form_submit_button("Predict Risk")

# -----------------------
# Prediction
# -----------------------
if submit_button:
    # Auto BMI
    bmi = round(weight / (height/100)**2,1)
    irregularity = 1 if abs(cycle_length - menstrual_history) > 7 else 0

    input_df = pd.DataFrame({
        "Age":[age],
        "BMI":[bmi],
        "CycleLength":[cycle_length],
        "Irregular":[irregularity],
        "Symptoms_Acne":[acne],
        "Symptoms_HairGrowth":[hair_growth],
        "Symptoms_WeightGain":[weight_gain],
        "Lifestyle_Sedentary":[lifestyle_sedentary],
        "Hormone_FamilyHistory":[family_history],
        "FertilityIssues":[fertility_issues],
        "HR":[heart_rate],
        "BodyTemp":[body_temp],
        "SleepHours":[sleep_hours],
        "Stress_HRV":[stress_hrv]
    })

    try:
        numeric_features = ["Age","BMI","CycleLength","HR","BodyTemp","SleepHours","Stress_HRV"]
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Your predicted PCOD Risk is: **{prediction}**")
    except:
        st.error("Prediction error. Check model files.")

# -----------------------
# Dashboard / Phase-wise HR
# -----------------------
st.header("📊 Dashboard (Simulated Phase-wise HR)")

# Calculate phase-wise HR from average HR & cycle info
phase_offsets = {"Menstrual":-3,"Follicular":-1,"Ovulatory":+2,"Luteal":+4}
hr_data = pd.DataFrame({
    "Phase": list(phase_offsets.keys()),
    "HeartRate": [heart_rate + v for v in phase_offsets.values()]
})
fig_hr = px.bar(hr_data, x="Phase", y="HeartRate", title="Estimated Phase-wise Heart Rate")
st.plotly_chart(fig_hr, use_container_width=True)

# Symptoms summary
symptoms = pd.DataFrame({
    "Symptom":["Acne","Hair Growth","Weight Gain"],
    "Presence":[acne,hair_growth,weight_gain]
})
fig_sym = px.bar(symptoms, x="Symptom", y="Presence", title="Symptoms Summary", range_y=[0,1])
st.plotly_chart(fig_sym, use_container_width=True)

# Sleep trend (simulated)
sleep_data = pd.DataFrame({
    "Day": np.arange(1,8),
    "SleepHours": np.random.randint(5,9, size=7)
})
fig_sleep = px.line(sleep_data, x="Day", y="SleepHours", title="Sleep Hours Trend")
st.plotly_chart(fig_sleep, use_container_width=True)

# -----------------------
# Lifestyle Tips
# -----------------------
if submit_button:
    st.header("💡 Lifestyle Suggestions")
    if prediction=="High":
        st.info("""
        - Consult a doctor for proper diagnosis  
        - Maintain healthy diet & exercise  
        - Track cycle & symptoms daily  
        - Reduce stress and ensure proper sleep
        """)
    elif prediction=="Medium":
        st.info("""
        - Monitor symptoms & cycle regularly  
        - Maintain balanced diet & activity  
        - Track trends using this app
        """)
    else:
        st.success("Low risk. Maintain healthy lifestyle!")
