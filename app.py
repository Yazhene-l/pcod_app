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
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=120.0, value=60.0)
    height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=210.0, value=165.0)
    
    # BMI calculation
    bmi = weight / ((height_cm/100)**2)
    st.text(f"Calculated BMI: {bmi:.1f}")
    
    cycle_length = st.number_input("Cycle Length (days)", min_value=20, max_value=60, value=30)
    menstrual_irregularity = st.slider("Menstrual Irregularity (0=regular, 5=highly irregular)", 0,5,2)
    sleep_hours = st.number_input("Average Sleep Hours per night", min_value=3, max_value=12, value=7)
    stress_level = st.slider("Stress Level (0=low,5=high)", 0,5,2)
    
    # Symptoms
    st.subheader("Symptoms (1 = Yes, 0 = No)")
    acne = st.selectbox("Acne", [0,1])
    hair_growth = st.selectbox("Excess Hair Growth", [0,1])
    weight_gain = st.selectbox("Weight Gain", [0,1])
    
    # Wearable Heart Rate Inputs
    st.subheader("Average Heart Rate from Wearable (bpm)")
    hr_menstrual = st.number_input("Menstrual Phase", min_value=50, max_value=120, value=70)
    hr_follicular = st.number_input("Follicular Phase", min_value=50, max_value=120, value=72)
    hr_ovulatory = st.number_input("Ovulatory Phase", min_value=50, max_value=120, value=75)
    hr_luteal = st.number_input("Luteal Phase", min_value=50, max_value=120, value=78)
    
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
        numeric_features = ["Age","BMI","CycleLength","MenstrualIrregularity",
                            "SleepHours","StressLevel","HR_Menstrual","HR_Follicular",
                            "HR_Ovulatory","HR_Luteal"]
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

# Heart Rate trend (simulated)
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
# Lifestyle Suggestions
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
