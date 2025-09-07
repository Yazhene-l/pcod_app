# app.py
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Advanced PCOD Risk Detection App")

# User Input Form
with st.form("pcod_form"):
    st.header("Enter Your Details")

    # Personal Info
    age = st.number_input("Age", min_value=10, max_value=60, value=25)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=150, value=60)
    height = st.number_input("Height (cm)", min_value=100, max_value=220, value=160)

    # Wearable + Health Metrics
    heart_rate = st.number_input("Average Heart Rate (bpm)", min_value=50, max_value=120, value=75)
    sleep_hours = st.number_input("Average Sleep Hours", min_value=0, max_value=12, value=7)

    # Menstrual Cycle Info
    cycle_length = st.number_input("Cycle Length (days)", min_value=15, max_value=60, value=30)
    cycle_regular = st.radio("Is your cycle regular?", ("Yes", "No"))

    # Symptoms
    acne = st.radio("Do you have acne?", ("Yes", "No"))
    hair_growth = st.radio("Excess hair growth?", ("Yes", "No"))
    weight_gain = st.radio("Recent weight gain?", ("Yes", "No"))
    mood_swings = st.radio("Mood swings?", ("Yes", "No"))
    fatigue = st.radio("Fatigue?", ("Yes", "No"))

    # Lifestyle
    physical_activity = st.radio("Are you physically active?", ("Yes", "No"))
    diet_quality = st.radio("Is your diet healthy?", ("Yes", "No"))
    stress = st.slider("Stress Level (0-10)", 0, 10, 5)

    # Family History
    family_history = st.radio("Family history of PCOD or diabetes?", ("Yes", "No"))

    submit = st.form_submit_button("Predict Risk")

if submit:
    # Convert Yes/No to 1/0
    acne = 1 if acne == "Yes" else 0
    hair_growth = 1 if hair_growth == "Yes" else 0
    weight_gain = 1 if weight_gain == "Yes" else 0
    mood_swings = 1 if mood_swings == "Yes" else 0
    fatigue = 1 if fatigue == "Yes" else 0
    physical_activity = 1 if physical_activity == "Yes" else 0
    diet_quality = 1 if diet_quality == "Yes" else 0
    cycle_regular = 1 if cycle_regular == "Yes" else 0
    family_history = 1 if family_history == "Yes" else 0

    # Calculate BMI
    bmi = weight / ((height/100) ** 2)

    # Prepare feature array
    user_data = np.array([[age, bmi, heart_rate, sleep_hours, cycle_length, cycle_regular,
                           acne, hair_growth, weight_gain, mood_swings, fatigue,
                           physical_activity, diet_quality, stress, family_history]])

    # Scale features
    user_data_scaled = scaler.transform(user_data)

    # Predict PCOD Risk
    prediction = rf_model.predict(user_data_scaled)[0]
    st.success(f"Your PCOD Risk Level: {prediction}")
