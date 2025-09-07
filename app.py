import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌸 PCOD Risk Prediction App")

st.header("👩 User Inputs (Manual)")
age = st.number_input("Age (years)", 18, 45, 25)
weight = st.number_input("Weight (kg)", 40, 120, 55)
height = st.number_input("Height (cm)", 140, 200, 160)
cycle_length = st.number_input("Average Cycle Length (days)", 21, 120, 30)
menstrual_history = st.text_input("Last 3-6 Cycle Lengths (comma separated)", "28,30,32")
acne = st.selectbox("Acne", [0,1])
hair_growth = st.selectbox("Excess Hair Growth", [0,1])
weight_gain = st.selectbox("Weight Gain", [0,1])
sedentary = st.selectbox("Sedentary Lifestyle", [0,1])
high_sugar = st.selectbox("High Sugar Intake", [0,1])

st.header("⌚ Device Inputs (Automatic)")
hr = st.number_input("Average Heart Rate (bpm)", 50, 120, 70)
body_temp = st.number_input("Body Temperature (°C)", 35.0, 40.0, 36.6, format="%.1f")
stress_level = st.number_input("Stress Level (0-5)", 0.0, 5.0, 2.5, format="%.1f")
sleep_hours = st.number_input("Sleep Hours", 0, 12, 7)

# Compute menstrual irregularity
history_list = [int(x.strip()) for x in menstrual_history.split(",")]
menstrual_irregularity = np.std(history_list)

# Approximate phase-wise HR
hr_menstrual = hr
hr_follicular = hr + 2
hr_ovulatory = hr + 5
hr_luteal = hr + 8

# Compute BMI
bmi = round(weight / (height/100)**2, 1)

# Prepare input DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Weight": weight,
    "Height": height,
    "CycleLength": cycle_length,
    "MenstrualIrregularity": menstrual_irregularity,
    "Acne": acne,
    "HairGrowth": hair_growth,
    "WeightGain": weight_gain,
    "Sedentary": sedentary,
    "HighSugar": high_sugar,
    "HR_Menstrual": hr_menstrual,
    "HR_Follicular": hr_follicular,
    "HR_Ovulatory": hr_ovulatory,
    "HR_Luteal": hr_luteal,
    "BodyTemp": body_temp,
    "SleepHours": sleep_hours,
    "StressLevel": stress_level,
    "BMI": bmi
}])

# Scale numeric features
numeric_features = ["Age", "Weight", "Height", "CycleLength", "MenstrualIrregularity",
                    "HR_Menstrual", "HR_Follicular", "HR_Ovulatory", "HR_Luteal",
                    "BodyTemp", "SleepHours", "StressLevel", "BMI"]
input_data[numeric_features] = scaler.transform(input_data[numeric_features])

# Prediction
if st.button("🔮 Predict PCOD Risk"):
    prediction = rf_model.predict(input_data)[0]
    st.success(f"Predicted PCOD Risk: {prediction}")

    # --- Suggestions ---
    st.subheader("💡 Suggestions")
    if prediction == "Low":
        st.info("✅ Maintain your healthy lifestyle. Keep exercising and eating balanced meals.")
    elif prediction == "Medium":
        st.warning("⚠️ You may be at moderate risk. Track your cycles, manage stress, and consult a gynecologist if irregularities persist.")
    else:
        st.error("🚨 High risk detected. Please consult a healthcare professional for further evaluation and management.")

    # --- Phase-wise Heart Rate Comparison Graph ---
    st.subheader("📊 Heart Rate Comparison (You vs Average PCOD / Non-PCOD)")
    phases = ["Menstrual", "Follicular", "Ovulatory", "Luteal"]
    your_hr = [hr_menstrual, hr_follicular, hr_ovulatory, hr_luteal]
    pcod_avg = [78, 80, 83, 85]  # assumed higher HR for PCOD
    non_pcod_avg = [70, 72, 75, 78]  # normal averages

    fig, ax = plt.subplots()
    ax.plot(phases, your_hr, marker='o', label="You")
    ax.plot(phases, pcod_avg, marker='o', label="PCOD Avg")
    ax.plot(phases, non_pcod_avg, marker='o', label="Non-PCOD Avg")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title("Cycle Phase vs Heart Rate")
    ax.legend()
    st.pyplot(fig)

    # --- BMI Comparison ---
    st.subheader("📊 BMI Comparison")
    fig2, ax2 = plt.subplots()
    categories = ["You", "PCOD Avg", "Non-PCOD Avg"]
    values = [bmi, 28, 22]  # example averages
    ax2.bar(categories, values, color=["blue","red","green"])
    ax2.set_ylabel("BMI")
    st.pyplot(fig2)

    # --- Stress Comparison ---
    st.subheader("📊 Stress Level Comparison")
    fig3, ax3 = plt.subplots()
    categories = ["You", "PCOD Avg", "Non-PCOD Avg"]
    values = [stress_level, 4, 2]
    ax3.bar(categories, values, color=["blue","red","green"])
    ax3.set_ylabel("Stress Level (0-5)")
    st.pyplot(fig3)
