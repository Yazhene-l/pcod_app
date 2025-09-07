import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_samples = 300

# Simulate dataset
data = pd.DataFrame({
    "Age": np.random.randint(18, 45, n_samples),
    "Weight": np.round(np.random.uniform(45, 90, n_samples), 1),
    "Height": np.round(np.random.uniform(150, 175, n_samples), 1),
    "CycleLength": np.random.randint(21, 50, n_samples),
    "LastPeriodStart": [datetime.today() - timedelta(days=np.random.randint(1,35)) for _ in range(n_samples)],
    "MenstrualHistory": np.random.randint(21, 50, n_samples),  # Average cycle length over past 3-6 cycles
    "Symptoms_Acne": np.random.randint(0,2, n_samples),
    "Symptoms_HairGrowth": np.random.randint(0,2, n_samples),
    "Symptoms_WeightGain": np.random.randint(0,2, n_samples),
    "Lifestyle_Sedentary": np.random.randint(0,2, n_samples),
    "Hormone_FamilyHistory": np.random.randint(0,2, n_samples),
    "FertilityIssues": np.random.randint(0,2, n_samples),
    "HR": np.random.randint(65, 95, n_samples),
    "BodyTemp": np.random.uniform(36.2, 37.5, n_samples),
    "SleepHours": np.random.uniform(4,9, n_samples).round(1),
    "Stress_HRV": np.random.randint(20,100, n_samples)
})

# BMI calculation
data["BMI"] = (data["Weight"] / (data["Height"]/100)**2).round(1)

# Phase-wise HR estimation based on average HR & cycle
data["HR_Menstrual"] = data["HR"] - np.random.randint(0,5, n_samples)
data["HR_Follicular"] = data["HR"] + np.random.randint(0,3, n_samples)
data["HR_Ovulatory"] = data["HR"] + np.random.randint(2,6, n_samples)
data["HR_Luteal"] = data["HR"] + np.random.randint(3,8, n_samples)

# Cycle irregularity score (simple heuristic)
data["CycleIrregularity"] = abs(data["CycleLength"] - data["MenstrualHistory"])
data["Irregular"] = data["CycleIrregularity"].apply(lambda x: 1 if x>7 else 0)

# Assign PCOD Risk (simple heuristic for demo)
def assign_risk(row):
    score = row["Symptoms_Acne"] + row["Symptoms_HairGrowth"] + row["Symptoms_WeightGain"] + row["Irregular"]
    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"

data["PCOD_Risk"] = data.apply(assign_risk, axis=1)

# Save CSV
data.to_csv("data/pcod_dataset.csv", index=False)
print("Dataset created: data/pcod_dataset.csv")
