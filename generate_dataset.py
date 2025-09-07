import pandas as pd
import numpy as np

np.random.seed(42)  # For reproducibility
n_samples = 200

data = pd.DataFrame({
    "Age": np.random.randint(18, 45, n_samples),
    "BMI": np.round(np.random.uniform(18, 35, n_samples), 1),
    "CycleLength": np.random.randint(21, 50, n_samples),
    "MenstrualIrregularity": np.random.randint(0, 6, n_samples),
    "SleepHours": np.random.uniform(4, 9, n_samples).round(1),
    "StressLevel": np.random.randint(0, 6, n_samples),
    "Acne": np.random.randint(0,2, n_samples),
    "HairGrowth": np.random.randint(0,2, n_samples),
    "WeightGain": np.random.randint(0,2, n_samples),
    "HR_Menstrual": np.random.randint(60, 85, n_samples),
    "HR_Follicular": np.random.randint(65, 90, n_samples),
    "HR_Ovulatory": np.random.randint(68, 95, n_samples),
    "HR_Luteal": np.random.randint(70, 92, n_samples),
})

# Assign PCOD_Risk based on simple heuristic (for demo purpose)
def risk(row):
    score = row["MenstrualIrregularity"] + row["Acne"] + row["HairGrowth"] + row["WeightGain"]
    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"

data["PCOD_Risk"] = data.apply(risk, axis=1)

# Save as CSV
data.to_csv("data/pcod_dataset.csv", index=False)
print("Dataset created: data/pcod_dataset.csv")
