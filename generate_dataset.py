import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 600

# Simulate irregular cycle lengths
possible_cycles = [28, 30, 32, 35, 45, 60, 90]

data = pd.DataFrame({
    "Age": np.random.randint(18, 45, n_samples),
    "Weight": np.random.randint(45, 90, n_samples),
    "Height": np.random.randint(150, 180, n_samples),
    "CycleLength": [np.mean(np.random.choice(possible_cycles, size=3)) for _ in range(n_samples)],
    "MenstrualIrregularity": [np.std(np.random.choice(possible_cycles, size=3)) for _ in range(n_samples)],
    "Acne": np.random.randint(0, 2, n_samples),
    "HairGrowth": np.random.randint(0, 2, n_samples),
    "WeightGain": np.random.randint(0, 2, n_samples),
    "Sedentary": np.random.randint(0, 2, n_samples),
    "HighSugar": np.random.randint(0, 2, n_samples),
    "HR_Menstrual": np.random.randint(60, 85, n_samples),
    "HR_Follicular": np.random.randint(65, 90, n_samples),
    "HR_Ovulatory": np.random.randint(68, 95, n_samples),
    "HR_Luteal": np.random.randint(70, 92, n_samples),
    "BodyTemp": np.round(np.random.uniform(36, 37.5, n_samples), 1),
    "SleepHours": np.round(np.random.uniform(4, 9, n_samples), 1),
    "StressLevel": np.round(np.random.uniform(0, 5, n_samples), 1)
})

# Compute BMI
data["BMI"] = np.round(data["Weight"] / (data["Height"]/100)**2, 1)

# Assign PCOD_Risk
def risk(row):
    score = row["MenstrualIrregularity"] + row["Acne"] + row["HairGrowth"] + row["WeightGain"] + row["Sedentary"] + row["HighSugar"]
    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"

data["PCOD_Risk"] = data.apply(risk, axis=1)

# Save dataset
data.to_csv("data/pcod_dataset.csv", index=False)
print("✅ Dataset created at data/pcod_dataset.csv")
