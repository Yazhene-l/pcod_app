import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv("data/pcod_dataset.csv")

# Features for model
features = [
    "Age","BMI","CycleLength","Irregular","Symptoms_Acne","Symptoms_HairGrowth",
    "Symptoms_WeightGain","Lifestyle_Sedentary","Hormone_FamilyHistory","FertilityIssues",
    "HR","BodyTemp","SleepHours","Stress_HRV"
]

X = data[features]
y = data["PCOD_Risk"]

# Scale numeric features
numeric_features = ["Age","BMI","CycleLength","HR","BodyTemp","SleepHours","Stress_HRV"]
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model & scaler
joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model trained and saved")
