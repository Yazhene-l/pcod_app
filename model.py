import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv("data/pcod_dataset.csv")

# Features and target
features = [
    "Age", "Weight", "Height", "CycleLength", "MenstrualIrregularity",
    "Acne", "HairGrowth", "WeightGain", "Sedentary", "HighSugar",
    "HR_Menstrual", "HR_Follicular", "HR_Ovulatory", "HR_Luteal",
    "BodyTemp", "SleepHours", "StressLevel", "BMI"
]

X = data[features]
y = data["PCOD_Risk"]

# Scale numeric features
numeric_features = ["Age", "Weight", "Height", "CycleLength", "MenstrualIrregularity",
                    "HR_Menstrual", "HR_Follicular", "HR_Ovulatory", "HR_Luteal",
                    "BodyTemp", "SleepHours", "StressLevel", "BMI"]
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X, y)

# Save model
joblib.dump(rf_model, "rf_model.pkl")
print("✅ Model trained and saved as rf_model.pkl")
