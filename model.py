# model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv("data/pcod_dataset.csv")

# Features & target
X = data.drop("PCOD_Risk", axis=1)
y = data["PCOD_Risk"]

# Scale numeric features
numeric_features = ["Age","BMI","CycleLength","MenstrualIrregularity",
                    "SleepHours","StressLevel",
                    "HR_Menstrual","HR_Follicular","HR_Ovulatory","HR_Luteal"]
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved as rf_model.pkl and scaler.pkl")
