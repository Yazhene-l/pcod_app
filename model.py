# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv("data/pcod_dataset.csv")

# Calculate BMI if not present
if 'BMI' not in data.columns and 'Weight' in data.columns and 'Height' in data.columns:
    data['BMI'] = data['Weight'] / ((data['Height']/100) ** 2)

# Features & target
features = ['Age', 'BMI', 'HeartRate', 'SleepHours', 'CycleLength', 'CycleRegularity',
            'Acne', 'HairGrowth', 'WeightGain', 'MoodSwings', 'Fatigue',
            'PhysicalActivity', 'DietQuality', 'Stress', 'FamilyHistory']

X = data[features]
y = data['PCOD_Risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Upgraded model and scaler saved successfully!")
