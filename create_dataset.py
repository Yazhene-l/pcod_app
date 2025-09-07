# create_dataset.py
import os
from generate_dataset import data

# Ensure data folder exists
if not os.path.exists("data"):
    os.makedirs("data")

# Save dataset
data.to_csv("data/pcod_dataset.csv", index=False)
print("Dataset created at data/pcod_dataset.csv")
