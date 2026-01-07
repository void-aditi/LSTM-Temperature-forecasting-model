import pandas as pd
import os

# Load raw data
df = pd.read_csv("../data/pune_temperature_raw.csv", parse_dates=["time"])

# Rename column
df = df.rename(columns={"time": "date"})

# Sort by date
df = df.sort_values("date")

# Fill missing dates (daily frequency)
df = df.set_index("date").asfreq("D").fillna(method="ffill")

# Make sure data folder exists
os.makedirs("../data", exist_ok=True)

# Save cleaned master dataset
df.to_csv("../data/pune_temperature.csv")
print("Master Pune dataset ready: pune_temperature.csv")
