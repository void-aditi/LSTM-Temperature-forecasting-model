from datetime import datetime
import pandas as pd
from meteostat import Point, Daily
import os  # <- add this

# Pune coordinates
pune = Point(18.5204, 73.8567)

start = datetime(2000, 1, 1)
end = datetime(2024, 12, 31)

data = Daily(pune, start, end)
df = data.fetch()

df = df[['tavg']]          # Keep only average temperature
df = df.dropna()
df = df.reset_index()

# Make sure data folder exists
os.makedirs("../data", exist_ok=True)

df.to_csv("../data/pune_temperature_raw.csv", index=False)

print("Raw Pune temperature data saved.")
