import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# Load master dataset
df = pd.read_csv("../data/pune_temperature.csv", parse_dates=["date"])

values = df[['tavg']].values

# Scale between 0 and 1
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# Save scaler for inverse transform later
os.makedirs("../models", exist_ok=True)
joblib.dump(scaler, "../models/scaler.save")

SEQ_LEN = 30  # past 30 days

X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN])

X, y = np.array(X), np.array(y)

# Save numpy arrays
os.makedirs("../data", exist_ok=True)
np.save("../data/X.npy", X)
np.save("../data/y.npy", y)

print("Training sequences ready. X.npy and y.npy saved.")
