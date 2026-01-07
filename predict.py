import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Load scaler and model
scaler = joblib.load("../models/scaler.save")
model = load_model("../models/pune_lstm.h5", compile=False)

# Load dataset
df = pd.read_csv("../data/pune_temperature.csv", parse_dates=["date"])
values = df[['tavg']].values
scaled = scaler.transform(values)

SEQ_LEN = 90  #looks at previous 30 days to predict next day.

# Split train-test (last 250 days for testing)
train = scaled[:-250]
test = scaled[-250:]

# Create test sequences
X_test, y_test = [], []
for i in range(len(test) - SEQ_LEN):
    X_test.append(test[i:i+SEQ_LEN])
    y_test.append(test[i+SEQ_LEN])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Predict on test data
predicted = model.predict(X_test)

# Inverse scale
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test.reshape(-1,1))

# Calculate accuracy metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print("\nMODEL ACCURACY")
print("MAE  :", round(mae, 2), "°C")
print("RMSE :", round(rmse, 2), "°C")
print("MAPE :", round(mape, 2), "%")

# --------- FUTURE FORECAST (same as before) ---------

last_seq = scaled[-SEQ_LEN:]
last_seq = last_seq.reshape((1, SEQ_LEN, 1))

N_DAYS = 7
future_preds = []

for _ in range(N_DAYS):
    pred = model.predict(last_seq)[0][0]
    future_preds.append(pred)
    last_seq = np.append(last_seq[:,1:,:], [[[pred]]], axis=1)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

future_dates = pd.date_range(start=df['date'].iloc[-1] + pd.Timedelta(days=1), periods=N_DAYS)

# Plot
plt.figure(figsize=(10,5))
plt.plot(df['date'][-60:], df['tavg'][-60:], label='Last 60 days Actual')
plt.plot(future_dates, future_preds, label='Next 7 Days Predicted', marker='o')
plt.legend()
plt.grid(True)
plt.title("Pune Temperature Forecast")
plt.show()
