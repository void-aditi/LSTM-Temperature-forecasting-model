import numpy as np
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Load training data
X = np.load("../data/X.npy")
y = np.load("../data/y.npy")

# LSTM expects 3D input: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build model
model = build_model(X.shape[1])

# Save best model only
os.makedirs("../models", exist_ok=True)
checkpoint = ModelCheckpoint("../models/pune_lstm.h5", save_best_only=True)

# Train model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, callbacks=[checkpoint])

print("Training complete! Model saved as pune_lstm.h5")
