from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model(seq_len):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
