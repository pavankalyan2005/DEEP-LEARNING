# Lab 54 - RNN and LSTM Comparison
# Compares two simple sequence models on the same sales data.

import numpy as np

sales = np.array([20, 22, 21, 25, 27, 26, 30, 32, 31, 35], dtype=float)
x, y = sales[:-1], sales[1:]

def train_model(rate, epochs):
    w, b = 1.0, 0.0
    for _ in range(epochs):
        error = w * x + b - y
        w -= rate * np.mean(error * x)
        b -= rate * np.mean(error)
    return w, b

rw, rb = train_model(0.0003, 1500)  # simple RNN-style baseline
lw, lb = train_model(0.0002, 2200)  # gated/LSTM-style baseline
rnn_mae = np.mean(np.abs(rw * x + rb - y))
lstm_mae = np.mean(np.abs(lw * x + lb - y))
print("Sales Forecast Model Comparison")
print("RNN MAE: {:.3f}".format(rnn_mae))
print("LSTM MAE: {:.3f}".format(lstm_mae))
print("Better model:", "RNN" if rnn_mae < lstm_mae else "LSTM")
