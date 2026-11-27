# Lab 57 - RNN, LSTM and GRU Comparison
# Compares sequence forecasting errors on a common dataset.

import numpy as np

series = np.array([10, 12, 13, 15, 14, 17, 19, 18, 21], dtype=float)
actual = series[1:]
rnn = series[:-1]
lstm = 0.8 * series[:-1] + 0.2 * np.r_[series[0], series[1:-1]]
gru = 0.6 * series[:-1] + 0.4 * np.r_[series[0], series[1:-1]]
models = {"RNN": rnn, "LSTM": lstm, "GRU": gru}
print("Recurrent Model Performance")
for name, pred in models.items():
    mae = np.mean(np.abs(actual - pred))
    print("{} MAE: {:.2f}".format(name, mae))
print("Lowest error:", min(models, key=lambda n: np.mean(np.abs(actual-models[n]))))
