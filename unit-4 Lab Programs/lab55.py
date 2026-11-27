# Lab 55 - LSTM Stock Price Prediction
# Uses a simple gated-memory style update for forecasting.

import numpy as np

prices = np.array([210, 212, 211, 215, 218, 217, 220, 224, 223], dtype=float)
memory = 0.0
predictions = []
for i in range(1, len(prices)):
    memory = 0.7 * memory + 0.3 * (prices[i-1] - prices[0])
    predictions.append(prices[0] + memory)
actual = prices[1:]
mae = np.mean(np.abs(np.array(predictions) - actual))
rmse = np.sqrt(np.mean((np.array(predictions) - actual) ** 2))
next_value = prices[-1] + (prices[-1] - prices[-2]) * 0.7
print("LSTM Stock Price Prediction")
print("Predicted next stock price: {:.2f}".format(next_value))
print("MAE: {:.2f}".format(mae))
print("RMSE: {:.2f}".format(rmse))
