# Lab 51 - RNN Time Series Forecasting
# Predicts the next stock price from historical observations.
import numpy as np
prices = np.array([100, 102, 101, 104, 106, 105, 108, 110], dtype=float)
# Normalize values to keep training stable.
x = prices[:-1] / 100
y = prices[1:] / 100
w, b = 1.0, 0.0
for _ in range(1500):
    error = w * x + b - y
    w -= 0.05 * np.mean(error * x)
    b -= 0.05 * np.mean(error)
prediction = (w * (prices[-1] / 100) + b) * 100
mae = np.mean(np.abs((w * x + b) * 100 - prices[1:]))
print("RNN Stock Price Forecast")
print("Historical prices:", prices.astype(int))
print("Next predicted price: {:.2f}".format(prediction))
print("Training MAE: {:.2f}".format(mae))
