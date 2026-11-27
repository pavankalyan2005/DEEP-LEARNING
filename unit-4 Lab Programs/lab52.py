# Lab 52 - RNN Electricity Demand Forecasting
# Forecasts next-day electricity consumption.
import numpy as np
demand = np.array([120, 125, 123, 130, 135, 132, 140, 145], dtype=float)
x, y = demand[:-1] / 100, demand[1:] / 100
w, b = 1.0, 0.0
for _ in range(1500):
    error = w * x + b - y
    w -= 0.05 * np.mean(error * x)
    b -= 0.05 * np.mean(error)
next_day = (w * demand[-1] / 100 + b) * 100
print("Electricity Demand Forecast")
print("Past demand:", demand.astype(int))
print("Predicted next-day demand: {:.2f} units".format(next_day))
