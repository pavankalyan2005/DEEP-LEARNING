# Lab 56 - GRU Multivariate Weather Forecast
# Uses temperature, humidity and pressure observations.

import numpy as np

data = np.array([[28, 70, 1012], [29, 68, 1011], [30, 65, 1010],
                 [31, 63, 1009], [30, 66, 1010]], dtype=float)
# GRU-like update gate combines previous state and latest observation.
state = data[0]
for row in data[1:]:
    update_gate = 0.4
    state = update_gate * row + (1 - update_gate) * state
forecast = state + (data[-1] - data[-2]) * 0.5
print("GRU Weather Forecast")
print("Next temperature: {:.1f} C".format(forecast[0]))
print("Next humidity: {:.1f} %".format(forecast[1]))
print("Next pressure: {:.1f} hPa".format(forecast[2]))
