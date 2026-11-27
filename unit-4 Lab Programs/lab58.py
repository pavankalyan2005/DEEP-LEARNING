# Lab 58 - Stacked RNN for Machine Condition Prediction
# Processes temperature, pressure and vibration readings.

import numpy as np

readings = np.array([[60, 100, 2.1], [62, 102, 2.2], [64, 103, 2.4],
                     [65, 105, 2.5], [67, 106, 2.7]], dtype=float)
layer1 = np.zeros(3)
layer2 = np.zeros(3)
for row in readings:
    layer1 = np.tanh(0.5 * row + 0.5 * layer1)
    layer2 = np.tanh(0.6 * layer1 + 0.4 * layer2)
print("Stacked RNN Machine Condition")
print("Final hidden state:", np.round(layer2, 3))
if layer2.mean() > 0.8:
    print("Machine condition: Normal")
else:
    print("Machine condition: Monitor required")
