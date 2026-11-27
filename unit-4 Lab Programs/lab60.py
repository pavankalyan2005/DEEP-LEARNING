# Lab 60 - Effect of Sequence Length on Forecasting
import numpy as np
series = np.array([10, 12, 11, 15, 14, 18, 17, 22, 20, 25], dtype=float)
print("Sequence Length Experiment")
for length in [2, 3, 4, 5]:
    errors = []
    for i in range(length, len(series)):
        window = series[i-length:i]
        # Estimate the next value using the average change in the window.
        prediction = window[-1] + np.mean(np.diff(window))
        errors.append(abs(prediction - series[i]))
    print("Sequence length {}: MAE = {:.2f}".format(length, np.mean(errors)))
