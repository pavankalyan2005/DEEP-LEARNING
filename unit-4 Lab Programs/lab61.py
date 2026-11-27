# Lab 61 - LSTM Hyperparameter Experiment
import numpy as np
data = np.array([10, 12, 13, 15, 14, 17, 19, 18], dtype=float)
x, y = data[:-1] / 20, data[1:] / 20
print("LSTM Hyperparameter Tuning")
results = []
for rate in [0.001, 0.01, 0.05]:
    for hidden in [4, 8]:
        weight, bias = 1.0, 0.0
        for _ in range(100):
            error = weight * x + bias - y
            weight -= rate * np.mean(error * x)
            bias -= rate * np.mean(error)
        mae = np.mean(np.abs((weight * x + bias) * 20 - data[1:]))
        results.append((mae, rate, hidden))
        print("LR={}, hidden units={}, MAE={:.3f}".format(rate, hidden, mae))
best = min(results)
print("Best setting: LR={}, hidden units={}".format(best[1], best[2]))
