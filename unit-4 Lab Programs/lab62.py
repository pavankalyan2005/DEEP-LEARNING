# Lab 62 - RNN Forecast Error Metrics

import numpy as np

actual = np.array([120, 130, 125, 140, 150], dtype=float)
predicted = np.array([118, 132, 123, 137, 148], dtype=float)
error = actual - predicted
mae = np.mean(np.abs(error))
mse = np.mean(error ** 2)
rmse = np.sqrt(mse)
print("Electricity Forecast Error Evaluation")
print("MAE: {:.2f}".format(mae))
print("MSE: {:.2f}".format(mse))
print("RMSE: {:.2f}".format(rmse))
