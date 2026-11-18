import numpy as np

actual = np.array([1, 0, 1, 1, 0])
predicted = np.array([0.90, 0.20, 0.70, 0.60, 0.10])
epsilon = 1e-15
predicted = np.clip(predicted, epsilon, 1 - epsilon)

loss = -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
print("Cross-entropy loss:", loss)
