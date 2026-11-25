import numpy as np
X = np.array([[2,1],[3,1],[4,2],[5,3],[6,4],[7,5]], dtype=float)
y = np.array([0,0,0,1,1,1])
weights = np.zeros(2)
bias = 0.0
learning_rate = 0.1
for epoch in range(100):
    for features, target in zip(X, y):
        prediction = int(np.dot(features, weights) + bias >= 0)
        error = target - prediction
        weights += learning_rate * error * features
        bias += learning_rate * error
predictions = (X @ weights + bias >= 0).astype(int)
print("Weights:", weights)
print("Bias:", bias)
print("Predictions:", predictions)
print("Accuracy:", np.mean(predictions == y))
