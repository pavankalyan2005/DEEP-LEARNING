import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)
np.random.seed(1)
W1 = np.random.uniform(-1, 1, (2,4))
W2 = np.random.uniform(-1, 1, (4,1))
learning_rate = 0.5
for epoch in range(10000):
    hidden = sigmoid(X @ W1)
    output = sigmoid(hidden @ W2)
    output_delta = (y - output) * sigmoid_derivative(output)
    hidden_delta = (output_delta @ W2.T) * sigmoid_derivative(hidden)
    W2 += learning_rate * hidden.T @ output_delta
    W1 += learning_rate * X.T @ hidden_delta
print("Predicted outputs:")
print(sigmoid(sigmoid(X @ W1) @ W2))
print("Updated hidden weights:")
print(W1)
print("Updated output weights:")
print(W2)
