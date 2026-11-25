import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
inputs = np.array([[0.5], [0.8]])
weights_hidden = np.array([[0.2,0.4],[0.3,0.1]])
bias_hidden = np.array([[0.1],[0.2]])
weights_output = np.array([[0.5,0.6]])
bias_output = np.array([[0.1]])
hidden_output = sigmoid(weights_hidden @ inputs + bias_hidden)
output = sigmoid(weights_output @ hidden_output + bias_output)
print("Hidden layer output:")
print(hidden_output)
print("Final output:")
print(output)
