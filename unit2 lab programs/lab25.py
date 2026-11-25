import numpy as np
from sklearn.neural_network import MLPClassifier
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
model = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs", max_iter=5000, random_state=42)
model.fit(X, y)
print("Predicted XOR output:", model.predict(X))
print("Actual XOR output:", y)
print("Accuracy:", model.score(X, y))
