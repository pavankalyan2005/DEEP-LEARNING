import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = 2 * X + 1

def train(method, epochs=100, lr=0.01, batch_size=2):
    w, b = 0.0, 0.0
    for _ in range(epochs):
        if method == "batch":
            batches = [np.arange(len(X))]
        elif method == "stochastic":
            batches = [np.array([i]) for i in np.random.permutation(len(X))]
        else:
            order = np.random.permutation(len(X))
            batches = [order[i:i + batch_size] for i in range(0, len(X), batch_size)]
        for indices in batches:
            error = w * X[indices] + b - y[indices]
            dw = 2 * np.mean(error * X[indices])
            db = 2 * np.mean(error)
            w -= lr * dw
            b -= lr * db
    return w, b, np.mean((w * X + b - y) ** 2)

for method in ["batch", "stochastic", "mini-batch"]:
    print(method, train(method))

 