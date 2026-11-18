import numpy as np

x = 10.0
learning_rate = 0.1
iterations = 100

for _ in range(iterations):
    gradient = 2 * (x - 3)
    x = x - learning_rate * gradient

print("Minimum point:", x)
print("Minimum cost:", (x - 3) ** 2)
