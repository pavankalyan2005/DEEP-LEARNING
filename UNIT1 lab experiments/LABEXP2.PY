import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])

print("A multiplied by B:")
print(A @ B)
print("Transpose of A:")
print(A.T)
print("Transpose of product:")
print((A @ B).T)
