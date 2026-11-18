import numpy as np

M = np.array([[3, 2], [5, 7]], dtype=float)
det = np.linalg.det(M)
rank = np.linalg.matrix_rank(M)

print("Determinant:", det)
print("Rank:", rank)
if abs(det) > 1e-10:
    print("Inverse:")
    print(np.linalg.inv(M))
    print("Matrix is invertible and suitable for this matrix-based method.")
else:
    print("Matrix is not invertible and is unsuitable for this method.")
