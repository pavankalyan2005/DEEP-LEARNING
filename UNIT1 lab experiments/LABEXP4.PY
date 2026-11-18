import numpy as np

P = np.array([[0, 0.5, 0.5], [1, 0, 0], [0.5, 0.5, 0]])
values, vectors = np.linalg.eig(P.T)
index = np.argmin(abs(values - 1))
rank = np.abs(vectors[:, index].real)
rank = rank / rank.sum()

print("Eigenvalues:", values)
print("Page ranking scores:", rank)
print("Most important page:", np.argmax(rank) + 1)
