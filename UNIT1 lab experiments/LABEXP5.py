import numpy as np

customer1 = np.array([4, 2, 0, 1])
customer2 = np.array([3, 1, 0, 2])
customer3 = np.array([0, 1, 4, 3])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Dot product C1-C2:", np.dot(customer1, customer2))
print("Cosine similarity C1-C2:", cosine_similarity(customer1, customer2))
print("Cosine similarity C1-C3:", cosine_similarity(customer1, customer3))
