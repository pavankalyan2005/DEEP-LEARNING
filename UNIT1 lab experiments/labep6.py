import numpy as np

customer_a = np.array([10, 20, 5])
customer_b = np.array([13, 24, 9])

euclidean = np.sqrt(np.sum((customer_a - customer_b) ** 2))
manhattan = np.sum(np.abs(customer_a - customer_b))

print("Euclidean distance:", euclidean)
print("Manhattan distance:", manhattan)

 