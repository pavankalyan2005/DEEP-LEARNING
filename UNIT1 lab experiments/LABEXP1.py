import numpy as np

position = np.array([2, 3, 4])
movement = np.array([1, -2, 3])
scale = 2

print("New position:", position + movement)
print("Difference:", position - movement)
print("Scaled movement:", scale * movement)

 