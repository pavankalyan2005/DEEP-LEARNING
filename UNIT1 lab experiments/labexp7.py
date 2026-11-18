import numpy as np

marks = np.array([78, 85, 67, 90, 72, 88, 76, 95, 81, 69])

print("Mean:", np.mean(marks))
print("Variance:", np.var(marks))
print("Standard deviation:", np.std(marks))
print("Coefficient of variation:", np.std(marks) / np.mean(marks) * 100)
