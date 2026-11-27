# lab43.py
# Evaluate a binary medical-image classifier with standard classification metrics.
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Example labels and predictions (0 = normal, 1 = abnormal).
actual = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
predicted = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])

print("Accuracy :", round(accuracy_score(actual, predicted), 3))
print("Precision:", round(precision_score(actual, predicted), 3))
print("Recall   :", round(recall_score(actual, predicted), 3))
print("F1-score :", round(f1_score(actual, predicted), 3))
print("Confusion matrix:\n", confusion_matrix(actual, predicted))
