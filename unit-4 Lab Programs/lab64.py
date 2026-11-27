# Lab 64 - Recursive Neural Network Sentiment Classification
# Demonstrates combining child-node representations in a sentence tree.

import numpy as np

# Leaf word vectors: positive words have positive values, negative words negative.
words = {"good": np.array([0.9, 0.8]), "great": np.array([1.0, 0.9]),
         "bad": np.array([-0.9, -0.8]), "poor": np.array([-0.8, -0.9])}
W = np.array([[0.6, 0.2], [0.2, 0.6]])

def combine(left, right):
    return np.tanh(W @ (left + right))

def classify(first, second):
    root = combine(words[first], words[second])
    score = root.mean()
    return "Positive" if score >= 0 else "Negative"

print("Recursive Neural Network Sentiment")
print("good great ->", classify("good", "great"))
print("bad poor ->", classify("bad", "poor"))
