# Lab 53 - RNN Next Word Prediction
# Predicts the next word in a short sample sentence.

import numpy as np

sentences = ["deep learning is useful", "deep learning is powerful",
             "machine learning is useful"]
pairs = {}
for sentence in sentences:
    words = sentence.split()
    for a, b in zip(words, words[1:]):
        pairs.setdefault(a, {})
        pairs[a][b] = pairs[a].get(b, 0) + 1

print("RNN Next Word Prediction")
for word in ["deep", "learning", "is", "machine"]:
    choices = pairs.get(word, {})
    if choices:
        next_word = max(choices, key=choices.get)
        print(word, "->", next_word)
