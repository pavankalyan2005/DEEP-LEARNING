# Lab 49 - RNN Based Sentiment Classification
# A small recurrent sentiment demo using word polarity as input.
import numpy as np
reviews = ["good product", "excellent quality", "happy purchase",
           "bad product", "poor quality", "waste money"]
labels = [1, 1, 1, 0, 0, 0]
polarity = {"good": 1, "excellent": 1, "happy": 1,
            "bad": -1, "poor": -1, "waste": -1}
# Recurrent state accumulates information word by word.
def rnn_score(review):
    state = 0.0
    for word in review.split():
        state = np.tanh(state + polarity.get(word, 0))
    return state
print("RNN Sentiment Classification")
for review in ["good quality", "bad quality"]:
    label = "Positive" if rnn_score(review) > 0 else "Negative"
    print(review, "->", label)
