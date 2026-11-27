# lab49.py
# RNN-based sentiment classification for positive and negative reviews.
import numpy as np
import tensorflow as tf

reviews = [
    "this product is excellent and useful", "very good quality and happy",
    "i love this item", "great value and fast delivery", "the product works perfectly",
    "this is wonderful", "bad quality and very disappointing", "i hate this product",
    "poor item and slow delivery", "not good and waste of money",
    "the product is terrible", "very bad experience"
]
labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype="float32")
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=1000, output_sequence_length=12, standardize="lower_and_strip_punctuation"
)
vectorizer.adapt(reviews)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    vectorizer,
    tf.keras.layers.Embedding(1000, 16, mask_zero=True),
    tf.keras.layers.SimpleRNN(16),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(np.array(reviews), labels, epochs=30, verbose=0)
tests = ["excellent product and very useful", "poor quality and disappointing"]
scores = model.predict(np.array(tests), verbose=0).flatten()
for review, score in zip(tests, scores):
    print(review, "->", "Positive" if score >= 0.5 else "Negative",
          f"(score={score:.3f})")
