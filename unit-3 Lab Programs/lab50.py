# lab50.py
# Character-level RNN text generation from a short training document.
import numpy as np
import tensorflow as tf

text = ("deep learning helps computers learn patterns from data. "
        "recurrent neural networks process sequential information. ") * 20
chars = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = np.array(chars)
sequence_length = 25
encoded = np.array([char_to_id[c] for c in text])
X, y = [], []
for i in range(len(encoded)-sequence_length):
    X.append(encoded[i:i+sequence_length])
    y.append(encoded[i+sequence_length])
X = np.array(X)
y = np.array(y)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(sequence_length,)),
    tf.keras.layers.Embedding(len(chars), 32),
    tf.keras.layers.SimpleRNN(128),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(X, y, epochs=15, batch_size=64, verbose=0)

seed = "deep learning helps computers"
generated = seed
for _ in range(100):
    ids = np.array([[char_to_id[c] for c in generated[-sequence_length:]]])
    probabilities = model.predict(ids, verbose=0)[0]
    next_id = np.random.choice(len(chars), p=probabilities / probabilities.sum())
    generated += id_to_char[next_id]
print("Generated text:")
print(generated)
