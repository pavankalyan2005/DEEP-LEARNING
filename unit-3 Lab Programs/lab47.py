# lab47.py
# Basic RNN for sequential machine sensor readings.
import numpy as np
import tensorflow as tf

np.random.seed(47)
tf.random.set_seed(47)
# Generate smooth sensor sequences with a repeating temporal pattern.
t = np.arange(1200)
signal = (np.sin(t / 12) + 0.3 * np.sin(t / 4)).astype("float32")
X, y = [], []
window = 20
for i in range(len(signal) - window):
    X.append(signal[i:i+window])
    y.append(signal[i+window])
X = np.array(X)[..., None]
y = np.array(y)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window, 1)),
    tf.keras.layers.SimpleRNN(32, activation="tanh"),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=8, batch_size=32, verbose=0)
prediction = model.predict(X[-1:], verbose=0)[0, 0]
print("Actual next sensor value:", round(float(y[-1]), 4))
print("Predicted next sensor value:", round(float(prediction), 4))
print("Training loss:", round(float(model.evaluate(X, y, verbose=0)), 5))
