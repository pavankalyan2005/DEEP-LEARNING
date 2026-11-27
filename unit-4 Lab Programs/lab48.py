# lab48.py
# RNN to predict the next daily temperature from historical observations.
import numpy as np
import tensorflow as tf

np.random.seed(48)
tf.random.set_seed(48)
days = np.arange(500)
temperature = (25 + 5*np.sin(days*2*np.pi/30) + 0.5*np.sin(days*2*np.pi/7)
               + np.random.normal(0, 0.4, len(days))).astype("float32")
mean, std = temperature.mean(), temperature.std()
scaled = (temperature - mean) / std
window = 14
X, y = [], []
for i in range(len(scaled)-window):
    X.append(scaled[i:i+window])
    y.append(scaled[i+window])
X = np.array(X)[..., None]
y = np.array(y)
split = int(len(X)*0.8)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window, 1)),
    tf.keras.layers.SimpleRNN(32, activation="tanh"),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X[:split], y[:split], epochs=20, batch_size=16, verbose=0)
scaled_prediction = model.predict(X[split:split+1], verbose=0)[0, 0]
prediction = scaled_prediction*std + mean
print("Predicted next temperature:", round(float(prediction), 2), "°C")
print("Actual temperature:", round(float(temperature[split+window]), 2), "°C")
