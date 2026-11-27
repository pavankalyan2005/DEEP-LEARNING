# lab35.py
# Compare a neural network before and after applying dropout.
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(35)
tf.random.set_seed(35)
X = np.random.normal(size=(1200, 12))
y = ((X[:, :4].sum(axis=1) + 0.4 * X[:, 4]) > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=35)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_model(dropout=False):
    layers = [tf.keras.layers.Input(shape=(12,)), tf.keras.layers.Dense(64, activation="relu")]
    if dropout:
        layers.append(tf.keras.layers.Dropout(0.5))
    layers += [tf.keras.layers.Dense(32, activation="relu"),
               tf.keras.layers.Dense(1, activation="sigmoid")]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

for label, use_dropout in [("Without Dropout", False), ("With Dropout", True)]:
    model = build_model(use_dropout)
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"{label}: training accuracy={train_acc:.3f}, testing accuracy={test_acc:.3f}")
