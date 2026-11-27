# lab36.py
# L2 regularization and its influence on model weights.
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(36)
tf.random.set_seed(36)
X = np.random.normal(size=(1000, 10))
y = (X[:, 0] - X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for label, regularizer in [
    ("Without L2", None),
    ("With L2", tf.keras.regularizers.l2(0.01))
]:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=12, verbose=0)
    weights = model.layers[0].get_weights()[0]
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{label}: test accuracy={acc:.3f}, first-layer weight norm={np.linalg.norm(weights):.3f}")
