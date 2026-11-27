# lab37.py
# Compare patient classification with and without regularization.
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(37)
tf.random.set_seed(37)
X = np.random.normal(size=(1500, 8))
y = (X[:, 0] + X[:, 1] - 0.7 * X[:, 2] > 0.2).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=37, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for label, reg in [("No regularization", None),
                   ("L2 regularization", tf.keras.regularizers.l2(0.01))]:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"{label}: train accuracy={train_acc:.3f}, test accuracy={test_acc:.3f}")
