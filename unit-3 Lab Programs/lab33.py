# lab33.py
# Compare SGD, RMSprop and Adam optimizers for defect classification.
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

# Synthetic sensor measurements: class 1 represents a defective product.
X = np.random.normal(0, 1, (1000, 4))
y = ((X[:, 0] + 0.8 * X[:, 1] - 0.5 * X[:, 2]) > 0.4).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for name, optimizer in [
    ("SGD", tf.keras.optimizers.SGD(learning_rate=0.05)),
    ("RMSprop", tf.keras.optimizers.RMSprop(learning_rate=0.001)),
    ("Adam", tf.keras.optimizers.Adam(learning_rate=0.001))
]:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=12, batch_size=32, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name}: final training loss={history.history['loss'][-1]:.4f}, "
          f"test accuracy={accuracy:.4f}")
