# lab42.py
# Study the effect of convolution filter size and filter count on accuracy.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:12000].astype("float32")[..., None] / 255.0
x_test = x_test[:2000].astype("float32")[..., None] / 255.0
y_train, y_test = y_train[:12000], y_test[:2000]

for filters, kernel_size in [(16, 3), (32, 3), (32, 5)]:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters, kernel_size, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Filters={filters}, kernel={kernel_size}x{kernel_size}, test accuracy={acc:.3f}")
