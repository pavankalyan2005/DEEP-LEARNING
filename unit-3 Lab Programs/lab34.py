# lab34.py
# Handwritten PIN/digit recognition using a deep feedforward network and MNIST.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("MNIST test loss:", round(float(loss), 4))
print("MNIST test accuracy:", round(float(accuracy) * 100, 2), "%")
print("Predicted digit for first test image:", int(model.predict(x_test[:1], verbose=0).argmax()))
