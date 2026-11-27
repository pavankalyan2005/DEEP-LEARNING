# lab38.py
# CNN for handwritten digit recognition using MNIST.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("CNN test accuracy:", round(float(accuracy) * 100, 2), "%")
print("Predicted digit:", int(model.predict(x_test[:1], verbose=0).argmax()))
