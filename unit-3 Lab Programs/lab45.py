# lab45.py
# Demonstrate image augmentation: rotation, horizontal flip and zoom.
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
image = x_train[0].astype("float32")[None, ..., None] / 255.0
augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.15)
])
augmented = augment(image, training=True)

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.imshow(image[0, :, :, 0], cmap="gray")
plt.title("Original image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(augmented[0, :, :, 0].numpy(), cmap="gray")
plt.title("Augmented image")
plt.axis("off")
plt.tight_layout()
plt.show()
print("Applied rotation, horizontal flip and zoom.")
