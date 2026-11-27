# lab41.py
# CNN cat/dog classifier. Arrange images in data/cats and data/dogs folders.
import tensorflow as tf
from pathlib import Path

data_dir = Path("data")
if not (data_dir / "cats").exists() or not (data_dir / "dogs").exists():
    raise SystemExit("Create data/cats and data/dogs folders and place labelled images inside.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=42,
    image_size=(128, 128), batch_size=32, label_mode="binary"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=42,
    image_size=(128, 128), batch_size=32, label_mode="binary"
)
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)
print("Validation result:", model.evaluate(val_ds, verbose=0))
