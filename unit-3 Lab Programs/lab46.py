# lab46.py
# Transfer learning for plant disease classification using pretrained ResNet50.
# Place labelled leaf images in data/<class_name>/ folders.
import tensorflow as tf
from pathlib import Path

data_dir = Path("data")
if not data_dir.exists():
    raise SystemExit("Create a data folder with one subfolder per plant-disease class.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=42,
    image_size=(224, 224), batch_size=16
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=42,
    image_size=(224, 224), batch_size=16
)
base = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.resnet50.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(len(train_ds.class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)
print("Plant disease validation result:", model.evaluate(val_ds, verbose=0))
