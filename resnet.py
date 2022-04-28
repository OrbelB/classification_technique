import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


(ds_train, ds_test), metadata = tfds.load(
    "stanford_dogs",
    split=["train", "test"],
    shuffle_files=True,
    with_info=True,
    as_supervised=True,
)
 
NUM_CLASSES = metadata.features["label"].num_classes

print("Number of training samples: %d" % tf.data.experimental.cardinality(ds_train))
print("Number of test samples: %d" % tf.data.experimental.cardinality(ds_test))
print("Number of classes: %d" % NUM_CLASSES)

# plt.figure(figsize=(10, 10))
# for i, (image, label) in enumerate(ds_train.take(9)):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(image)
#     plt.title(int(label))
#     plt.axis("off")


IMG_SIZE = 224
BATCH_SIZE = 64
BUFFER_SIZE = 2
 
size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
 
def input_preprocess(image, label):
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
 
ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
 
ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)

inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_tensor=inputs
)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
# x = tf.keras.layers.Dense(NUM_CLASSES, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES)(x)
 
model = tf.keras.Model(inputs, outputs)

base_model.trainable = False

MODEL_PATH = "resnet-dogs"
checkpoint_path = os.path.join(MODEL_PATH, "save_at_{epoch}")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
    #tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
]

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

test_data = ds_test
epochs=20
train_data = ds_train
model.fit(
    train_data, epochs=epochs, callbacks=callbacks, validation_data=test_data, verbose=2
)
model_bin_path = os.path.join("model_bin")
model.save(model_bin_path)
print(model.evaluate(test_data))

