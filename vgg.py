# import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv()
import os


## Loading images and labels

TRAIN_DIR = os.environ.get('TRAIN_DIR')
TEST_DIR = os.environ.get('TEST_DIR')

train_ds = tf.keras.utils.image_dataset_from_directory(
  TRAIN_DIR,
  labels='inferred',
  image_size=(224, 224),
  batch_size=64)
test_ds = tf.keras.utils.image_dataset_from_directory(
  TEST_DIR,
  labels='inferred',
  image_size=(224, 224),
  batch_size=64)


train_labels = np.concatenate([y for x, y in train_ds], axis=0)
test_labels = np.concatenate([y for x, y in test_ds], axis=0)

train_labels = to_categorical(train_labels, num_classes=len(train_ds.class_names))
test_labels = to_categorical(test_labels, num_classes=len(test_ds.class_names))

train_ds = np.concatenate([x for x, y in train_ds.take(1)], axis=0)
test_ds = np.concatenate([x for x, y in test_ds.take(1)], axis=0)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
base_model.trainable = False ## Not trainable weights

## Preprocessing input
train_ds = preprocess_input(train_ds) 
test_ds = preprocess_input(test_ds)

base_model.summary()

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(120, activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])


# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy'],
# )
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


es = EarlyStopping(monitor='val_accuracy', mode='max', patience=500,  restore_best_weights=True)

model.fit(train_ds, train_labels, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])
