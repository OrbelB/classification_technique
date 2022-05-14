import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import os
import numpy as np 
import tensorflow_datasets as tfds

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.2),
  layers.RandomCrop(224, 224)
])

def input_preprocess(image, label):
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image, label

def augment(image, label):
    image = data_augmentation(image)
    return image, label

# TEST_DIR = os.path.join("DogTestImage")


IMG_SIZE = 224
# BATCH_SIZE = 64
# BUFFER_SIZE = 2


# ds_test = tf.keras.utils.image_dataset_from_directory(
#   TEST_DIR,
#   image_size=(224, 224),
#   batch_size=64)

(ds_train, ds_test), metadata = tfds.load(
    "stanford_dogs",
    split=["train", "test"],
    shuffle_files=True,
    with_info=True,
    as_supervised=True,
)
class_names = metadata.features['label'].names
NUM_CLASSES = metadata.features["label"].num_classes


# IMG_SIZE = 224
BATCH_SIZE = 64
# BUFFER_SIZE = 2
size = (IMG_SIZE, IMG_SIZE)


ds_test = ds_test.map(augment)
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))





ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)


model_bin_path = os.path.join("bcnn-dogs/save_at_19")
model = tf.keras.models.load_model(model_bin_path)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)
model.summary()
results = model.evaluate(ds_test)

