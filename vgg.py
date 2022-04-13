# import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
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
print(train_labels)
print(len(train_labels))

# print(len(test_ds.class_names))
# print(train_ds.class_names)
# train_labels = to_categorical(train_ds.class_names, num_classes=len(train_ds.class_names))
# test_labels = to_categorical(test_ds.class_names, num_classes=len(test_ds.class_names))