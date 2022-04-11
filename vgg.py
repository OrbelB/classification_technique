# import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt

## Loading images and labels


train_ds = tf.keras.utils.image_dataset_from_directory(
  '/Users/marvinharootoonyan/ct_541/classification_technique/Images/',
  validation_split=0.7,
  subset="training",
  seed=123,
  image_size=(224, 224),
  batch_size=64)

test_ds = tf.keras.utils.image_dataset_from_directory(
  '/Users/marvinharootoonyan/ct_541/classification_technique/Images/',
  validation_split=0.2,
  subset="test",
  seed=123,
  image_size=(224, 224),
  batch_size=64)

val_ds = tf.keras.utils.image_dataset_from_directory(
  '/Users/marvinharootoonyan/ct_541/classification_technique/Images/',
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(224, 224),
  batch_size=64)

(images,labels) = val_ds

print(labels)

