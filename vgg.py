# import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt

## Loading images and labels


train_ds = tf.keras.utils.image_dataset_from_directory(
  '/Users/marvinharootoonyan/ct_541/classification_technique/TrainImages/',
  seed=123,
  image_size=(224, 224),
  batch_size=64)
test_ds = tf.keras.utils.image_dataset_from_directory(
  '/Users/marvinharootoonyan/ct_541/classification_technique/TestImages/',
  seed=123,
  image_size=(224, 224),
  batch_size=64)



print(len(train_ds.class_names))
print(len(test_ds.class_names))
train_labels = to_categorical(train_ds.class_names, num_classes=len(train_ds.class_names))
test_labels = to_categorical(test_ds.class_names, num_classes=len(test_ds.class_names))