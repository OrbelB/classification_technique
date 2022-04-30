import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from dotenv import load_dotenv
load_dotenv()
import os

#point to image
PREDICT_IMAGE = os.environ.get('PREDICT_IMAGE')
#point and load to compiled and trained model
model_bin_path = os.path.join("model_bin")
model = keras.models.load_model(model_bin_path)

#we have to get class names again
dataset, info  = tfds.load(
    "stanford_dogs",
    split=["train", "test"],
    shuffle_files=True,
    with_info=True,
    as_supervised=True,
)

class_names = info.features['label'].names

#load image to array
img = tf.keras.utils.load_img(
    PREDICT_IMAGE, target_size=(224, 224)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

