import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import os
import numpy as np 
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

(ds_train, ds_test), metadata = tfds.load(
    "stanford_dogs",
    split=["train", "test"],
    shuffle_files=True,
    with_info=True,
    as_supervised=True,
)
class_names = metadata.features['label'].names
NUM_CLASSES = metadata.features["label"].num_classes



model_bin_path = os.path.join("save_at_18")
model = tf.keras.models.load_model(model_bin_path)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)


path = ["orbel_chihuahua.jpeg","shamu.png","shamu2.png","Ginny.png","mia.jpg","patil.jpeg"]

plt.figure(figsize=(12, 12))
index = 0
for p in path:
    IMG_SIZE = 224
    size = (IMG_SIZE, IMG_SIZE)
    img = tf.keras.utils.load_img(
        os.path.join(p),
        target_size=size
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    score = predictions = model.predict(img_array)[0]
    #score = tf.nn.softmax(predictions[0])
    ind = np.argpartition(score, -3)[-3:]
    ax = plt.subplot(2, 3, index + 1)
    plt.imshow(img)
    l = ""
    for i in ind:
        l = l + "{}, {:.2f} conf".format(class_names[i], 100 * score[i]) + "\n"
    plt.title(l)
    plt.axis("off")
    index = index + 1
    
plt.show()

history = np.load(os.path.join("resnet_2_history.npy"),allow_pickle=True)
history = history.item()

acc = history['accuracy']
val_acc = history['val_accuracy']

loss = history['loss']
val_loss = history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# def input_preprocess(image, label):
#     image = tf.keras.applications.resnet50.preprocess_input(image)
#     return image, label

# (ds_train, ds_test), metadata = tfds.load(
#     "stanford_dogs",
#     split=["train", "test"],
#     shuffle_files=True,
#     with_info=True,
#     as_supervised=True,
# )
# class_names = metadata.features['label'].names
# NUM_CLASSES = metadata.features["label"].num_classes

# data_augmentation = tf.keras.Sequential([
#   layers.RandomFlip("horizontal"),
#   layers.RandomRotation(0.2),
#   layers.RandomCrop(224, 224)
# ])

# size = (224,224)

# plt.figure(figsize=(10, 10))
# it = iter(ds_test)
# for i in range(0,9):
#     image, label = next(it)
#     image = data_augmentation(image)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(image.numpy().astype("uint8"))
#     plt.axis("off")
# plt.show()

# ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

# fig = tfds.show_examples(ds_test, metadata)
# Iterate over the dataset for preview:
# plt.figure(figsize=(10, 10))
# for images, labels in unbatch:
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
# model.compile(
#     optimizer=optimizer,
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )