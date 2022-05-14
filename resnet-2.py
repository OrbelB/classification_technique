import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers

import os
import datetime
import numpy as np



def outer_product(x):
    #Einstein Notation  [batch,1,1,depth] x [batch,1,1,depth] -> [batch,depth,depth]
    phi_I = tf.einsum('ijkm,ijkn->imn',x[0],x[1])
    
    # Reshape from [batch_size,depth,depth] to [batch_size, depth*depth]
    phi_I = tf.reshape(phi_I,[-1,x[0].shape[3]*x[1].shape[3]])
    
    # Divide by feature map size [sizexsize]
    size1 = int(x[1].shape[1])
    size2 = int(x[1].shape[2])
    phi_I = tf.divide(phi_I, size1*size2)
    
    # Take signed square root of phi_I
    y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))
    
    # Apply l2 normalization
    z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=1)
    return z_l2

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

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

base_model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_tensor=inputs
)

base_model.summary()


conv=base_model.get_layer('conv5_block3_2_relu')
d1=Dropout(0.5)(conv.output)
d2=Dropout(0.5)(conv.output)

x = Lambda(outer_product, name='outer_product')([d1,d2])
predictions=Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=predictions)

base_model.trainable = False


MODEL_PATH = "resnet-2-dogs"
checkpoint_path = os.path.join(MODEL_PATH, "save_at_{epoch}")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
]

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

test_data = ds_test
epochs=20
train_data = ds_train
model.summary()

history = model.fit(
    train_data, epochs=epochs, callbacks=callbacks, validation_data=test_data, verbose=2
)

np.save('resnet_2_history.npy',history.history)

model_bin_path = os.path.join("resnet_2_model_bin")
model.save(model_bin_path)