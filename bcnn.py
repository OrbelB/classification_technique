# import tensorflow_datasets as tfds
from random import shuffle
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
load_dotenv()
import os

def dot_product(x):
    return keras.backend.batch_dot(x[0], x[1], axes=[1,1]) / x[0].get_shape().as_list()[1] 

def signed_sqrt(x):
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def L2_norm(x, axis=-1):
    return keras.backend.l2_normalize(x, axis=axis)

def build_model():
    tensor_input = keras.layers.Input(shape=[150,150,3])

#   load pre-trained model
    tensor_input = keras.layers.Input(shape=[150,150,3])
    

    
    model_detector = keras.applications.vgg16.VGG16(
                            input_tensor=tensor_input, 
                            include_top=False,
                            weights='imagenet')
    
    model_detector2 = keras.applications.vgg16.VGG16(
                            input_tensor=tensor_input, 
                            include_top=False,
                            weights='imagenet')
    
    
    model_detector2 = keras.models.Sequential(layers=model_detector2.layers)
  
    for i, layer in enumerate(model_detector2.layers):
        layer._name = layer.name  +"_second"

    model2 = keras.models.Model(inputs=[tensor_input], outputs = [model_detector2.layers[-1].output])
                       
    x = model_detector.layers[17].output
    z = model_detector.layers[17].output_shape
    y = model2.layers[17].output
    
    print(model_detector.summary())
    
    print(model2.summary())
#   rehape to (batch_size, total_pixels, filter_size)
    x = keras.layers.Reshape([z[1] * z[2] , z[-1]])(x)
        
    y = keras.layers.Reshape([z[1] * z[2] , z[-1]])(y)
    
#   outer products of x, y
    x = keras.layers.Lambda(dot_product)([x, y])
    
#   rehape to (batch_size, filter_size_vgg_last_layer*filter_vgg_last_layer)
    x = keras.layers.Reshape([z[-1]*z[-1]])(x)
        
#   signed_sqrt
    x = keras.layers.Lambda(signed_sqrt)(x)
        
#   L2_norm
    x = keras.layers.Lambda(L2_norm)(x)

#   FC-Layer

    initializer = tf.keras.initializers.GlorotNormal()
            
    x = keras.layers.Dense(units=120, 
                           kernel_regularizer=keras.regularizers.l2(0.0),
                           kernel_initializer=initializer)(x)

    tensor_prediction = keras.layers.Activation("softmax")(x)

    model_bilinear = keras.models.Model(inputs=[tensor_input],
                                        outputs=[tensor_prediction])
    
    
#   Freeze VGG layers
    for layer in model_detector.layers:
        layer.trainable = False
        

    sgd = keras.optimizers.SGD(lr=1.0, 
                               decay=0.0,
                               momentum=0.9)

    model_bilinear.compile(loss="categorical_crossentropy", 
                           optimizer=sgd,
                           metrics=["categorical_accuracy"])

    model_bilinear.summary()
    
    return model_bilinear



## Loading images and labels

TRAIN_DIR = os.environ.get('TRAIN_DIR')
TEST_DIR = os.environ.get('TEST_DIR')

train_datagen = image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.2,
        horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        subset='training',
        seed=123,
        class_mode='categorical')
val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        subset='validation',
        seed=123,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(150, 150),
        color_mode="rgb",
        shuffle = False,
        class_mode=None,
        batch_size=1)


# class_names = train_generator.class_names;
# (train_images, train_labels) = train_generator;

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

model = build_model()

hist = model.fit_generator(
                train_generator, 
                epochs=20, 
                validation_data=val_generator,
                workers=3,
                verbose=1
            )
        
model.save_weights("./bilinear_weights/val_acc_" + hist.history['val_categorical_accuracy'][-1] +"_"+ str(20)+ ".h5")
    

for layer in model.layers:
    layer.trainable = True

sgd = keras.optimizers.SGD(lr=1e-3, decay=1e-9, momentum=0.9)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["categorical_accuracy"])

hist = model.fit_generator(
                train_generator, 
                epochs=20, 
                validation_data=val_generator,
                workers=3,
                verbose=1
            )

model.save('./model_bilin')
model2 = keras.models.load_model('./model_bilin')

preds = model2.predict_generator(test_generator, verbose=1)