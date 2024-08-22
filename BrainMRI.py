import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam



size = [224, 224]
train_dataset = './train'
test_dataset = './train'

model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in model.layers:
    layer.trainable = False

x = model.output
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output_layer = layers.Dense(2, activation='softmax')(x)


BrainTumorModel = keras.Model(inputs=model.input, outputs=output_layer)

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dataset, target_size=(224, 224),
                                                    batch_size=32, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(test_dataset, target_size=(224, 224),
                                                        batch_size=32, class_mode='categorical')


BrainTumorModel.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

BTfit = BrainTumorModel.fit(train_generator, epochs=30, verbose=1, validation_data=validation_generator)

BrainTumorModel.save('BrainTumorModel_VGG16_Final.h5')
