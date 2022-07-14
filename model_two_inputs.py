from model import *
from data import *
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense
# from keras.models import Model
from tensorflow.keras import Sequential

encoder1 = Sequential(
    [
        Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Dropout(0.5)
    ]
)

encoder2 = Sequential(
    [
        Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Dropout(0.2)
    ]
)

decoder = Sequential(
    [
        UpSampling2D(size = (2,2)),
        Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        
        UpSampling2D(size = (2,2)),
        Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        
        UpSampling2D(size = (2,2)),
        Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        
        Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
        Conv2D(1, 1, activation = 'sigmoid')
    ]
)


def MyModel():
    
    # Define two input layers
    image_input1 = Input((256, 256, 3))
    image_input2 = Input((256, 256, 3))

    embedding_1 = encoder1(image_input1)
    embedding_2 = encoder2(image_input2)
    
    merge = concatenate([embedding_1, embedding_2], axis = 3)
    output = decoder(merge)
    
    model = Model(inputs=[image_input1, image_input2], outputs=output)
                  
    model.compile(optimizer = Adam(learning_rate = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # model.summary()

    return model
    
    
if __name__ == '__main__':
    
    model = MyModel()
    model.summary()