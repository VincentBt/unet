import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
from tensorflow.keras.activations import relu

def concatenate_kernels_sizes(kernel_sizes_hor, kernel_sizes_up):
    """
    Joins the kernel sizes (not proper concatenation: instead, concatenating parts of the lists) to create a single list
    """
    assert len(kernel_sizes_hor) == 19
    assert len(kernel_sizes_up) == 4
    kernel_sizes_hor, kernel_sizes_up = list(kernel_sizes_hor), list(kernel_sizes_up)
    kernel_sizes = kernel_sizes_hor[:10]
    kernel_sizes += [kernel_sizes_up[0]]
    kernel_sizes += kernel_sizes_hor[10:12]
    kernel_sizes += [kernel_sizes_up[1]]
    kernel_sizes += kernel_sizes_hor[12:14]
    kernel_sizes += [kernel_sizes_up[2]]
    kernel_sizes += kernel_sizes_hor[14:16]
    kernel_sizes += [kernel_sizes_up[3]]
    kernel_sizes += kernel_sizes_hor[16:19]
    return kernel_sizes

def unet(pretrained_weights = None, input_size = (256,256,1), 
         n_filters=[64,64,128,128,256,256,512,512,1024,1024,512,512,512,256,256,256,128,128,128,64,64,64,2], 
         kernel_sizes_hor=[3]*19, kernel_sizes_up=[2]*4, kernel_sizes_skip=[1]*9,
         resnet_blocks=False
        ):
    """
    n_filters is a list, where each element represents the number of output filters used for the convolutions
    
    kernel_sizes_hor represents the list of kernel sizes for the horizontal convolutional layers (in the U-Net architecture)
    kernel_sizes_up represents the list of kernel sizes for the up convolutional layers (in the U-Net architecture)
    kernel_sizes_skip represents the list of kernel sizes for the potential convolutions in ResNet blocks
    kernel_sizes_hor, kernel_sizes_up and kernel_sizes_skip are lists, where each element of the list can be an integer or a tuple
    
    resnet_blocks indicates whether the convolutional blocks of the original U-Net architecture are replaced with ResNet blocks or not. 
    By default, convolutional blocks are used rather than ResNet blocks
    Note that there are two versions of the ResNet blocks: identity or convolutional - see https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/ and https://medium.com/@nishanksingla/unet-with-resblock-for-semantic-segmentation-dd1766b4ff66  
    """
    kernel_sizes = concatenate_kernels_sizes(kernel_sizes_hor, kernel_sizes_up)
    assert resnet_blocks in [False, 'id_ResNet', 'conv_ResNet']
    # print(len(n_filters), len(kernel_sizes))
    
    inputs = Input(input_size)
    conv1 = Conv2D(n_filters[0], kernel_sizes[0], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    if resnet_blocks in ['id_ResNet', 'conv_ResNet']:
        conv1 = Conv2D(n_filters[1], kernel_sizes[1], activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv1)
        if resnet_blocks == 'conv_ResNet': #convolutional ResNet block
            input_skip = Conv2D(n_filters[1], kernel_sizes_skip[0], activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs) 
        else: #identity ResNet block
            input_skip = inputs
        conv1 = relu(conv1 + input_skip)
    else:
        conv1 = Conv2D(n_filters[1], kernel_sizes[1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(n_filters[2], kernel_sizes[2], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    if resnet_blocks in ['id_ResNet', 'conv_ResNet']:
        conv2 = Conv2D(n_filters[3], kernel_sizes[3], activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv2)
        if resnet_blocks == 'conv_ResNet': #convolutional ResNet block
            pool1_skip = Conv2D(n_filters[3], kernel_sizes_skip[1], activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool1)
        else: #identity ResNet block
            pool1_skip = pool1
        conv2 = relu(conv2 + pool1_skip)
    else:
        conv2 = Conv2D(n_filters[3], kernel_sizes[3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(n_filters[4], kernel_sizes[4], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    if resnet_blocks in ['id_ResNet', 'conv_ResNet']:
        conv3 = Conv2D(n_filters[5], kernel_sizes[5], activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3)
        if resnet_blocks == 'conv_ResNet': #convolutional ResNet block
            pool2_skip = Conv2D(n_filters[5], kernel_sizes_skip[2], activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool2)
        else: #identity ResNet block
            pool2_skip = pool2
        conv3 = relu(conv3 + pool2_skip)
    else:
        conv3 = Conv2D(n_filters[5], kernel_sizes[5], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(n_filters[6], kernel_sizes[6], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    if resnet_blocks in ['id_ResNet', 'conv_ResNet']:
        conv4 = Conv2D(n_filters[7], kernel_sizes[7], activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4)
        if resnet_blocks == 'conv_ResNet': #convolutional ResNet block
            pool3_skip = Conv2D(n_filters[7], kernel_sizes_skip[3], activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool3)
        else: #identity ResNet block
            pool3_skip = pool3
        conv4 = relu(conv4 + pool3_skip)
    else:
        conv4 = Conv2D(n_filters[7], kernel_sizes[7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(n_filters[8], kernel_sizes[8], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    if resnet_blocks in ['id_ResNet', 'conv_ResNet']:
        conv5 = Conv2D(n_filters[9], kernel_sizes[9], activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5)
        if resnet_blocks == 'conv_ResNet': #convolutional ResNet block
            pool4_skip = Conv2D(n_filters[9], kernel_sizes_skip[4], activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool4)
        else: #identity ResNet block
            pool4_skip = pool4
        conv5 = relu(conv5 + pool4_skip)
    else:
        conv5 = Conv2D(n_filters[9], kernel_sizes[9], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    
    up6 = Conv2D(n_filters[10], kernel_sizes[10], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(n_filters[11], kernel_sizes[11], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    if resnet_blocks in ['id_ResNet', 'conv_ResNet']:
        conv6 = Conv2D(n_filters[12], kernel_sizes[12], activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6)
        if resnet_blocks == 'conv_ResNet': #convolutional ResNet block
            merge6_skip = Conv2D(n_filters[12], kernel_sizes_skip[5], activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge6)
        else: #identity ResNet block
            merge6_skip = merge6
        conv6 = relu(conv6 + merge6_skip)
    else:
        conv6 = Conv2D(n_filters[12], kernel_sizes[12], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(n_filters[13], kernel_sizes[13], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(n_filters[14], kernel_sizes[14], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    if resnet_blocks in ['id_ResNet', 'conv_ResNet']:
        conv7 = Conv2D(n_filters[15], kernel_sizes[15], activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv7)
        if resnet_blocks == 'conv_ResNet': #convolutional ResNet block
            merge7_skip = Conv2D(n_filters[15], kernel_sizes_skip[6], activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge7)
        else: #identity ResNet block
            merge7_skip = merge7
        conv7 = relu(conv7 + merge7_skip)
    else:
        conv7 = Conv2D(n_filters[15], kernel_sizes[15], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(n_filters[16], kernel_sizes[16], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(n_filters[17], kernel_sizes[17], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    if resnet_blocks in ['id_ResNet', 'conv_ResNet']:
        conv8 = Conv2D(n_filters[18], kernel_sizes[18], activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv8)
        if resnet_blocks == 'conv_ResNet': #convolutional ResNet block
            merge8_skip = Conv2D(n_filters[18], kernel_sizes_skip[7], activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge8)
        else: #identity ResNet block
            merge8_skip = merge8
        conv8 = relu(conv8 + merge8_skip)
    else:
        conv8 = Conv2D(n_filters[18], kernel_sizes[18], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(n_filters[19], kernel_sizes[19], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(n_filters[20], kernel_sizes[20], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    if resnet_blocks in ['id_ResNet', 'conv_ResNet']:
        conv9 = Conv2D(n_filters[21], kernel_sizes[21], activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)
        if resnet_blocks == 'conv_ResNet': #convolutional ResNet block
            merge9_skip = Conv2D(n_filters[21], kernel_sizes_skip[8], activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge9)
        else: #identity ResNet block
            merge9_skip = merge9
        conv9 = relu(conv9 + merge9_skip)
    else:
        conv9 = Conv2D(n_filters[21], kernel_sizes[21], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(n_filters[22], kernel_sizes[22], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # model = Model(input = inputs, output = conv10) #this line doesn't work with newer versions of Tensorflow
    model = Model(inputs = inputs, outputs = conv10)
    # model_encoder = Model(inputs = inputs, outputs = drop5) #simple way of creating a submodel composed of only the encoder part
    
    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def extract_layers(main_model, starting_layer_ix, ending_layer_ix) :
    """ 
    Creates a sub-model from some layer (starting_layer_ix) to some other layer (ending_layer_ix) from the initial model
    Taken from https://stackoverflow.com/questions/63550042/extract-subnetwork-from-keras-sequential-model
    """
    # create an empty model
    new_model = Sequential()
    for ix in range(starting_layer_ix, ending_layer_ix + 1):
        curr_layer = main_model.get_layer(index=ix)
        # copy this layer over to the new model
        new_model.add(curr_layer)
    return new_model 




if __name__ == '__main__':
    #create a U-Net model (encoder + decoder)
    model = unet()
    
    #create submodel (with encoder part only)
    layer_names = [layer.name for layer in model.layers]
    # print(layer_names)
    starting_layer_ix = layer_names.index('input_1')
    ending_layer_ix = layer_names.index('dropout_1')
    model_encoder = extract_layers(model, starting_layer_ix, ending_layer_ix)
    model_encoder.summary()

    